import itertools
import os
import time
from functools import reduce
from statistics import mean
from typing import List

import ctranslate2
import nltk
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration


class Translator:
    """
    Allow detokenizing sequences in batchs
    **Performance tips**
    Below are some general recommendations to further improve performance. Many of these recommendations were used in the WNGT 2020 efficiency task submission.
    Set the compute type to "auto" to automatically select the fastest execution path on the current system
    Reduce the beam size to the minimum value that meets your quality requirement
    When using a beam size of 1, keep return_scores disabled if you are not using prediction scores: the final softmax layer can be skipped
    Set max_batch_size and pass a larger batch to *_batch methods: the input sentences will be sorted by length and split by chunk of max_batch_size elements for improved efficiency
    Prefer the "tokens" batch_type to make the total number of elements in a batch more constant
    Consider using {ref}translation:dynamic vocabulary reduction for translation

    <On CPU>
    Use an Intel CPU supporting AVX512
    If you are processing a large volume of data, prefer increasing inter_threads over intra_threads and use stream methods (methods whose name ends with _file or _iterable)
    Avoid the total number of threads inter_threads * intra_threads to be larger than the number of physical cores
    For single core execution on Intel CPUs, consider enabling packed GEMM (set the environment variable CT2_USE_EXPERIMENTAL_PACKED_GEMM=1)

    <On GPU>
    Use a larger batch size
    Use a NVIDIA GPU with Tensor Cores (Compute Capability >= 7.0)
    Pass multiple GPU IDs to device_index to execute on multiple GPUs
    """

    def __init__(
        self,
        model_path=None,
        device="cuda",
        device_index=[0],
        max_length=200,
        batch_size=8,
        logger=None,
    ):
        if logger is None:
            import logging

            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)

            formatter = logging.Formatter(
                "%(asctime)s;%(name)s - %(levelname)s - %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
            streamhandler = logging.StreamHandler()
            streamhandler.setFormatter(formatter)
            logger.addHandler(streamhandler)

        try:
            if device == "cuda":
                if not isinstance(device_index, list):
                    msg = "Device index must be list type in cuda mode"
                    logger.error(msg)
                    raise TypeError(msg)
                else:
                    inter_threads = len(device_index)
            elif device == "cpu":
                inter_threads = os.cpu_count()
            else:
                msg = "Supported device type: cpu or cuda"
                logger.error(msg)
                raise ValueError(msg)

            while inter_threads >= 1:
                try:
                    intra_threads = max(2, os.cpu_count() // inter_threads)
                    self.model = ctranslate2.Translator(
                        model_path,
                        inter_threads=inter_threads,
                        intra_threads=intra_threads,
                        device="cpu" if device != "cuda" else "cuda",
                        device_index=device_index if device == "cuda" else 0,
                    )
                    logger.info(
                        f"C-Translator is loaded on {'cpu' if device != 'cuda' else 'cuda'} runtime with inter_threads({inter_threads}) and intra_threads({intra_threads})"
                    )
                    break
                except MemoryError:
                    inter_threads //= 2
                    logger.error(
                        f"Reducing inter_threads to {inter_threads} due to memory error."
                    )
        except Exception as ex:
            logger.error(ex)
            inter_threads = os.cpu_count()

            while inter_threads >= 1:
                try:
                    intra_threads = max(2, os.cpu_count() // inter_threads)
                    self.model = ctranslate2.Translator(
                        model_path,
                        inter_threads=inter_threads,
                        intra_threads=max(2, os.cpu_count() // inter_threads),
                        device="cpu",
                        device_index=0,
                    )
                    logger.info(
                        f"C-Translator is loaded on cpu runtime with inter_threads({inter_threads}) and intra_threads({intra_threads})"
                    )
                    break
                except MemoryError:
                    inter_threads //= 2
                    logger.error(
                        f"Reducing inter_threads to {inter_threads} due to memory error."
                    )

        self.meta = {
            "device": self.model.device,
            "inter_threads": inter_threads,
            "intra_threads": intra_threads,
        }

        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        self.max_length = max_length
        self.batch_size = batch_size

    def do_reassemble(self, tokenized_sents):
        """
        Recombine sentences according to max length
        Args:
            tokenized_sents : a sentence separated by two or more sentences
        Return:
            segments : Recombined sentence elements from splitted sentence
        """
        input_len, start_ix = 0, 0
        segments = []

        for i, t_s in enumerate(tokenized_sents):
            input_len += len(t_s)
            if (
                i + 1 == len(tokenized_sents)
                or input_len + len(tokenized_sents[i + 1]) > self.max_length
            ):
                end_ix = i + 1 if i + 1 == len(tokenized_sents) else i + 1
                seg = list(itertools.chain(*tokenized_sents[start_ix:end_ix]))
                segments.append(
                    {
                        "seg": [self.tokenizer.src_lang]
                        + seg
                        + [self.tokenizer.eos_token],
                        "src_chr_len": len(
                            self.tokenizer.convert_tokens_to_string(seg)
                        ),
                        "src_tok_len": len(seg),
                    }
                )
                input_len = 0
                start_ix = i + 1

        # print(f"Divided input's length : {len(segments)}")
        # print(f"Segments reassembled : {segments}")

        return segments

    def convert_to_inputs(self, src_sent):
        """
        Create Ctranslate input format according to the number of sent element
        """

        def divide_inputs(l, n):
            for i in range(0, len(l), n):
                yield {
                    "seg": [self.tokenizer.src_lang]
                    + l[i : i + n]
                    + [self.tokenizer.eos_token],
                    "src_chr_len": len(
                        self.tokenizer.convert_tokens_to_string(l[i : i + n])
                    ),
                    "src_tok_len": len(l[i : i + n]),
                }

        splitted_sents = nltk.sent_tokenize(src_sent)
        # print(f"\nSplitted Length of input : {len(splitted_sents)}")

        if len(splitted_sents) == 1:
            # print(f"Do simply segmentation by max decoding length")

            tokenized_sent = self.tokenizer.tokenize(src_sent)
            segments = list(divide_inputs(tokenized_sent, self.max_length))

            # print(f"Divided input's length : {len(segments)}")
            # print(f"Segments simply splitted : {segments}")

            return segments

        else:
            # print(f"Do reassembling setns by max decoding length")

            return self.do_reassemble(
                list(map(self.tokenizer.tokenize, splitted_sents))
            )

    def __detokenize(self, x):
        """
        Args:
            x : <class 'ctranslate2.translator.TranslationResult'>
        """
        return self.tokenizer.convert_tokens_to_string(x.hypotheses[0][1:]).replace(
            "<unk>", ""
        )

    def __scoring(self, x):
        """
        Args:
            x : <class 'ctranslate2.translator.TranslationResult'>
        """
        return x.scores[0]

    def __get_tgt_lengths(self, x):
        """
        Args:
            x : <class 'ctranslate2.translator.TranslationResult'>
        """
        return len(x.hypotheses[0][1:])

    def __post_process(self, TranslationResults):
        """
        Args:
            TranslationResults : List of <class 'ctranslate2.translator.TranslationResult'>
        """
        return {
            "translated": " ".join(list(map(self.__detokenize, TranslationResults))),
            "score": mean(list(map(self.__scoring, TranslationResults))),
            "tgt_tok_len": sum(list(map(self.__get_tgt_lengths, TranslationResults))),
        }

    def post_process(self, translated_tokens):
        """
        Detokenize translated tokens
        """
        start = time.time()
        result = list(map(self.__post_process, translated_tokens))
        end = time.time()
        # print(f"\nElapsed for detokenizing : {end-start}")
        return result

    def generate(
        self, src_sents: List[str], src_lang: str, tgt_lang: str, return_scores=True
    ):
        """
        Main function of batch generation
        Args :
            src_sents : [sent1, sent2, ...sent#n]
            src_lang : ex) 'en_XX'
            tgt_lang : ex) 'ko_KR'
            return_scores : Boolean
        Returns :
            preds :
                [{'translated': 'Hello', 'score': -1.49609375, 'tgt_tok_len': 1, 'src_chr_len': 5, 'src_tok_len': 4}, {'translated': 'Nice to meet you.', 'score': -1.357421875, 'tgt_tok_len': 6, 'src_chr_len': 5, 'src_tok_len': 4}, ...]
        """
        self.tokenizer.src_lang = src_lang

        def batch(iterable, n=1):
            """
            Generator configuring a list of sentences by a predefined batch size
            """
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx : min(ndx + n, l)]

        sentence_batch = (
            batch(src_sents, self.batch_size)
            if isinstance(src_sents, list)
            else batch([src_sents], self.batch_size)
        )
        results = []

        for i, src_sent in enumerate(sentence_batch):
            # Apply 'seperation or recombination for sents module' in parallel
            src_sent = [s for s in src_sent if s.strip() != ""]
            converted_inputs = list(map(self.convert_to_inputs, src_sent))

            inputs, src_chr_len, src_tok_len = (
                [[e["seg"] for e in c_i] for c_i in converted_inputs],
                [[e["src_chr_len"] for e in c_i] for c_i in converted_inputs],
                [[e["src_tok_len"] for e in c_i] for c_i in converted_inputs],
            )

            # Apply 'batch translation module' for inputs in parallel
            start = time.time()
            translated_tokens = map(
                lambda source: self.model.translate_batch(
                    source=source,
                    target_prefix=[[tgt_lang]] * len(source),
                    max_batch_size=self.batch_size,
                    batch_type="examples",
                    beam_size=2,
                    max_decoding_length=self.max_length,
                    asynchronous=False,
                    return_scores=return_scores,
                ),
                inputs,
            )
            end = time.time()
            # print(f"\nElapsed time for translation : {end-start}")
            # Apply 'detokenize module for translated tokens' in parallel
            pred = self.post_process(translated_tokens)

            assert len(pred) == len(src_chr_len) == len(src_tok_len)

            for i, p in enumerate(pred):
                p.update(
                    {
                        "src_chr_len": sum(src_chr_len[i]),
                        "src_tok_len": sum(src_tok_len[i]),
                    }
                )
            results.extend(pred)

        return results
