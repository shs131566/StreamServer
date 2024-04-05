import json

import numpy as np
import triton_python_backend_utils as pb_utils
from loguru import logger
from translater import Translator


class TritonPythonModel:
    def initialize(self, args):
        self.device = "cuda" if args["model_instance_kind"] == "GPU" else "cpu"
        self.device_id = 0 if args["model_instance_device_id"] else None
        self.model_path = "/models/nmt/1/ctrans_fp16"
        self.translator = Translator(
            model_path=self.model_path,
            device=self.device,
            device_index=[self.device_id],
            max_length=200,
            batch_size=32,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                query, src_lang, tgt_lang = self.extract_request_details(request)
                if query == "":
                    responses.append(self.empty_response())
                    continue

                query_list = [query]
                query_order = self.identify_query_order(query_list)
                single_lines, multi_lines = self.separate_lines(query_list, query_order)

                translated_lines = self.translate_lines(
                    single_lines, multi_lines, src_lang, tgt_lang, query_order
                )
                response = self.construct_response(translated_lines)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to execute request: {e}")

        return responses

    def extract_request_details(self, request):
        query = (
            pb_utils.get_input_tensor_by_name(request, "query")
            .as_numpy()[0]
            .decode("utf-8")
        )
        src_lang = (
            pb_utils.get_input_tensor_by_name(request, "src_lang")
            .as_numpy()[0]
            .decode("utf-8")
        )
        tgt_lang = (
            pb_utils.get_input_tensor_by_name(request, "tgt_lang")
            .as_numpy()[0]
            .decode("utf-8")
        )
        return query, src_lang, tgt_lang

    def empty_response(self):
        return self.construct_response(
            [
                {
                    "translated": "",
                    "score": 0,
                    "tgt_tok_len": 0,
                    "src_chr_len": 0,
                    "src_tok_len": 0,
                }
            ]
        )

    def identify_query_order(self, queries):
        return [
            "single" if len(q.split("\n")) == 1 and q.split("\n") != [""] else "multi"
            for q in queries
        ]

    def separate_lines(self, queries, query_order):
        single_lines = [
            line.strip()
            for line, order in zip(queries, query_order)
            if order == "single"
        ]
        multi_lines = [
            line.strip().split("\n")
            for line, order in zip(queries, query_order)
            if order == "multi"
        ]
        return single_lines, multi_lines

    def translate_lines(
        self, single_lines, multi_lines, src_lang, tgt_lang, query_order
    ):
        if set(query_order) == {"single"}:
            translated_single_lines = self.translator.generate(
                single_lines, src_lang=src_lang, tgt_lang=tgt_lang
            )
            return translated_single_lines

        translated_multi_lines = [
            self.process_multi_lines(
                self.translate_multi_lines(line, src_lang=src_lang, tgt_lang=tgt_lang)
            )
            for line in multi_lines
        ]

        if set(query_order) == {"multi"}:
            return translated_multi_lines

        # Mixed single and multi lines
        return self.merge_translated_lines(
            single_lines, translated_multi_lines, src_lang, tgt_lang, query_order
        )

    def merge_translated_lines(
        self, single_lines, multi_lines, src_lang, tgt_lang, query_order
    ):
        translated_single_lines = self.translator.generate(
            single_lines, src_lang=src_lang, tgt_lang=tgt_lang
        )
        hypotheses, single_index = [], 0
        for sent_type in query_order:
            if sent_type == "single":
                hypotheses.append(translated_single_lines[single_index])
                single_index += 1
            else:
                hypotheses.append(multi_lines.pop(0))
        return hypotheses

    def construct_response(self, output):
        return pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "translated_txt",
                    np.array([json.dumps(output)], dtype=np.string_),
                )
            ]
        )

    def translate_multi_lines(self, multi_lines, src_lang, tgt_lang):
        while "" in multi_lines:
            multi_lines.remove("")
        return self.translator.generate(
            multi_lines, src_lang=src_lang, tgt_lang=tgt_lang
        )

    def process_multi_lines(self, translated_multi_lines):
        result = {}
        for d in translated_multi_lines:
            for key, value in d.items():
                if key == "translated":
                    result[key] = (
                        result.get(key, "") + f"\n{value}" if key in result else value
                    )
                else:
                    result[key] = result.get(key, 0) + value
        return result
