import os

import whisper
from huggingface_hub import hf_hub_download

REPO_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
FILENAME = "pytorch_model.bin"

PATH = "./server/models/embedding/1/"
file = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir=PATH,
    force_download=True,
    local_dir_use_symlinks=False,
)

os.rename(file, f"{PATH}/wespeaker-voxceleb-resnet34-LM.bin")

PATH = "./server/models/whisper/1/"
model = whisper.load_model("large-v3", device="cpu", download_root=PATH)
