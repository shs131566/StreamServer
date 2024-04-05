import json

import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput

# 사용 예시
server_url = "localhost:8001"
model_name = "nmt"
query = "안녕하세요 반갑습니다"
src_lang = "ko_KR"
tgt_lang = "en_XX"


triton_client = InferenceServerClient(url=server_url)

inputs = [
    InferInput(name="query", shape=[1], datatype="BYTES"),
    InferInput(name="src_lang", shape=[1], datatype="BYTES"),
    InferInput(name="tgt_lang", shape=[1], datatype="BYTES"),
]

inputs[0].set_data_from_numpy(np.array([query.encode("utf-8")], dtype=object))
inputs[1].set_data_from_numpy(np.array([src_lang.encode("utf-8")], dtype=object))
inputs[2].set_data_from_numpy(np.array([tgt_lang.encode("utf-8")], dtype=object))

response = triton_client.infer(
    model_name=model_name,
    inputs=inputs,
    timeout=360000,
)

result = response.as_numpy("translated_txt")
print("Translated Text:", json.loads(result[0]))
