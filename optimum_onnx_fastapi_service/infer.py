import os

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

ONNXMODEL_PATH = os.environ["ONNX_MODEL_PATH"]

tokenizer = AutoTokenizer.from_pretrained(ONNXMODEL_PATH)
model = ORTModelForSeq2SeqLM.from_pretrained(ONNXMODEL_PATH)


def summarize(
    input_text: str,
    max_new_tokens: int = 200,
    num_beams: int = 4,
    skip_special_tokens: bool = True,
):
    inputs = tokenizer(input_text, return_tensors="pt").input_ids
    gen_tokens = model.generate(
        inputs, max_new_tokens=max_new_tokens, num_beams=num_beams
    )
    outputs = tokenizer.batch_decode(
        gen_tokens, skip_special_tokens=skip_special_tokens
    )
    return outputs[0]
