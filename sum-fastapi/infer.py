import logging
import os
from typing import List

import torch

NUM_THREADS = os.getenv("NUM_THREADS")
NUM_INTEROP_THREADS = os.getenv("NUM_INTEROP_THREADS")

if NUM_INTEROP_THREADS:
    torch.set_num_interop_threads(int(NUM_INTEROP_THREADS))
    logging.warning("set_num_interop_threads %s", NUM_INTEROP_THREADS)

if NUM_THREADS:
    torch.set_num_threads(int(NUM_THREADS))
    logging.warning("set_num_threads %s", NUM_THREADS)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "philschmid/bart-large-cnn-samsum"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)


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


def summarize_batch(
    input_texts: List[str],
    max_new_tokens: int = 200,
    num_beams: int = 4,
    skip_special_tokens: bool = True,
):
    inputs = tokenizer(input_texts, return_tensors="pt")
    gen_tokens = model.generate(
        inputs.input_ids, max_new_tokens=max_new_tokens, num_beams=num_beams
    )
    outputs = tokenizer.batch_decode(
        gen_tokens, skip_special_tokens=skip_special_tokens
    )
    return outputs
