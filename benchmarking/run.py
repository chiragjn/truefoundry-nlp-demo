import os
import time
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import mlfoundry

MODEL_PATH = "philschmid/bart-large-cnn-samsum"
MACHINE_TYPE = os.environ["MACHINE_TYPE"]
CPU_LIMIT = float(os.getenv("CPU_LIMIT", -1) or -1)
MEM_LIMIT = float(os.getenv("MEM_LIMIT", -1) or -1)
CPU_COUNT = os.cpu_count()


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

def set_random_seed(seed_value: int, use_cuda: bool = False):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
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


def bench():
    with open('data.txt', 'r') as f:
        text = f.read()

    client = mlfoundry.get_client()
    run_name_parts = [
        "machine",
        MACHINE_TYPE,
        "cpu",
        str(CPU_LIMIT).replace('.', '-'),
        "mem",
        str(MEM_LIMIT).replace('.', '-'),
    ]
    run = client.create_run(
        project_name="sum-bench",
        run_name='-'.join(run_name_parts)
    )
    run.log_params({
        "model_path": MODEL_PATH,
        "machine_type": MACHINE_TYPE,
    })

    run.log_metrics({
        "cpu_limit": CPU_LIMIT,
        "mem_limit": MEM_LIMIT,
        "cpu_count": CPU_COUNT,
    })
    
    for max_length in (64, 128, 200):
        for num_beams in (1, 2, 4): 
            print(f"Running max length {max_length} and beams {num_beams}")
            set_random_seed(42)
            times = []
            for i in range(5):
                start = time.monotonic()
                summarize(text, max_new_tokens=max_length, num_beams=num_beams)
                end = time.monotonic()
                times.append(end - start)
            min_time = min(times)
            max_time = max(times)
            avg_time = float(sum(times)) / float(len(times))
            ns = f"max_len_{max_length}_beams_{num_beams}"
            run.log_metrics({
                f"{ns}/avg": avg_time,
                f"{ns}/min": min_time,
                f"{ns}/max": max_time,
            })

    run.end()

if __name__ == '__main__':
    bench()
