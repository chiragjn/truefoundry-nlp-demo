import argparse

from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="philschmid/bart-large-cnn-samsum")
    parser.add_argument("--output-dir", type=str, default="./ort_onnx/")
    args = parser.parse_args()
    model = ORTModelForSeq2SeqLM.from_pretrained(args.model, from_transformers=True)
    model.save_pretrained(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(args.output_dir)
