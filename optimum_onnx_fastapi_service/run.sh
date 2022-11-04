set -ex

MODEL="philschmid/bart-large-cnn-samsum"
ONNX_MODEL_PATH="./ort_onnx/"

python download_and_convert_model.py --model $MODEL --output-dir $ONNX_MODEL_PATH

ONNX_MODEL_PATH=$ONNX_MODEL_PATH gunicorn \
    app:app \
    --timeout 120 \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile -

