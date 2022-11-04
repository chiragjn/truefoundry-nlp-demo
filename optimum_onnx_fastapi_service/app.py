import time

import fastapi
from pydantic import BaseModel

from infer import summarize

app = fastapi.FastAPI(docs_url="/")


@app.get("/readyz")
def readyz():
    return True


class RequestModel(BaseModel):
    input_text: str
    max_new_tokens: int = 200
    num_beams: int = 4
    skip_special_tokens: bool = True


class ResponseModel(BaseModel):
    summarized_text: str
    inference_time_seconds: float


@app.post("/infer", response_model=ResponseModel)
def infer(request: RequestModel):
    start_time = time.monotonic()

    summarized_text = summarize(
        input_text=request.input_text,
        max_new_tokens=request.max_new_tokens,
        num_beams=request.num_beams,
        skip_special_tokens=request.skip_special_tokens,
    )

    end_time = time.monotonic()
    inference_time = end_time - start_time

    return ResponseModel(
        summarized_text=summarized_text,
        inference_time_seconds=inference_time,
    )
