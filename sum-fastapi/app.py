import os
import time
from typing import Dict

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, constr, conint

from infer import summarize, summarize_batch

app = FastAPI(docs_url="/")
# security = HTTPBearer()

@app.get("/readyz")
def readyz():
    return True


class InferRequestModel(BaseModel):
    input_text: constr(min_length=1)
    max_new_tokens: conint(gt=0, le=400) = 200
    num_beams: conint(gt=0, le=6) = 4
    skip_special_tokens: bool = True


class InferResponseModel(BaseModel):
    summarized_text: str
    inference_time_seconds: float


class BatchInferRequestModel(BaseModel):
    input_texts: Dict[constr(min_length=1), constr(min_length=1)]
    max_new_tokens: conint(gt=0, le=400) = 200
    num_beams: conint(gt=0, le=4) = 4
    skip_special_tokens: bool = True


class BatchInferResponseModel(BaseModel):
    summarized_texts: Dict[str, str]
    inference_time_seconds: float


def authorize(credentials):
    if credentials.credentials != os.getenv("API_TOKEN"):
        raise HTTPException(status_code=403, detail="Invalid API token")


@app.post("/infer", response_model=InferResponseModel)
def infer(
    request: InferRequestModel, 
    # credentials: HTTPAuthorizationCredentials = Security(security)
):
    # authorize(credentials)
    start_time = time.monotonic()

    summarized_text = summarize(
        input_text=request.input_text,
        max_new_tokens=request.max_new_tokens,
        num_beams=request.num_beams,
        skip_special_tokens=request.skip_special_tokens,
    )

    end_time = time.monotonic()
    inference_time = end_time - start_time

    return InferResponseModel(
        summarized_text=summarized_text,
        inference_time_seconds=inference_time,
    )


@app.post("/batch-infer", response_model=BatchInferResponseModel)
def batch_infer(
    request: BatchInferRequestModel,
    # credentials: HTTPAuthorizationCredentials = Security(security)
):
    # authorize(credentials)
    start_time = time.monotonic()

    if len(request.input_texts) > 5:
        raise HTTPException(
            status_code=400,
            detail=(
                "`input_texts` should contain less than 6 elements."
                f"You have sent {len(request.input_texts)}"
            ),
        )

    input_texts = list(request.input_texts.values())

    summarized_texts = summarize_batch(
        input_texts=input_texts,
        max_new_tokens=request.max_new_tokens,
        num_beams=request.num_beams,
        skip_special_tokens=request.skip_special_tokens,
    )

    end_time = time.monotonic()
    inference_time = end_time - start_time

    return BatchInferResponseModel(
        summarized_texts=dict(zip(request.input_texts.keys(), summarized_texts)),
        inference_time_seconds=inference_time,
    )
