from pathlib import Path
import logging
import sys

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import GoInference
from selfplay_storage import SelfPlayDataError, SelfPlayStorage, decode_game


BOARD_SIZE = 19
STATE_CHANNELS = 9
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE + 1
OWNERSHIP_CHANNELS = 2
FLOAT32 = np.dtype("<f4")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 模型权重配置。服务启动时会全部加载，/predict/a 和 /predict/b 分别使用对应模型。
MODEL_WEIGHTS = {
    "a": PROJECT_ROOT / "data/weights/s.pt",
    "b": PROJECT_ROOT / "data/weights/s.pt",
}
DEFAULT_MODEL = "a"

SELFPLAY_PATH = PROJECT_ROOT / "data/selfplay/d.h5"   # 数据保存位置

logger = logging.getLogger(__name__)

app = FastAPI()
inferencers: dict[str, GoInference] = {}
selfplay_storage = SelfPlayStorage(SELFPLAY_PATH)


def load_models() -> None:
    if inferencers:
        return

    for model_name, weights_path in MODEL_WEIGHTS.items():
        inferencers[model_name] = GoInference(weights_path)


def decode_and_save_selfplay(body: bytes, sample_count: int) -> None:
    game = decode_game(body, sample_count)
    selfplay_storage.append(game)


def parse_batch_size(x_batch_size: str | None) -> int:
    if x_batch_size is None:
        raise HTTPException(status_code=400, detail="X-Batch-Size is required")
    try:
        batch_size = int(x_batch_size)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="X-Batch-Size must be an integer",
        ) from exc
    if batch_size <= 0:
        raise HTTPException(status_code=400, detail="X-Batch-Size must be positive")
    return batch_size


@app.on_event("startup")
def startup() -> None:
    load_models()


@app.get("/health")
def health():
    return {"status": "ok", "models": sorted(MODEL_WEIGHTS)}


async def predict_with_model(
    model_name: str,
    request: Request,
    x_batch_size: str | None,
) -> Response:
    if model_name not in MODEL_WEIGHTS:
        raise HTTPException(status_code=404, detail="unknown model")
    batch_size = parse_batch_size(x_batch_size)
    if model_name not in inferencers:
        raise HTTPException(status_code=503, detail="model is not loaded")

    body = await request.body()
    expected_values = batch_size * STATE_CHANNELS * BOARD_SIZE * BOARD_SIZE
    expected_bytes = expected_values * FLOAT32.itemsize
    if len(body) != expected_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"expected {expected_bytes} bytes, got {len(body)} bytes",
        )

    states = np.frombuffer(body, dtype=FLOAT32).copy().reshape(
        batch_size,
        STATE_CHANNELS,
        BOARD_SIZE,
        BOARD_SIZE,
    )
    outputs = inferencers[model_name].predict(states)

    response_values = np.concatenate([
        outputs["policy_probs"].reshape(-1),
        outputs["value"].reshape(-1),
        outputs["score"].reshape(-1),
        outputs["ownership_probs"].reshape(-1),
    ]).astype(FLOAT32, copy=False)

    return Response(
        content=response_values.tobytes(),
        media_type="application/octet-stream",
    )


@app.post("/predict/{model_name}")
async def predict(
    model_name: str,
    request: Request,
    x_batch_size: str | None = Header(default=None, alias="X-Batch-Size"),
):
    return await predict_with_model(model_name, request, x_batch_size)


@app.post("/predict")
async def predict_default(
    request: Request,
    x_batch_size: str | None = Header(default=None, alias="X-Batch-Size"),
):
    return await predict_with_model(DEFAULT_MODEL, request, x_batch_size)


@app.post("/selfplay/game")
async def save_selfplay_game(
    request: Request,
    x_sample_count: str | None = Header(default=None, alias="X-Sample-Count"),
):
    try:
        if x_sample_count is None:
            raise SelfPlayDataError(
                "invalid_sample_count",
                "X-Sample-Count is required",
            )
        try:
            sample_count = int(x_sample_count)
        except ValueError as exc:
            raise SelfPlayDataError(
                "invalid_sample_count",
                "X-Sample-Count must be an integer",
            ) from exc

        body = await request.body()
        await run_in_threadpool(decode_and_save_selfplay, body, sample_count)
    except SelfPlayDataError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": exc.code, "message": exc.message},
        )
    except Exception:
        logger.exception("failed to save selfplay data")
        return JSONResponse(
            status_code=500,
            content={
                "error": "storage_failure",
                "message": "failed to save selfplay data",
            },
        )

    return {"samples": sample_count, "status": "written"}
