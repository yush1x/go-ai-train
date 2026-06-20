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
WEIGHTS_PATH = PROJECT_ROOT / "data/weights/go_net.pt"  # 配置模型参数
SELFPLAY_PATH = PROJECT_ROOT / "data/selfplay/selfplay.h5"   # 数据保存位置

logger = logging.getLogger(__name__)

app = FastAPI()
infer = GoInference(WEIGHTS_PATH)
selfplay_storage = SelfPlayStorage(SELFPLAY_PATH)


def decode_and_save_selfplay(body: bytes, sample_count: int) -> None:
    game = decode_game(body, sample_count)
    selfplay_storage.append(game)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request, x_batch_size: int = Header(alias="X-Batch-Size")):
    if x_batch_size <= 0:
        raise HTTPException(status_code=400, detail="X-Batch-Size must be positive")

    body = await request.body()
    expected_values = x_batch_size * STATE_CHANNELS * BOARD_SIZE * BOARD_SIZE
    expected_bytes = expected_values * FLOAT32.itemsize
    if len(body) != expected_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"expected {expected_bytes} bytes, got {len(body)} bytes",
        )

    states = np.frombuffer(body, dtype=FLOAT32).copy().reshape(
        x_batch_size,
        STATE_CHANNELS,
        BOARD_SIZE,
        BOARD_SIZE,
    )
    outputs = infer.predict(states)

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
