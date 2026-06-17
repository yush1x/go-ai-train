from pathlib import Path
import sys

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, Response

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import GoInference


BOARD_SIZE = 19
STATE_CHANNELS = 9
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE + 1
OWNERSHIP_CHANNELS = 2
FLOAT32 = np.dtype("<f4")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "data/weights/agon_go_net.pt"

app = FastAPI()
infer = GoInference(WEIGHTS_PATH)


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
