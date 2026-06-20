import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

import src.server as server
from src.selfplay_storage import SelfPlayStorage
from test_selfplay_storage import encode_game


async def request_raw(path: str, body: bytes, headers: list[tuple[bytes, bytes]]):
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "root_path": "",
    }
    receive_sent = False
    messages = []

    async def receive():
        nonlocal receive_sent
        if receive_sent:
            return {"type": "http.disconnect"}
        receive_sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    await server.app(scope, receive, send)

    status = next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )
    response_body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    return status, response_body


async def request_app(path: str, body: bytes, headers: list[tuple[bytes, bytes]]):
    status, response_body = await request_raw(path, body, headers)
    return status, json.loads(response_body)


class FakeInference:
    def __init__(self):
        self.received_states = None

    def predict(self, states):
        self.received_states = states
        batch_size = states.shape[0]
        return {
            "policy_probs": np.zeros((batch_size, server.POLICY_SIZE), dtype=np.float32),
            "value": np.zeros((batch_size,), dtype=np.float32),
            "score": np.zeros((batch_size,), dtype=np.float32),
            "ownership_probs": np.zeros(
                (
                    batch_size,
                    server.OWNERSHIP_CHANNELS,
                    server.BOARD_SIZE,
                    server.BOARD_SIZE,
                ),
                dtype=np.float32,
            ),
        }


class PredictApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_predict_routes_to_named_models(self):
        original_inferencers = server.inferencers
        model_a = FakeInference()
        model_b = FakeInference()
        server.inferencers = {"a": model_a, "b": model_b}
        states = np.zeros(
            (1, server.STATE_CHANNELS, server.BOARD_SIZE, server.BOARD_SIZE),
            dtype=server.FLOAT32,
        )

        try:
            status_a, response_a = await request_raw(
                "/predict/a",
                states.tobytes(),
                [
                    (b"content-type", b"application/octet-stream"),
                    (b"x-batch-size", b"1"),
                ],
            )
            status_b, response_b = await request_raw(
                "/predict/b",
                states.tobytes(),
                [
                    (b"content-type", b"application/octet-stream"),
                    (b"x-batch-size", b"1"),
                ],
            )
        finally:
            server.inferencers = original_inferencers

        expected_bytes = (
            server.POLICY_SIZE
            + 1
            + 1
            + server.OWNERSHIP_CHANNELS * server.BOARD_SIZE * server.BOARD_SIZE
        ) * server.FLOAT32.itemsize
        self.assertEqual(status_a, 200)
        self.assertEqual(status_b, 200)
        self.assertEqual(len(response_a), expected_bytes)
        self.assertEqual(len(response_b), expected_bytes)
        self.assertIsNotNone(model_a.received_states)
        self.assertIsNotNone(model_b.received_states)

    async def test_predict_uses_default_model(self):
        original_inferencers = server.inferencers
        model_a = FakeInference()
        model_b = FakeInference()
        server.inferencers = {"a": model_a, "b": model_b}
        states = np.zeros(
            (1, server.STATE_CHANNELS, server.BOARD_SIZE, server.BOARD_SIZE),
            dtype=server.FLOAT32,
        )

        try:
            status, response = await request_raw(
                "/predict",
                states.tobytes(),
                [
                    (b"content-type", b"application/octet-stream"),
                    (b"x-batch-size", b"1"),
                ],
            )
        finally:
            server.inferencers = original_inferencers

        expected_bytes = (
            server.POLICY_SIZE
            + 1
            + 1
            + server.OWNERSHIP_CHANNELS * server.BOARD_SIZE * server.BOARD_SIZE
        ) * server.FLOAT32.itemsize
        self.assertEqual(status, 200)
        self.assertEqual(len(response), expected_bytes)
        self.assertIsNotNone(model_a.received_states)
        self.assertIsNone(model_b.received_states)

    async def test_rejects_unknown_predict_model(self):
        status, response = await request_app(
            "/predict/c",
            b"",
            [(b"x-batch-size", b"1")],
        )

        self.assertEqual(status, 404)
        self.assertEqual(response["detail"], "unknown model")

    async def test_rejects_missing_batch_size(self):
        status, response = await request_app(
            "/predict/a",
            b"",
            [(b"content-type", b"application/octet-stream")],
        )

        self.assertEqual(status, 400)
        self.assertEqual(response["detail"], "X-Batch-Size is required")


class SelfPlayApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_saves_game(self):
        original_storage = server.selfplay_storage
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "selfplay.h5"
            server.selfplay_storage = SelfPlayStorage(path)
            try:
                status, response = await request_app(
                    "/selfplay/game",
                    encode_game(2),
                    [
                        (b"content-type", b"application/octet-stream"),
                        (b"x-sample-count", b"2"),
                    ],
                )
            finally:
                server.selfplay_storage = original_storage

            self.assertEqual(status, 200)
            self.assertEqual(response, {"samples": 2, "status": "written"})
            with h5py.File(path, "r") as h5_file:
                self.assertEqual(h5_file["features"].shape[0], 2)

    async def test_rejects_missing_sample_count(self):
        status, response = await request_app(
            "/selfplay/game",
            b"",
            [(b"content-type", b"application/octet-stream")],
        )

        self.assertEqual(status, 400)
        self.assertEqual(response["error"], "invalid_sample_count")


if __name__ == "__main__":
    unittest.main()
