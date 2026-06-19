import json
import tempfile
import unittest
from pathlib import Path

import h5py

import src.server as server
from src.selfplay_storage import SelfPlayStorage
from test_selfplay_storage import encode_game


async def request_app(path: str, body: bytes, headers: list[tuple[bytes, bytes]]):
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
    return status, json.loads(response_body)


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
