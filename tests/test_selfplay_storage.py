import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.selfplay_storage import (
    BOARD_SIZE,
    BYTES_PER_SAMPLE,
    FEATURE_CHANNELS,
    POLICY_SIZE,
    SelfPlayDataError,
    SelfPlayStorage,
    decode_game,
)


def encode_game(sample_count: int, *, non_finite: bool = False) -> bytes:
    features = np.arange(
        sample_count * FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE,
        dtype="<f4",
    ).reshape(sample_count, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    policy = np.full((sample_count, POLICY_SIZE), 1 / POLICY_SIZE, dtype="<f4")
    value = np.ones(sample_count, dtype="<f4")
    score = np.arange(sample_count, dtype="<f4")
    ownership = np.zeros(
        (sample_count, BOARD_SIZE, BOARD_SIZE),
        dtype="i1",
    )

    if non_finite:
        features[0, 0, 0, 0] = np.nan

    return b"".join(
        values.tobytes()
        for values in (features, policy, value, score, ownership)
    )


class DecodeGameTest(unittest.TestCase):
    def test_decodes_protocol_layout(self):
        body = encode_game(2)

        game = decode_game(body, 2)

        self.assertEqual(len(body), 2 * BYTES_PER_SAMPLE)
        self.assertEqual(game.features.shape, (2, 9, 19, 19))
        self.assertEqual(game.policy.shape, (2, 362))
        self.assertEqual(game.value.shape, (2,))
        self.assertEqual(game.score.shape, (2,))
        self.assertEqual(game.ownership.shape, (2, 19, 19))

    def test_rejects_invalid_body_size(self):
        with self.assertRaises(SelfPlayDataError) as context:
            decode_game(b"short", 1)

        self.assertEqual(context.exception.code, "invalid_body_size")

    def test_rejects_non_positive_sample_count(self):
        with self.assertRaises(SelfPlayDataError) as context:
            decode_game(b"", 0)

        self.assertEqual(context.exception.code, "invalid_sample_count")

    def test_rejects_non_finite_float(self):
        with self.assertRaises(SelfPlayDataError) as context:
            decode_game(encode_game(1, non_finite=True), 1)

        self.assertEqual(context.exception.code, "non_finite_float")


class SelfPlayStorageTest(unittest.TestCase):
    def test_appends_games_to_training_datasets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "selfplay.h5"
            storage = SelfPlayStorage(path)

            storage.append(decode_game(encode_game(2), 2))
            storage.append(decode_game(encode_game(3), 3))

            with h5py.File(path, "r") as h5_file:
                self.assertEqual(h5_file["features"].shape, (5, 9, 19, 19))
                self.assertEqual(h5_file["policy"].shape, (5, 362))
                self.assertEqual(h5_file["value"].shape, (5,))
                self.assertEqual(h5_file["score"].shape, (5,))
                self.assertEqual(h5_file["ownership"].shape, (5, 19, 19))
                self.assertEqual(h5_file["features"].dtype, np.dtype("<f4"))
                self.assertEqual(h5_file["ownership"].dtype, np.dtype("i1"))


if __name__ == "__main__":
    unittest.main()
