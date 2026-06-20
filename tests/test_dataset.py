import tempfile
import unittest
from pathlib import Path

from src.dataset import SelfPlayDataset
from src.selfplay_storage import SelfPlayStorage, decode_game
from test_selfplay_storage import encode_game


class SelfPlayDatasetTest(unittest.TestCase):
    def test_reads_multiple_h5_files_from_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first_path = directory / "selfplay_001.h5"
            second_path = directory / "selfplay_002.h5"

            SelfPlayStorage(first_path).append(decode_game(encode_game(2), 2))
            SelfPlayStorage(second_path).append(decode_game(encode_game(3), 3))

            dataset = SelfPlayDataset(directory)

            self.assertEqual(len(dataset), 5)
            features, policy, value, score, ownership = dataset[4]
            self.assertEqual(features.shape, (9, 19, 19))
            self.assertEqual(policy.shape, (362,))
            self.assertEqual(ownership.shape, (19, 19))
            self.assertEqual(value.item(), 1.0)
            self.assertEqual(score.item(), 2.0)


if __name__ == "__main__":
    unittest.main()
