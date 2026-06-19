from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import h5py
import numpy as np


BOARD_SIZE = 19
FEATURE_CHANNELS = 9
POLICY_SIZE = BOARD_SIZE * BOARD_SIZE + 1

FLOAT32 = np.dtype("<f4")
INT8 = np.dtype("i1")

FEATURE_VALUES_PER_SAMPLE = FEATURE_CHANNELS * BOARD_SIZE * BOARD_SIZE
POLICY_VALUES_PER_SAMPLE = POLICY_SIZE
OWNERSHIP_VALUES_PER_SAMPLE = BOARD_SIZE * BOARD_SIZE
BYTES_PER_SAMPLE = (
    FEATURE_VALUES_PER_SAMPLE * FLOAT32.itemsize
    + POLICY_VALUES_PER_SAMPLE * FLOAT32.itemsize
    + FLOAT32.itemsize
    + FLOAT32.itemsize
    + OWNERSHIP_VALUES_PER_SAMPLE * INT8.itemsize
)


class SelfPlayDataError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SelfPlayGame:
    features: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    score: np.ndarray
    ownership: np.ndarray

    @property
    def sample_count(self) -> int:
        return self.features.shape[0]


def validate_sample_count(sample_count: int) -> None:
    if sample_count <= 0:
        raise SelfPlayDataError(
            "invalid_sample_count",
            "X-Sample-Count must be positive",
        )


def decode_game(body: bytes, sample_count: int) -> SelfPlayGame:
    validate_sample_count(sample_count)

    expected_bytes = sample_count * BYTES_PER_SAMPLE
    if len(body) != expected_bytes:
        raise SelfPlayDataError(
            "invalid_body_size",
            f"got {len(body)}, expected {expected_bytes}",
        )

    offset = 0

    def read(dtype: np.dtype, count: int) -> np.ndarray:
        nonlocal offset
        size = dtype.itemsize * count
        values = np.frombuffer(body, dtype=dtype, count=count, offset=offset)
        offset += size
        return values

    features = read(
        FLOAT32,
        sample_count * FEATURE_VALUES_PER_SAMPLE,
    ).reshape(sample_count, FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    policy = read(
        FLOAT32,
        sample_count * POLICY_VALUES_PER_SAMPLE,
    ).reshape(sample_count, POLICY_SIZE)
    value = read(FLOAT32, sample_count)
    score = read(FLOAT32, sample_count)
    ownership = read(
        INT8,
        sample_count * OWNERSHIP_VALUES_PER_SAMPLE,
    ).reshape(sample_count, BOARD_SIZE, BOARD_SIZE)

    for name, values in (
        ("features", features),
        ("policy", policy),
        ("value", value),
        ("score", score),
    ):
        if not np.isfinite(values).all():
            raise SelfPlayDataError(
                "non_finite_float",
                f"{name} contains NaN or infinity",
            )

    return SelfPlayGame(
        features=features,
        policy=policy,
        value=value,
        score=score,
        ownership=ownership,
    )


class SelfPlayStorage:
    _DATASETS = {
        "features": (FLOAT32, (FEATURE_CHANNELS, BOARD_SIZE, BOARD_SIZE)),
        "policy": (FLOAT32, (POLICY_SIZE,)),
        "value": (FLOAT32, ()),
        "score": (FLOAT32, ()),
        "ownership": (INT8, (BOARD_SIZE, BOARD_SIZE)),
    }

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._write_lock = Lock()

    def append(self, game: SelfPlayGame) -> None:
        with self._write_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(self.path, "a") as h5_file:
                datasets = self._get_or_create_datasets(h5_file)
                current_size = self._validate_dataset_lengths(datasets)
                new_size = current_size + game.sample_count

                try:
                    for dataset in datasets.values():
                        dataset.resize(new_size, axis=0)

                    datasets["features"][current_size:new_size] = game.features
                    datasets["policy"][current_size:new_size] = game.policy
                    datasets["value"][current_size:new_size] = game.value
                    datasets["score"][current_size:new_size] = game.score
                    datasets["ownership"][current_size:new_size] = game.ownership
                    h5_file.flush()
                except Exception:
                    for dataset in datasets.values():
                        dataset.resize(current_size, axis=0)
                    h5_file.flush()
                    raise

    def _get_or_create_datasets(
        self,
        h5_file: h5py.File,
    ) -> dict[str, h5py.Dataset]:
        datasets = {}
        for name, (dtype, sample_shape) in self._DATASETS.items():
            if name in h5_file:
                dataset = h5_file[name]
                expected_shape = sample_shape
                if dataset.shape[1:] != expected_shape or dataset.dtype != dtype:
                    raise RuntimeError(
                        f"dataset {name} has incompatible shape or dtype"
                    )
            else:
                dataset = h5_file.create_dataset(
                    name,
                    shape=(0, *sample_shape),
                    maxshape=(None, *sample_shape),
                    chunks=(64, *sample_shape),
                    dtype=dtype,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )
            datasets[name] = dataset
        return datasets

    @staticmethod
    def _validate_dataset_lengths(
        datasets: dict[str, h5py.Dataset],
    ) -> int:
        lengths = {dataset.shape[0] for dataset in datasets.values()}
        if len(lengths) != 1:
            raise RuntimeError("selfplay datasets have inconsistent lengths")
        return lengths.pop()
