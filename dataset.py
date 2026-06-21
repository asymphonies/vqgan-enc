from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

FSDD_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/tags/v1.0.10.tar.gz"
DATA_ROOT = Path(__file__).parent / "data"
ARCHIVE_PATH = DATA_ROOT / "fsdd.tar.gz"
RECORDINGS_DIR = DATA_ROOT / "recordings"
SAMPLE_RATE = 8000
CLIP_SECONDS = 1.0
CLIP_LENGTH = int(SAMPLE_RATE * CLIP_SECONDS)


def download_and_extract() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if RECORDINGS_DIR.exists() and any(RECORDINGS_DIR.glob("*.wav")):
        print(f"FSDD already present at {RECORDINGS_DIR}")
        return

    if not ARCHIVE_PATH.exists():
        print(f"Downloading FSDD from {FSDD_URL}")
        urllib.request.urlretrieve(FSDD_URL, ARCHIVE_PATH)

    print("Extracting archive")
    with tarfile.open(ARCHIVE_PATH) as tar:
        members = [
            m
            for m in tar.getmembers()
            if "/recordings/" in m.name and m.name.endswith(".wav")
        ]
        for member in members:
            member.name = Path(member.name).name
        tar.extractall(path=RECORDINGS_DIR, members=members)

    n_files = len(list(RECORDINGS_DIR.glob("*.wav")))
    print(f"Extracted {n_files} wav files into {RECORDINGS_DIR}")


@dataclass
class Utterance:
    path: Path
    digit: int
    speaker: str
    index: int


def list_utterances() -> list[Utterance]:
    utterances = []
    for path in sorted(RECORDINGS_DIR.glob("*.wav")):
        digit_str, speaker, index_str = path.stem.split("_")
        utterances.append(Utterance(path, int(digit_str), speaker, int(index_str)))
    return utterances


def fixed_length_crop_or_pad(waveform: np.ndarray, length: int) -> np.ndarray:
    n = waveform.shape[0]
    if n >= length:
        start = (n - length) // 2
        return waveform[start : start + length]
    pad_total = length - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(waveform, (pad_left, pad_right))


class SpokenDigitDataset(Dataset):
    def __init__(self, utterances: list[Utterance], clip_length: int = CLIP_LENGTH):
        self.utterances = utterances
        self.clip_length = clip_length

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> torch.Tensor:
        utt = self.utterances[idx]
        waveform, sr = sf.read(utt.path, dtype="float32")
        assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE}Hz, got {sr}Hz for {utt.path}"
        waveform = fixed_length_crop_or_pad(waveform, self.clip_length)
        peak = np.abs(waveform).max()
        if peak > 1e-6:
            waveform = waveform / peak * 0.95
        return torch.from_numpy(waveform).unsqueeze(0)


def train_test_split(
    utterances: list[Utterance], test_fraction: float = 0.1, seed: int = 0
):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(utterances))
    n_test = int(len(utterances) * test_fraction)
    test_idx, train_idx = indices[:n_test], indices[n_test:]
    train = [utterances[i] for i in train_idx]
    test = [utterances[i] for i in test_idx]
    return train, test


if __name__ == "__main__":
    download_and_extract()
    utterances = list_utterances()
    train, test = train_test_split(utterances)
    print(f"{len(utterances)} total utterances, {len(train)} train, {len(test)} test")
    speakers = sorted({u.speaker for u in utterances})
    print(f"speakers: {speakers}")
