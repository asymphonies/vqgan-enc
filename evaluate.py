from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from dataset import SAMPLE_RATE, list_utterances, train_test_split, SpokenDigitDataset
from model import AudioCodec
from train import CHECKPOINT_PATH, pick_device

OUTPUT_DIR = Path(__file__).parent / "outputs"


def si_sdr(prediction: np.ndarray, target: np.ndarray) -> float:
    target = target - target.mean()
    prediction = prediction - prediction.mean()
    scale = np.dot(prediction, target) / (np.dot(target, target) + 1e-8)
    projection = scale * target
    noise = prediction - projection
    ratio = (np.sum(projection ** 2) + 1e-8) / (np.sum(noise ** 2) + 1e-8)
    return float(10 * np.log10(ratio))


def load_model() -> AudioCodec:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model = AudioCodec(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def evaluate(n_examples_to_save: int = 8):
    device = pick_device()
    model = load_model().to(device)

    utterances = list_utterances()
    _, test_utts = train_test_split(utterances)
    test_loader = DataLoader(SpokenDigitDataset(test_utts), batch_size=1, shuffle=False)

    OUTPUT_DIR.mkdir(exist_ok=True)
    si_sdr_scores, code_usage_total = [], torch.zeros(model.quantizer.codebook_size)

    with torch.no_grad():
        for i, waveform in enumerate(test_loader):
            waveform = waveform.to(device)
            reconstruction, _, codes, _ = model(waveform)

            target_np = waveform.squeeze().cpu().numpy()
            pred_np = reconstruction.squeeze().cpu().numpy()
            si_sdr_scores.append(si_sdr(pred_np, target_np))

            code_usage_total += torch.bincount(codes.flatten(), minlength=model.quantizer.codebook_size).cpu()

            if i < n_examples_to_save:
                utt = test_utts[i]
                sf.write(OUTPUT_DIR / f"{utt.path.stem}_original.wav", target_np, SAMPLE_RATE)
                sf.write(OUTPUT_DIR / f"{utt.path.stem}_reconstructed.wav", pred_np, SAMPLE_RATE)

    probs = code_usage_total / code_usage_total.sum().clamp_min(1)
    entropy_bits = -(probs[probs > 0] * probs[probs > 0].log2()).sum().item()
    max_bits = np.log2(model.quantizer.codebook_size)

    metrics = {
        "n_test_examples": len(test_utts),
        "mean_si_sdr_db": float(np.mean(si_sdr_scores)),
        "median_si_sdr_db": float(np.median(si_sdr_scores)),
        "codebook_entropy_bits": entropy_bits,
        "max_possible_bits": max_bits,
        "bitrate_efficiency": entropy_bits / max_bits,
        "frame_rate_hz": SAMPLE_RATE / model.total_stride,
        "codes_used": int((code_usage_total > 0).sum().item()),
        "codebook_size": model.quantizer.codebook_size,
    }

    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"saved {n_examples_to_save} reconstructed/original wav pairs to {OUTPUT_DIR}")


if __name__ == "__main__":
    evaluate()
