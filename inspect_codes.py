from __future__ import annotations

import torch

from dataset import SpokenDigitDataset, list_utterances, train_test_split
from evaluate import load_model
from train import pick_device


def inspect(n_examples: int = 5):
    device = pick_device()
    model = load_model().to(device)

    utterances = list_utterances()
    _, test_utts = train_test_split(utterances)
    dataset = SpokenDigitDataset(test_utts)

    for i in range(min(n_examples, len(test_utts))):
        utt = test_utts[i]
        waveform = dataset[i].unsqueeze(0).to(device)
        codes = model.encode(waveform).squeeze(0).cpu()

        n_frames = codes.shape[0]
        seconds = waveform.shape[-1] / 8000
        bits_per_frame = torch.log2(torch.tensor(float(model.quantizer.codebook_size)))
        kbps = (n_frames / seconds) * bits_per_frame.item() / 1000

        print(f"\nfile: {utt.path.name} (digit {utt.digit}, speaker {utt.speaker})")
        print(f"  {n_frames} code frames over {seconds:.2f}s -> {kbps:.2f} kbps")
        print(f"  unique codes used: {codes.unique().numel()} / {n_frames}")
        print(f"  code sequence: {codes.tolist()}")


if __name__ == "__main__":
    inspect()
