from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import (
    download_and_extract,
    list_utterances,
    train_test_split,
    SpokenDigitDataset,
)
from losses import reconstruction_loss
from model import AudioCodec
from tqdm import tqdm

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "codec.pt"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    vq_loss_weight: float = 1.0,
):
    download_and_extract()
    utterances = list_utterances()
    train_utts, test_utts = train_test_split(utterances)

    train_loader = DataLoader(SpokenDigitDataset(train_utts), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(SpokenDigitDataset(test_utts), batch_size=batch_size, shuffle=False)

    device = pick_device()
    print(f"training on {device}, {len(train_utts)} train / {len(test_utts)} test utterances")

    model = AudioCodec().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.8, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    max_entropy_bits = torch.log2(torch.tensor(float(model.quantizer.codebook_size)))

    for epoch in range(1, epochs + 1):
        model.train()
        running_recon, running_vq, running_entropy, n_batches = 0.0, 0.0, 0.0, 0
        progress = tqdm(train_loader, desc=f"epoch {epoch:3d}/{epochs}")
        for waveform in progress:
            waveform = waveform.to(device)
            reconstruction, vq_loss, _, entropy_bits = model(waveform)
            recon_loss = reconstruction_loss(reconstruction, waveform)
            loss = recon_loss + vq_loss_weight * vq_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_recon += recon_loss.item()
            running_vq += vq_loss.item()
            running_entropy += entropy_bits.item()
            n_batches += 1
            progress.set_postfix(recon=running_recon / n_batches, vq=running_vq / n_batches)
        scheduler.step()

        train_recon = running_recon / n_batches
        train_vq = running_vq / n_batches
        bitrate_efficiency = running_entropy / n_batches / max_entropy_bits.item()

        model.eval()
        with torch.no_grad():
            test_recon_total, test_batches = 0.0, 0
            for waveform in test_loader:
                waveform = waveform.to(device)
                reconstruction, _, _, _ = model(waveform)
                test_recon_total += reconstruction_loss(reconstruction, waveform).item()
                test_batches += 1
            test_recon = test_recon_total / max(test_batches, 1)

        print(
            f"epoch {epoch:3d}/{epochs} | "
            f"train recon {train_recon:.4f} | vq {train_vq:.4f} | "
            f"bitrate efficiency {bitrate_efficiency:.1%} | "
            f"test recon {test_recon:.4f}"
        )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": {
        "base_channels": 32, "latent_dim": 64, "strides": (2, 4, 5),
        "codebook_size": 1024, "codebook_dim": 8,
    }}, CHECKPOINT_PATH)
    print(f"saved checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()
