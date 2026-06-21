from __future__ import annotations

import torch
import torch.nn.functional as F

STFT_WINDOW_LENGTHS = (128, 256, 512, 1024)


def stft_magnitude(waveform: torch.Tensor, window_length: int) -> torch.Tensor:
    hop_length = window_length // 4
    device = waveform.device
    waveform_cpu = waveform.squeeze(1).to("cpu")
    window = torch.hann_window(window_length, device="cpu")
    spec = torch.stft(
        waveform_cpu,
        n_fft=window_length,
        hop_length=hop_length,
        win_length=window_length,
        window=window,
        return_complex=True,
    )
    return spec.abs().to(device)


def multi_resolution_stft_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    total = prediction.new_zeros(())
    for window_length in STFT_WINDOW_LENGTHS:
        pred_mag = stft_magnitude(prediction, window_length)
        target_mag = stft_magnitude(target, window_length)
        total = total + F.l1_loss(pred_mag, target_mag)
        log_pred = torch.log(pred_mag.clamp_min(1e-5))
        log_target = torch.log(target_mag.clamp_min(1e-5))
        total = total + F.l1_loss(log_pred, log_target)
    return total / len(STFT_WINDOW_LENGTHS)


def waveform_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(prediction, target)


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, stft_weight: float = 1.0) -> torch.Tensor:
    return waveform_loss(prediction, target) + stft_weight * multi_resolution_stft_loss(prediction, target)


if __name__ == "__main__":
    a = torch.randn(2, 1, 8000)
    b = a + 0.01 * torch.randn_like(a)
    print("loss for near-identical signals:", reconstruction_loss(a, b).item())
    c = torch.randn(2, 1, 8000)
    print("loss for unrelated signals:", reconstruction_loss(a, c).item())
