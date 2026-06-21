from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Snake(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (torch.sin(self.alpha * x) ** 2) / (self.alpha + 1e-9)


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = dilation * (7 - 1) // 2
        self.block = nn.Sequential(
            Snake(channels),
            nn.Conv1d(
                channels, channels, kernel_size=7, dilation=dilation, padding=padding
            ),
            Snake(channels),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        kernel = 2 * stride
        padding = math.ceil((kernel - stride) / 2)
        self.res_units = nn.Sequential(
            ResidualUnit(in_channels, dilation=1),
            ResidualUnit(in_channels, dilation=3),
        )
        self.downsample = nn.Sequential(
            Snake(in_channels),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_units(x)
        return self.downsample(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        kernel = 2 * stride
        padding = math.ceil((kernel - stride) / 2)
        output_padding = (kernel - stride) % 2
        self.upsample = nn.Sequential(
            Snake(in_channels),
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
        )
        self.res_units = nn.Sequential(
            ResidualUnit(out_channels, dilation=1),
            ResidualUnit(out_channels, dilation=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.res_units(x)


class Encoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        latent_dim: int = 64,
        strides: tuple[int, ...] = (2, 4, 5),
    ):
        super().__init__()
        channels = base_channels
        layers = [nn.Conv1d(1, channels, kernel_size=7, padding=3)]
        for stride in strides:
            next_channels = channels * 2
            layers.append(EncoderBlock(channels, next_channels, stride))
            channels = next_channels
        layers += [
            Snake(channels),
            nn.Conv1d(channels, latent_dim, kernel_size=3, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        latent_dim: int = 64,
        strides: tuple[int, ...] = (2, 4, 5),
    ):
        super().__init__()
        channels = base_channels * (2 ** len(strides))
        layers = [nn.Conv1d(latent_dim, channels, kernel_size=7, padding=3)]
        for stride in reversed(strides):
            next_channels = channels // 2
            layers.append(DecoderBlock(channels, next_channels, stride))
            channels = next_channels
        layers += [
            Snake(channels),
            nn.Conv1d(channels, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.in_proj = nn.Conv1d(latent_dim, codebook_dim, kernel_size=1)
        self.out_proj = nn.Conv1d(codebook_dim, latent_dim, kernel_size=1)
        self.codebook = nn.Parameter(torch.randn(codebook_size, codebook_dim) * 0.02)

    def forward(self, z_e: torch.Tensor):
        z_proj = self.in_proj(z_e)
        b, c, t = z_proj.shape
        flat = z_proj.permute(0, 2, 1).reshape(b * t, c)

        flat_n = F.normalize(flat, dim=-1)
        codebook_n = F.normalize(self.codebook, dim=-1)
        distances = (
            flat_n.pow(2).sum(1, keepdim=True)
            - 2 * flat_n @ codebook_n.t()
            + codebook_n.pow(2).sum(1)
        )
        code_indices = distances.argmin(dim=1)

        quantized_n = codebook_n[code_indices]

        codebook_loss = F.mse_loss(quantized_n, flat_n.detach())
        commitment_loss = F.mse_loss(flat_n, quantized_n.detach())
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        quantized_n_st = flat_n + (quantized_n - flat_n).detach()
        quantized_st = quantized_n_st.view(b, t, c).permute(0, 2, 1)
        z_q = self.out_proj(quantized_st)

        usage = torch.bincount(code_indices, minlength=self.codebook_size).float()
        probs = usage / usage.sum().clamp_min(1.0)
        entropy_bits = -(probs * (probs.clamp_min(1e-12)).log2()).sum()

        return z_q, vq_loss, code_indices.view(b, t), entropy_bits


class AudioCodec(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        latent_dim: int = 64,
        strides: tuple[int, ...] = (2, 4, 5),
        codebook_size: int = 1024,
        codebook_dim: int = 8,
    ):
        super().__init__()
        self.encoder = Encoder(base_channels, latent_dim, strides)
        self.quantizer = VectorQuantizer(latent_dim, codebook_size, codebook_dim)
        self.decoder = Decoder(base_channels, latent_dim, strides)
        self.total_stride = 1
        for s in strides:
            self.total_stride *= s

    def forward(self, waveform: torch.Tensor):
        z_e = self.encoder(waveform)
        z_q, vq_loss, codes, entropy_bits = self.quantizer(z_e)
        reconstruction = self.decoder(z_q)
        reconstruction = reconstruction[..., : waveform.shape[-1]]
        return reconstruction, vq_loss, codes, entropy_bits

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(waveform)
        _, _, codes, _ = self.quantizer(z_e)
        return codes

    @torch.no_grad()
    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        codebook_n = F.normalize(self.quantizer.codebook, dim=-1)
        quantized_flat = codebook_n[codes.reshape(-1)]
        b, t = codes.shape
        c = quantized_flat.shape[-1]
        quantized = quantized_flat.view(b, t, c).permute(0, 2, 1)
        z_q = self.quantizer.out_proj(quantized)
        return self.decoder(z_q)


if __name__ == "__main__":
    model = AudioCodec()
    dummy = torch.randn(2, 1, 8000)
    recon, vq_loss, codes, entropy_bits = model(dummy)
    print("input shape:", dummy.shape)
    print("reconstruction shape:", recon.shape)
    print("codes shape:", codes.shape, "total stride:", model.total_stride)
    print("vq_loss:", vq_loss.item(), "entropy bits:", entropy_bits.item())
    n_params = sum(p.numel() for p in model.parameters())
    print("total parameters:", n_params)
