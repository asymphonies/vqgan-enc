# Minimal Neural Audio Codec (inspired by Improved RVQGAN)

A minimal implementation of a neural
encode-quantize-decode pipeline for raw speech waveforms, trained and
evaluated on the Free Spoken Digit Dataset (FSDD), developed as a side project during the **Seminar: Selected Topics in Communications Engineering**, WS 24/25, RWTH Aachen. The current version on this repository is an improved and refactored version of the original implementation.

This is inspired by **High-Fidelity Audio
Compression with Improved RVQGAN** _(Kumar et al., NeurIPS 2023)_ and **Neural Discrete Representation Learning** _(van den Oord et al., VQ-VAE, NeurIPS 2017)_:

- Convolutional encoder/decoder operating directly on raw audio
  samples like SoundStream/EnCodec/DAC.
- Snake periodic activation (`x + sin(ax)^2 / a`) following
  BigVGAN and the Improved RVQGAN paper, motivated by Ziyin et al. 2020's
  result that standard activations cannot extrapolate periodic signals.
- A single VQ-VAE codebook with the factorized low-dimensional lookup and
  L2-normalized cosine-similarity matching introduced in Improved VQGAN and
  ViT-VQGAN (Yu et al. 2021).
- The straight-through gradient estimator and the codebook - commitment loss
  from the original VQ-VAE paper.

## Files

- `dataset.py` - downloads FSDD, loads wav files, fixed-length cropping/padding.
- `model.py` - Snake activation, conv encoder, VQ bottleneck, conv decoder.
- `losses.py` - multi-resolution STFT/L1 reconstruction loss.
- `train.py` - training loop, checkpointing, codebook usage logging.
- `evaluate.py` - reconstructs held-out audio, saves wav files and a metrics report.
- `inspect_codes.py` - encodes a handful of files and prints/saves their discrete code sequences for understanding what the bottleneck looks like.

## How to run

```bash
python -m venv venv # with uv: uv venv
source venv/bin/activate
pip install -r requirements.txt

python dataset.py
python train.py
python evaluate.py
python inspect_codes.py
```

On an Apple Silicon Mac, `train.py` will automatically use the `mps` backend
if available, falling back to CPU otherwise.

## How to read the code

The code is intended to be read in this order: `model.py` (the actual codec), `losses.py` (what
"reconstruction quality" means here), `dataset.py` (data plumbing), then
`train.py` (how they're wired together). Run `evaluate.py` and
`inspect_codes.py` to gain insight into what the model is learning.

## Future further improvements

1. **Residual Vector Quantization:** Extend the `VectorQuantizer` in
   `model.py` into `ResidualVectorQuantizer` that holds multiple codebooks and
   loops `residual = residual - quantize(residual)`. Add quantizer dropout.
2. **Multi-scale mel loss:** Extend `losses.py` to compute log-mel L1 loss at
   several window lengths instead of only an STFT magnitude loss.
3. **Adversarial training:** Add a multi-period waveform discriminator
   (e.g. reshape the 1D signal at alternating strides and run a small 2D CNN over
   each) and a hinge GAN loss plus feature-matching loss.
4. **Bigger dataset:** Swap FSDD for a larger audio dataset once the pipeline is
   verified correct on the small dataset.
