# HW4 PromptIR Rain/Snow Restoration

This folder contains the code used for the HW4 image restoration task. The
model is based on PromptIR and is trained from scratch on the provided rain/snow
training pairs. No pretrained weights or external data are used.

## Files

```text
HW4/
  train.py
  infer.py
  data.py
  model.py
  requirements.txt
  assets/
    loss_curve.png
    psnr_curve.png
    metrics.csv
```

## Source

- Paper: PromptIR: Prompting for All-in-One Image Restoration, NeurIPS 2023
- Official code: https://github.com/va1shn9v/PromptIR
- HW4 requirement: train one model for both rain and snow restoration.

## Setup

```bash
git clone https://github.com/ABparadise33/NYCU-VRDL-2026-Spring.git
cd NYCU-VRDL-2026-Spring/HW4

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Dataset

```bash
python -m pip install gdown
gdown 1bEIU9TZVQa-AF_z6JkOKaGp4wYGnqQ8w -O hw4_realse_dataset.zip
unzip hw4_realse_dataset.zip
```

Expected layout:

```text
HW4/
  hw4_realse_dataset/
    train/
      degraded/
      clean/
    test/
      degraded/
```

## Final Training Command

The final submitted setting uses L1 loss, 2x rain oversampling, and directional
gradient loss. Rain oversampling is used because rain was the main validation
bottleneck. The gradient loss is used to penalize residual streak-like edge
artifacts.

```bash
python train.py \
  --dataset-root hw4_realse_dataset \
  --output-dir runs/promptir_rain_over_grad \
  --epochs 120 \
  --batch-size 4 \
  --patch-size 128 \
  --num-workers 4 \
  --device cuda \
  --loss-type l1 \
  --rain-oversample 2 \
  --grad-weight 0.05 \
  --grad-x-weight 2.0 \
  --grad-y-weight 1.0
```

Training outputs:

```text
runs/promptir_rain_over_grad/latest.pt
runs/promptir_rain_over_grad/best_psnr.pt
runs/promptir_rain_over_grad/best_loss.pt
runs/promptir_rain_over_grad/metrics.csv
runs/promptir_rain_over_grad/loss_curve.png
runs/promptir_rain_over_grad/psnr_curve.png
```

`best_psnr.pt` is selected by the highest overall validation PSNR.

## Inference

Use the best checkpoint from the final rain-oversampling + gradient-loss model:

```bash
python infer.py \
  --dataset-root hw4_realse_dataset \
  --checkpoint runs/promptir_rain_over_grad/best_psnr.pt \
  --output pred.npz \
  --device cuda
```

If inference runs out of memory:

```bash
python infer.py \
  --dataset-root hw4_realse_dataset \
  --checkpoint runs/promptir_rain_over_grad/best_psnr.pt \
  --output pred.npz \
  --device cuda \
  --tile-size 256 \
  --tile-overlap 32
```

The generated `pred.npz` uses the required submission format: each key is the
original test filename, and each value is a restored `uint8` image array with
shape `(3, H, W)`.

## Report Assets

The `assets/` folder contains the curves and metric log from the final
experiment:

```text
assets/loss_curve.png
assets/psnr_curve.png
assets/metrics.csv
```

This task is image restoration rather than classification, so a confusion
matrix is not applicable.
