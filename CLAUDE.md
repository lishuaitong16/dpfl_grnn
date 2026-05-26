# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research implementation of **Federated Learning with Differential Privacy (DP-FL)** paired with a **GRNN gradient inversion attack** (from the ICLR paper). The goal is to study the privacy-utility trade-off: DP noise protects against gradient inversion, but degrades model accuracy.

Dataset: MNIST, padded to 32×32. Model: LeNet-5. All commands run from the `dpfl_grnn/` directory using `python -m` for correct imports.

## Commands

### Run the four experiment stages (in order)

```bash
bash run_fl.sh          # Step 1: FL baseline (no DP) → results/baseline_acc.png
bash run_dpfl.sh        # Step 2: Noise sweep → results/dp_laplace_acc.png, dp_gaussian_acc.png
bash run_attack.sh      # Step 3: GRNN attack (no DP) → results/attack/
bash run_defense.sh     # Step 4: Noise + GRNN attack → results/defense/
```

### Run individual experiments directly

```bash
python -m experiments.exp0_fl_baseline --clients 5 --rounds 30 --gpu 0
python -m experiments.exp1_dp_params --noise_stds 1e-1,1e-2,0 --gpu 1 --outdir ./results
python -m experiments.exp2_attack --batch_size 1 --iterations 2000 --gpu 0 --outdir ./results/attack
python -m experiments.exp3_defense --noise_stds 1e-1,1e-2,0 --iterations 2000 --gpu 0 --outdir ./results/defense
python -m experiments.exp4_train_stage --gpu 0 --outdir ./results/train_stage
```

## Architecture

### Data flow

```
fl/data.py       → MNIST load + pad 28→32 + IID split across clients
fl/train.py      → FedAvg loop: download global model → local SGD (+optional noise) → upload delta → average
fl/dp.py         → privatize(): add Laplace or Gaussian noise to a flattened gradient vector
```

### Attack flow

```
attack/grnn.py     → GRNNGenerator: latent vector → (fake_img, fake_label)
attack/losses.py   → grnn_loss(): MSE + Wasserstein distance + TV regularization
attack/run_attack.py → grnn_attack(): freeze global model, optimize generator to match true_grad
```

### Models

`models/lenet.py` — LeNet-5 with switchable activation:
- `act="sigmoid"` for GRNN attack (required for second-order gradients via `create_graph=True`)
- `act="relu"` for FL training (better convergence)

`models/resnet.py` — ResNet variant (available but not used in current experiments).

## Critical Implementation Details

**Activation choice matters:** The GRNN attack computes gradients of gradients (`create_graph=True`). ReLU is not twice-differentiable at 0, so the attack model **must** use `act="sigmoid"`. FL training experiments use `act="relu"` for accuracy.

**Gradient flattening order:** True gradients and fake gradients are both flattened by iterating `model.parameters()` in the same order. Any inconsistency breaks the attack. See `params_to_vector()` in `fl/train.py` and `compute_true_gradient()` / `compute_fake_gradient()` in `attack/run_attack.py`.

**GLU channel halving:** `UpsamplingBlock` outputs `2*out_ch` channels from its Conv, then `F.glu(dim=1)` halves it to `out_ch`. The FC layer before the first block also uses GLU: `fc_img` outputs `base_ch * 4 * 4 * 2`, then GLU gives `base_ch * 4 * 4`.

**Noise injection point:** In `fl/train.py`, noise is added to gradients after `loss.backward()` and before `optimizer.step()` — the DP-SGD style. No gradient clipping is applied; `fl/dp.py` only adds noise.

**PSNR interpretation:** Higher = more similar to original. After GRNN attack with no noise: high PSNR (attack succeeds). With large noise_std: low PSNR (DP protects privacy).

## Output Files

| Script | Key outputs |
|--------|-------------|
| `run_fl.sh` | `results/baseline_acc.png` |
| `run_dpfl.sh` | `results/dp_laplace_acc.png`, `dp_gaussian_acc.png`, `dp_acc_data.csv` |
| `run_attack.sh` | `results/attack/recover_process.png`, `compare.png`, `loss_curve.png` |
| `run_defense.sh` | `results/defense/defense_laplace.png`, `defense_gaussian.png`, `psnr_bar.png`, `psnr_data.csv` |
| `run_train_stage.sh` | `results/train_stage/recovered.png`, `stage_curve.png`, `stage_data.csv` |
