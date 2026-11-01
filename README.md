<!-- MTech IIT Madras || System Engineering for Deep Learning || Assignment 1 -->
<!-- Created by Tanuj Gupta (CS25M540) -->
# VGG6 on CIFAR-10: activations, optimizers and hyperparameters

CS6886W Systems Engineering for Deep Learning, Assignment 1. Name: Tanuj. Roll no: CS25M540.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tanujgupta12/CS25M540_SEDL_Assignment1_VGG6/blob/main/colab_run.ipynb)

Code for training VGG6 on CIFAR-10 under different configurations, with W&B sweeps for the comparison plots.

## Repository layout

```
vgg6/
  model.py    VGG6 (4 conv + 2 FC layers, BatchNorm) with a configurable activation
  data.py     CIFAR-10 loading, normalisation, augmentation, fixed 45k/5k train/val split
  optim.py    Optimizer factory (SGD, Nesterov, Adam, Adagrad, RMSprop, Nadam)
  engine.py   train_one_epoch() and evaluate()
  utils.py    Seeding, device selection, local loss/accuracy plots
train.py      Train one configuration, log to W&B, save the best checkpoint
evaluate.py   Evaluate a saved checkpoint on validation and test sets
wandb_results.py  Pick the best sweep configuration and export all runs from W&B
sweeps/       W&B sweep configs for Q2(a), Q2(b) and Q2(c)
run_sweeps.sh Creates and runs all three sweeps
colab_run.ipynb  Convenience launcher for Google Colab (all logic lives in the .py files)
checkpoints/best/best.pt   Trained model for the best configuration
```

## Environment

| | Version |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.11.0 (CUDA) |
| torchvision | 0.26.0 |
| wandb | 0.28.1 |
| Hardware | Google Colab, NVIDIA Tesla T4 (the code also runs on Apple MPS and on CPU) |

```bash
pip install -r requirements.txt
wandb login
```

## Data

- **Split:** the 50,000 CIFAR-10 training images are split into 45,000 train and 5,000 validation images, using a fixed seed. The 10,000-image test set is used only for final reporting.
- **Normalisation:** per-channel mean `(0.4914, 0.4822, 0.4465)` and std `(0.2470, 0.2435, 0.2616)`.
- **Training augmentation:** `RandomCrop(32, padding=4)` and `RandomHorizontalFlip()`.
- **Validation and test:** `ToTensor()` and `Normalize` only, with no augmentation.

## Seed

All runs use `--seed 42`. The seed sets Python, NumPy and PyTorch (CPU and CUDA), and the train/val split. cuDNN runs in deterministic mode (`deterministic=True`, `benchmark=False`).

## Reproduce

**Reproduce the best configuration (Q4):**

```bash
python train.py --activation silu --optimizer nesterov --lr 0.05 --batch_size 128 --epochs 30 --seed 42 --save_dir checkpoints/best
python evaluate.py --checkpoint checkpoints/best/best.pt
```

**Evaluate the uploaded model without retraining:**

```bash
python evaluate.py --checkpoint checkpoints/best/best.pt
```

**Run all configuration sweeps (Q2, Q3):**

```bash
bash run_sweeps.sh CS6886W-SEDL-Assignment1-VGG6
```

**Train one configuration by hand:**

```bash
python train.py --activation relu --optimizer adam --batch_size 128 --epochs 10
```

If `--lr` is left out, each optimizer gets its own default:

| Optimizer | Default `--lr` |
|---|---|
| SGD, Nesterov | 0.05 |
| Adagrad | 0.01 |
| Adam, RMSprop, Nadam | 0.001 |

All runs use these settings:

- weight decay 5e-4
- momentum 0.9, for the optimizers that use it
- cosine learning-rate schedule
- cross-entropy loss

## Results

| Configuration | Best val acc | Test acc |
|---|---|---|
| **SiLU, Nesterov, lr 0.05, bs 128, 30 epochs (best, Q4)** | **87.66%** | **86.90%** |
| SiLU, Nesterov, lr 0.05, bs 128, 10 epochs | 80.26% | 80.45% |
| GELU, Nesterov, lr 0.05, bs 128, 10 epochs | 79.66% | 79.43% |
| ReLU, Nadam, lr 0.001, bs 128, 10 epochs | 77.38% | 76.52% |
| ReLU, Adam, lr 0.001, bs 128, 10 epochs | 76.56% | 76.50% |
| ReLU, Nesterov, lr 0.05, bs 256, 10 epochs | 75.68% | 74.76% |
| ReLU, SGD, lr 0.05, bs 128, 10 epochs | 75.44% | 74.66% |
| ReLU, Nesterov, lr 0.05, bs 128, 10 epochs (baseline) | 75.14% | 74.19% |
| ReLU, Nesterov, lr 0.01, bs 64, 5 epochs | 72.42% | 71.23% |
| ReLU, RMSprop, lr 0.001, bs 128, 10 epochs | 71.34% | 69.84% |
| ReLU, Nesterov, lr 0.1, bs 64, 15 epochs | 70.72% | 69.44% |
| Tanh, Nesterov, lr 0.05, bs 128, 10 epochs | 70.62% | 70.23% |
| ReLU, Nesterov, lr 0.1, bs 256, 10 epochs | 68.46% | 67.20% |
| Sigmoid, Nesterov, lr 0.05, bs 128, 10 epochs | 65.44% | 65.33% |
| ReLU, Adagrad, lr 0.01, bs 128, 10 epochs | 64.92% | 63.58% |
| ReLU, Nesterov, lr 0.1, bs 256, 5 epochs | 51.96% | 51.12% |

All 19 W&B runs (baseline, 17 sweep runs and the final 30-epoch run) are in the W&B project `CS6886W-SEDL-Assignment1-VGG6`. Test accuracy is for the best-validation checkpoint of each run.

**Best configuration:** SiLU activation, SGD with Nesterov momentum (0.9), lr 0.05 with cosine schedule, batch size 128, weight decay 5e-4, 30 epochs, seed 42. Validation 87.66%, test 86.90%. The model is in `checkpoints/best/best.pt`.

## How performance varies across configurations

**Activations.** With everything else fixed (Nesterov, lr 0.05, 10 epochs), smooth non-saturating activations did best: SiLU 80.3% and GELU 79.7%, against ReLU 75.1%, tanh 70.6% and sigmoid 65.4%. SiLU reached 70% validation accuracy by epoch 4 and GELU by epoch 5; ReLU needed 8 epochs. SiLU and GELU keep a small gradient for negative inputs, which suits this fairly high learning rate. Tanh and sigmoid saturate, so gradients shrink; sigmoid was the least stable, falling from 36.9% at epoch 1 to 13.5% at epoch 5 before recovering as the learning rate dropped.

**Optimizers.** With ReLU and each optimizer's default learning rate, Nadam (77.4%) and Adam (76.6%) were best and also fastest early on (49-54% after epoch 1, against 23-25% for SGD and Nesterov). Plain SGD with momentum reached 75.4%, Nesterov 75.1% and RMSprop 71.3%. Adagrad was worst (64.9%) because its accumulated squared gradients keep shrinking the effective step size. The SGD-family runs start slowly at lr 0.05 but keep improving as the cosine schedule lowers the learning rate.

**Batch size, epochs and learning rate.** lr 0.05 worked well at 10 epochs, and lr 0.01 with batch size 64 reached 72.4% in only 5 epochs. lr 0.1 was the worst setting at both batch sizes tried (68.5% at batch 256 against 75.7% for lr 0.05, and unstable validation accuracy at batch 64). Batch size made little difference to wall-clock time on the T4, at about 20-21 s per epoch for 64, 128 and 256. More epochs helped: batch 256 at lr 0.1 went from 52.0% at 5 epochs to 68.5% at 10, and the best configuration went from 80.3% at 10 epochs to 87.7% at 30, with its best checkpoint at epoch 29.
