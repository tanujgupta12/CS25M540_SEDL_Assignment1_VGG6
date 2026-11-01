# MTech IIT Madras || System Engineering for Deep Learning || Assignment 1
# Created by Tanuj Gupta (CS25M540)
"""Evaluate a saved VGG6 checkpoint on the CIFAR-10 validation and test sets.

Example:
    python evaluate.py --checkpoint checkpoints/best/best.pt
"""
import argparse

import torch

from vgg6.data import get_loaders
from vgg6.engine import evaluate
from vgg6.model import VGG6
from vgg6.utils import get_device, set_seed


def main():
    p = argparse.ArgumentParser(description="Evaluate a VGG6 checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    set_seed(cfg["seed"])

    model = VGG6(activation=cfg["activation"]).to(device)
    model.load_state_dict(ckpt["model"])
    _, val_loader, test_loader = get_loaders(args.data_dir, cfg["batch_size"], seed=cfg["seed"],
                                             num_workers=args.num_workers)

    val_loss, val_acc = evaluate(model, val_loader, device)
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Config: {cfg}")
    print(f"Checkpoint from epoch {ckpt['epoch']}")
    print(f"Validation: loss {val_loss:.4f}  top-1 acc {val_acc * 100:.2f}%")
    print(f"Test:       loss {test_loss:.4f}  top-1 acc {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
