# MTech IIT Madras || System Engineering for Deep Learning || Assignment 1
# Created by Tanuj Gupta (CS25M540)
"""Train VGG6 on CIFAR-10 with one configuration and log everything to W&B.

Run directly with command-line flags, or let a W&B sweep agent call it
(the agent passes the sweep's chosen values as --flag=value arguments).
"""
import argparse
import os
import time

import torch
import wandb

from vgg6.data import get_loaders
from vgg6.engine import evaluate, train_one_epoch
from vgg6.model import ACTIVATIONS, VGG6
from vgg6.optim import DEFAULT_LR, build_optimizer
from vgg6.utils import get_device, save_history, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train VGG6 on CIFAR-10")
    p.add_argument("--activation", default="relu", choices=list(ACTIVATIONS))
    p.add_argument("--optimizer", default="nesterov", choices=list(DEFAULT_LR))
    p.add_argument("--lr", type=float, default=None, help="defaults to a per-optimizer value (vgg6/optim.py)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "none"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_dir", default=None, help="where to save the best checkpoint (default: runs/<run name>)")
    p.add_argument("--project", default="CS6886W-SEDL-Assignment1-VGG6")
    p.add_argument("--no_wandb", action="store_true", help="train without logging to W&B")
    return p.parse_args()


def main():
    args = parse_args()
    if args.lr is None:
        args.lr = DEFAULT_LR[args.optimizer]

    run = wandb.init(project=args.project, config=vars(args), mode="disabled" if args.no_wandb else None)
    # A sweep agent passes its values as CLI flags, so args already holds the sweep's choices
    cfg = args
    run.name = f"{cfg.activation}-{cfg.optimizer}-bs{cfg.batch_size}-lr{cfg.lr:g}-ep{cfg.epochs}"

    set_seed(cfg.seed)
    device = get_device()
    print(f"Device: {device} | config: {dict(vars(args))}")

    train_loader, val_loader, test_loader = get_loaders(cfg.data_dir, cfg.batch_size, seed=cfg.seed,
                                                        num_workers=cfg.num_workers)
    model = VGG6(activation=cfg.activation).to(device)
    optimizer = build_optimizer(cfg.optimizer, model.parameters(), lr=cfg.lr,
                                momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
                 if cfg.scheduler == "cosine" else None)

    save_dir = args.save_dir or os.path.join("runs", run.name)
    os.makedirs(save_dir, exist_ok=True)
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val_acc, step = 0.0, 0

    def log_step(loss, acc):
        nonlocal step
        step += 1
        if step % 50 == 0:
            wandb.log({"batch/train_loss": loss, "batch/train_acc": acc}, step=step)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, on_step=log_step)
        val_loss, val_acc = evaluate(model, val_loader, device)
        if scheduler:
            scheduler.step()

        for k, v in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[k].append(v)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                   "val_loss": val_loss, "val_acc": val_acc,
                   "lr": optimizer.param_groups[0]["lr"]}, step=step)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "config": dict(vars(args)), "epoch": epoch,
                        "val_acc": val_acc}, os.path.join(save_dir, "best.pt"))
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - t0:.1f}s")

    # Report test accuracy of the best-validation checkpoint
    model.load_state_dict(torch.load(os.path.join(save_dir, "best.pt"), map_location=device)["model"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    wandb.run.summary.update({"best_val_acc": best_val_acc, "test_acc": test_acc, "test_loss": test_loss})
    save_history(history, save_dir)
    print(f"Best val acc {best_val_acc:.4f} | test acc {test_acc:.4f} | saved to {save_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
