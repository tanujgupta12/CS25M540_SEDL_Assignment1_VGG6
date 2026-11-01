# MTech IIT Madras || System Engineering for Deep Learning || Assignment 1
# Created by Tanuj Gupta (CS25M540)
"""Read finished runs back from W&B.

Examples:
    python wandb_results.py best                       # flags of the best sweep run
    python wandb_results.py export --out results.json  # every run, for the report
"""
import argparse
import json

import wandb

HISTORY_KEYS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]


def finished_runs(project):
    api = wandb.Api()
    return list(api.runs(f"{api.default_entity}/{project}", filters={"state": "finished"}))


def best_flags(project):
    """Returns train.py flags of the sweep run with the best validation accuracy.

    Selection uses validation accuracy only, so the test set stays out of it.
    """
    scored = [r for r in finished_runs(project) if r.sweep and r.summary.get("best_val_acc") is not None]
    if not scored:
        raise SystemExit(f"No finished sweep runs with best_val_acc in project '{project}'")
    best = max(scored, key=lambda r: r.summary.get("best_val_acc"))
    c = best.config
    return f"--activation {c['activation']} --optimizer {c['optimizer']} --lr {c['lr']:g} --batch_size {c['batch_size']}"


def export(project, out):
    """Writes config, summary and per-epoch history of every finished run to a JSON file."""
    results = []
    for r in finished_runs(project):
        results.append({
            "name": r.name,
            "sweep": r.sweep.name if r.sweep else None,
            "config": dict(r.config),
            "best_val_acc": r.summary.get("best_val_acc"),
            "test_acc": r.summary.get("test_acc"),
            "test_loss": r.summary.get("test_loss"),
            "runtime_s": r.summary.get("_runtime"),
            "history": [dict(row) for row in r.scan_history(keys=HISTORY_KEYS)],
        })
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported {len(results)} runs to {out}")


def main():
    p = argparse.ArgumentParser(description="Read finished runs back from W&B")
    p.add_argument("command", choices=["best", "export"])
    p.add_argument("--project", default="CS6886W-SEDL-Assignment1-VGG6")
    p.add_argument("--out", default="results.json")
    args = p.parse_args()

    if args.command == "best":
        print(best_flags(args.project))
    else:
        export(args.project, args.out)


if __name__ == "__main__":
    main()
