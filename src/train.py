"""
Supervised pretraining for PetraNet.

Trains value head and policy head jointly on Lichess game data:
  value  loss: MSE  (predicted value vs game outcome)
  policy loss: cross-entropy (predicted move distribution vs move played)
  total  loss: value_loss + policy_loss  (equal weighting, AlphaZero convention)

Validation
----------
Runs after every epoch on the held-out val set (split at game level in data.py).
Reports: total loss, value MSE, value R², policy top-1 accuracy, policy top-5 accuracy.

Early stopping on val total loss (patience=5 epochs).
Best checkpoint saved separately from latest.

Usage
-----
    python3 train.py --dataset dataset.pt --out models/
    python3 train.py --dataset dataset.pt --out models/ --epochs 20 --lr 3e-4
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from model import PetraNet
from config import device


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(path: str):
    """Load a dataset saved by data.py. Returns (train_loader, val_loader)."""
    print(f"Loading dataset from {path} ...")
    data = torch.load(path, map_location="cpu", weights_only=False)

    def make_loader(split, batch_size, shuffle):
        d = data[split]
        ds = TensorDataset(d["tensors"], d["values"], d["move_idxs"])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    meta = data.get("meta", {})
    print(f"  train: {meta.get('n_train', '?'):,}  val: {meta.get('n_val', '?'):,}")
    return make_loader, data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, is_train: bool):
    model.train(is_train)
    total_loss = total_vloss = total_ploss = 0.0
    n_batches = 0
    all_value_preds, all_value_targets = [], []
    policy_correct_top1 = policy_correct_top5 = policy_total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for tensors, values, move_idxs in loader:
            tensors   = tensors.float().to(device)   # uint8 → float32
            values    = values.to(device)
            move_idxs = move_idxs.to(device)

            value_pred, policy_logits = model(tensors)
            value_pred = value_pred.squeeze(1)

            vloss = F.mse_loss(value_pred, values)
            ploss = F.cross_entropy(policy_logits, move_idxs)
            loss  = vloss + ploss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss  += loss.item()
            total_vloss += vloss.item()
            total_ploss += ploss.item()
            n_batches   += 1

            # Accumulate for R² and accuracy
            all_value_preds.append(value_pred.detach().cpu())
            all_value_targets.append(values.detach().cpu())

            topk = policy_logits.topk(5, dim=1).indices
            policy_correct_top1 += (topk[:, 0] == move_idxs).sum().item()
            policy_correct_top5 += (topk == move_idxs.unsqueeze(1)).any(dim=1).sum().item()
            policy_total        += len(move_idxs)

    preds   = torch.cat(all_value_preds)
    targets = torch.cat(all_value_targets)
    ss_res  = ((preds - targets) ** 2).sum().item()
    ss_tot  = ((targets - targets.mean()) ** 2).sum().item() + 1e-8
    r2      = 1.0 - ss_res / ss_tot

    return {
        "loss":       total_loss  / n_batches,
        "value_loss": total_vloss / n_batches,
        "policy_loss":total_ploss / n_batches,
        "value_r2":   r2,
        "top1":       policy_correct_top1 / policy_total,
        "top5":       policy_correct_top5 / policy_total,
    }


def train(dataset_path: str,
          out_dir: str,
          epochs: int = 15,
          batch_size: int = 512,
          lr: float = 1e-3,
          patience: int = 5,
          seed: int = 42):

    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    make_loader, data = load_dataset(dataset_path)
    train_loader = make_loader("train", batch_size=batch_size, shuffle=True)
    val_loader   = make_loader("val",   batch_size=batch_size, shuffle=False)

    model = PetraNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PetraNet: {n_params:,} parameters  |  device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-5
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"\n{'Epoch':>5}  {'T-loss':>7}  {'V-loss':>7}  {'V-MSE':>6}  "
          f"{'V-R²':>6}  {'Top1':>5}  {'Top5':>5}  {'LR':>8}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_m = run_epoch(model, train_loader, optimizer, is_train=True)
        val_m   = run_epoch(model, val_loader,   optimizer, is_train=False)

        scheduler.step(val_m["loss"])
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"{epoch:>5}  "
              f"{train_m['loss']:>7.4f}  "
              f"{val_m['loss']:>7.4f}  "
              f"{val_m['value_loss']:>6.4f}  "
              f"{val_m['value_r2']:>6.3f}  "
              f"{val_m['top1']:>5.3f}  "
              f"{val_m['top5']:>5.3f}  "
              f"{lr_now:>8.2e}  "
              f"({elapsed:.0f}s)")

        # Checkpointing
        torch.save(model.state_dict(), os.path.join(out_dir, "latest.pt"))

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            print(f"         ↳ new best val loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping: no improvement for {patience} epochs.")
                break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model → {os.path.join(out_dir, 'best.pt')}")

    # Final sanity check on value range
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"),
                                     map_location=device, weights_only=True))
    _sanity_check(model)


# ---------------------------------------------------------------------------
# Sanity check — runs after training, not a substitute for ELO testing
# ---------------------------------------------------------------------------

def _sanity_check(model: PetraNet):
    import chess
    print("\nPost-training sanity checks:")
    model.eval()

    tests = [
        ("Start position",          chess.Board(),                          None),
        ("White up queen",          chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1"),  1.0),
        ("Black up queen",          chess.Board("4K3/8/8/8/8/8/8/q3k3 w - - 0 1"), -1.0),
        ("KQ vs K, White to move",  chess.Board("8/8/8/8/4k3/8/8/3QK3 w - - 0 1"),  1.0),
        ("KQ vs K, Black to move",  chess.Board("8/8/8/8/4k3/8/8/3QK3 b - - 0 1"), -1.0),
    ]

    all_pass = True
    for name, board, expected in tests:
        val = model.value(board, device)
        if expected is None:
            ok = True
            mark = "~"
        else:
            ok = (val * expected) > 0   # correct sign
            mark = "✓" if ok else "✗"
        if not ok:
            all_pass = False
        print(f"  {mark} {name:35s}  value={val:+.3f}"
              + (f"  (expected sign {'+'  if expected > 0 else '-'})" if expected else ""))

    if all_pass:
        print("  All sign checks passed.")
    else:
        print("  WARNING: some sign checks failed — review training data.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",    required=True)
    ap.add_argument("--out",        default="models")
    ap.add_argument("--epochs",     type=int,   default=15)
    ap.add_argument("--batch-size", type=int,   default=512)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--patience",   type=int,   default=5)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    train(
        dataset_path=args.dataset,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
