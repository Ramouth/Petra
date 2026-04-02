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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PetraNet
from config import device


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(path: str):
    """
    Load a dataset saved by data.py or selfplay.py.

    Supervised datasets have 'move_idxs' (int64 one-hot target).
    Self-play datasets additionally have 'visit_dists' (float32 dense target).
    Always returns 4-tuple loaders — supervised positions get a one-hot visit_dist
    so they can be mixed with self-play data in a single DataLoader.
    """
    print(f"Loading dataset from {path} ...")
    data = torch.load(path, map_location="cpu", weights_only=False)

    has_visit_dists = "visit_dists" in data.get("train", {})
    if has_visit_dists:
        print("  Self-play dataset detected — using dense policy loss (visit distributions)")
    else:
        print("  Supervised dataset detected — one-hot visit distributions will be created")

    def make_loader(split, batch_size, shuffle):
        d = data[split]
        ds = TensorDataset(
            d["tensors"], d["values"], d["move_idxs"], _ensure_visit_dists(d)
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    meta = data.get("meta", {})
    print(f"  train: {meta.get('n_train', '?'):,}  val: {meta.get('n_val', '?'):,}  "
          f"source: {meta.get('source', 'supervised')}")
    return make_loader, data, True   # always dense_policy=True now


def _ensure_visit_dists(d: dict) -> torch.Tensor:
    """Return visit_dists tensor; create one-hot from move_idxs if absent."""
    if "visit_dists" in d:
        return d["visit_dists"]
    n = len(d["move_idxs"])
    vd = torch.zeros(n, 4096, dtype=torch.float32)
    vd[torch.arange(n), d["move_idxs"]] = 1.0
    return vd


def mix_anchor(primary_data: dict, anchor_path: str, anchor_frac: float) -> dict:
    """
    Mix a fraction of anchor (supervised SF) positions into the primary training split.

    anchor_frac : fraction of primary train size to sample from anchor.
                  e.g. 0.15 → 15% extra positions from the SF oracle dataset.

    Val split is kept pure (primary only) so validation measures self-play generalisation.
    """
    print(f"\nMixing anchor dataset: {anchor_path}  (frac={anchor_frac})")
    anchor = torch.load(anchor_path, map_location="cpu", weights_only=False)

    n_primary = len(primary_data["train"]["tensors"])
    n_sample  = max(1, int(n_primary * anchor_frac))
    n_anchor  = len(anchor["train"]["tensors"])
    n_sample  = min(n_sample, n_anchor)

    idx = torch.randperm(n_anchor)[:n_sample]
    a   = anchor["train"]

    # One-hot visit_dists for supervised anchor positions
    vd = torch.zeros(n_anchor, 4096, dtype=torch.float32)
    vd[torch.arange(n_anchor), a["move_idxs"]] = 1.0

    primary_vd = _ensure_visit_dists(primary_data["train"])
    mixed = {
        "tensors":     torch.cat([primary_data["train"]["tensors"],   a["tensors"][idx]]),
        "values":      torch.cat([primary_data["train"]["values"],    a["values"][idx]]),
        "move_idxs":   torch.cat([primary_data["train"]["move_idxs"], a["move_idxs"][idx]]),
        "visit_dists": torch.cat([primary_vd,                         vd[idx]]),
    }

    print(f"  anchor positions sampled: {n_sample:,} / {n_anchor:,}")
    print(f"  mixed train size: {len(mixed['tensors']):,}  "
          f"(was {n_primary:,}, +{n_sample:,} anchor)")

    result = dict(primary_data)
    result["train"] = mixed
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_epoch(model, loader, optimizer, is_train: bool, dense_policy: bool = False,
              policy_weight: float = 1.0):
    model.train(is_train)
    total_loss = total_vloss = total_ploss = 0.0
    n_batches = 0
    all_value_preds, all_value_targets = [], []
    policy_correct_top1 = policy_correct_top5 = policy_total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            if dense_policy:
                tensors, values, move_idxs, visit_dists = batch
                visit_dists = visit_dists.to(device)
            else:
                tensors, values, move_idxs = batch

            tensors   = tensors.float().to(device)   # uint8 → float32
            values    = values.to(device)
            move_idxs = move_idxs.to(device)

            value_pred, policy_logits = model(tensors)
            value_pred = value_pred.squeeze(1)

            vloss = F.mse_loss(value_pred, values)
            if dense_policy:
                # KL(visit_dist || softmax(logits)) — dense target from MCTS
                log_probs = F.log_softmax(policy_logits, dim=-1)
                ploss = -(visit_dists * log_probs).sum(dim=-1).mean()
            else:
                ploss = F.cross_entropy(policy_logits, move_idxs)

            loss = vloss + policy_weight * ploss

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


def _make_loader_from_data(data, split, batch_size, shuffle):
    d = data[split]
    ds = TensorDataset(
        d["tensors"], d["values"], d["move_idxs"], _ensure_visit_dists(d)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(device.type == "cuda"))


def train(dataset_path: str = None,
          out_dir: str = "models",
          epochs: int = 15,
          batch_size: int = 512,
          lr: float = 1e-3,
          patience: int = 5,
          tight_patience: int = 3,
          transition_drop: float = 0.5,
          seed: int = 42,
          init_model: str = None,
          anchor_dataset: str = None,
          anchor_frac: float = 0.15,
          policy_weight: float = 1.0,
          endgame_positions: int = 0,
          endgame_stages=None):
    """
    endgame_positions > 0: regenerate endgame positions each epoch instead of
    using a fixed dataset. Prevents position memorisation. endgame_stages
    controls which position classes to mix (e.g. [1,2] = KQK+KRK together).

    tight_patience: patience used after a phase transition is detected.
    transition_drop: fraction drop in val loss in one epoch that signals a
    phase transition (default 0.5 = 50% drop).
    """
    from generate_endgame import generate_positions, build_dataset as _build_eg

    _stages = endgame_stages if endgame_stages is not None else [1]
    if isinstance(_stages, int):
        _stages = [_stages]

    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    def _fresh_endgame_data():
        positions = generate_positions(endgame_positions, include_mirrors=True,
                                       stages=_stages)
        return _build_eg(positions)

    regenerate = endgame_positions > 0

    if regenerate:
        stage_str = "+".join(f"stage{s}" for s in _stages)
        print(f"Endgame curriculum: {stage_str}  "
              f"positions/epoch={endgame_positions:,} + mirrors")
        data = _fresh_endgame_data()
        dense_policy = True
    else:
        _, data, dense_policy = load_dataset(dataset_path)
        if anchor_dataset:
            data = mix_anchor(data, anchor_dataset, anchor_frac)

    train_loader = _make_loader_from_data(data, "train", batch_size, shuffle=True)
    val_loader   = _make_loader_from_data(data, "val",   batch_size, shuffle=False)

    model = PetraNet().to(device)
    if init_model:
        model.load_state_dict(torch.load(init_model, map_location=device, weights_only=True))
        print(f"Loaded starting weights from {init_model}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PetraNet: {n_params:,} parameters  |  device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, min_lr=1e-5
    )

    print(f"Patience: {patience} (tight={tight_patience} after >{transition_drop*100:.0f}% val drop)")

    print(f"\n{'Epoch':>5}  {'T-loss':>7}  {'V-loss':>7}  {'V-MSE':>6}  "
          f"{'V-R²':>6}  {'Top1':>5}  {'Top5':>5}  {'LR':>8}")
    print("-" * 66)

    best_val_loss   = float("inf")
    prev_val_loss   = float("inf")
    epochs_no_improve = 0
    current_patience  = patience
    phase_transition  = False

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Regenerate dataset each epoch to prevent position memorisation
        if regenerate and epoch > 1:
            data = _fresh_endgame_data()
            train_loader = _make_loader_from_data(data, "train", batch_size, shuffle=True)
            val_loader   = _make_loader_from_data(data, "val",   batch_size, shuffle=False)

        train_m = run_epoch(model, train_loader, optimizer, is_train=True,
                            dense_policy=dense_policy,
                            policy_weight=policy_weight)
        val_m   = run_epoch(model, val_loader,   optimizer, is_train=False,
                            dense_policy=dense_policy,
                            policy_weight=policy_weight)

        scheduler.step(val_m["loss"])
        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Phase transition detection: val loss drops >transition_drop in one epoch
        if not phase_transition and prev_val_loss < float("inf"):
            drop = (prev_val_loss - val_m["loss"]) / (prev_val_loss + 1e-8)
            if drop > transition_drop:
                phase_transition  = True
                current_patience  = tight_patience
                print(f"         *** Phase transition detected (val drop {drop*100:.0f}%) "
                      f"— switching to tight patience={tight_patience} ***")

        prev_val_loss = val_m["loss"]

        print(f"{epoch:>5}  "
              f"{train_m['loss']:>7.4f}  "
              f"{val_m['loss']:>7.4f}  "
              f"{val_m['value_loss']:>6.4f}  "
              f"{val_m['value_r2']:>6.3f}  "
              f"{val_m['top1']:>5.3f}  "
              f"{val_m['top5']:>5.3f}  "
              f"{lr_now:>8.2e}  "
              f"({elapsed:.0f}s)")

        torch.save(model.state_dict(), os.path.join(out_dir, "latest.pt"))

        if val_m["loss"] < best_val_loss:
            best_val_loss     = val_m["loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            print(f"         ↳ new best val loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= current_patience:
                print(f"\nEarly stopping: no improvement for {current_patience} epochs.")
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
    ap.add_argument("--dataset",     default=None,
                    help="Path to pre-built dataset (.pt). Required unless --endgame-positions > 0.")
    ap.add_argument("--out",         default="models")
    ap.add_argument("--epochs",      type=int,   default=15)
    ap.add_argument("--batch-size",  type=int,   default=512)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--patience",    type=int,   default=5)
    ap.add_argument("--tight-patience", type=int, default=3,
                    help="Patience after phase transition detected (default: 3)")
    ap.add_argument("--transition-drop", type=float, default=0.5,
                    help="Val loss drop fraction that triggers phase transition (default: 0.5)")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--init-model",  default=None,
                    help="Load these weights before training (zigzag fine-tuning)")
    ap.add_argument("--anchor-dataset", default=None,
                    help="Path to supervised SF dataset to mix in as an anchor "
                         "(prevents value collapse). Typically dataset_sf.pt.")
    ap.add_argument("--anchor-frac",   type=float, default=0.15,
                    help="Fraction of primary train size to sample from anchor (default: 0.15)")
    ap.add_argument("--policy-weight", type=float, default=1.0,
                    help="Weight for policy loss (default: 1.0). Set to 0 for pure "
                         "geometry/value training on endgame positions.")
    ap.add_argument("--endgame-positions", type=int, default=0,
                    help="If > 0, regenerate this many endgame positions each epoch "
                         "(prevents memorisation). Replaces --dataset for endgame curriculum.")
    ap.add_argument("--endgame-stages", type=int, nargs="+", default=[1],
                    help="Endgame stages to mix, space-separated: 1=KQK, 2=KRK "
                         "(e.g. --endgame-stages 1 2). Default: 1")
    args = ap.parse_args()

    if args.dataset is None and args.endgame_positions == 0:
        ap.error("--dataset is required unless --endgame-positions > 0")

    train(
        dataset_path=args.dataset,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        tight_patience=args.tight_patience,
        transition_drop=args.transition_drop,
        seed=args.seed,
        init_model=args.init_model,
        anchor_dataset=args.anchor_dataset,
        anchor_frac=args.anchor_frac,
        policy_weight=args.policy_weight,
        endgame_positions=args.endgame_positions,
        endgame_stages=args.endgame_stages,
    )


if __name__ == "__main__":
    main()
