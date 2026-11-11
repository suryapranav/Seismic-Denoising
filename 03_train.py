"""Step 3: train the U-Net at one noise level and evaluate it on the test split.

Modes (Table V ablation settings and the supervised reference):
  semiclean   noisy -> semiclean target          (the proposed method; alpha > 0)
  supervised  noisy -> clean target              (upper bound; needs clean data)
  noisy       noisy -> noisy target              (ablation settings 1 and 2; use --alpha 0 for setting 1)

Model selection uses validation loss against the *training-mode target only*. In semiclean
mode that means no clean data is ever used for training or choosing the epoch; clean
traces are used only to report test SNR at the end.

Examples:
  python 03_train.py --snr 0 --limit 300 --epochs 2      # smoke test, ~2 min
  python 03_train.py --snr 0                              # paper setting at 0 dB
  python 03_train.py --snr 0 --mode supervised            # plain MSE (alpha defaults to 0)
  python 03_train.py --snr 5 --tau 0.9                    # semiclean target keeping more energy
  python 03_train.py --snr 0 --mode semiclean --alpha 0   # ablation setting 3
"""
import argparse
import csv
import json
import time
from datetime import datetime

import numpy as np
import torch

from common import (HERE, add_gaussian_noise, from_stft, load_subset, semiclean_svd,
                    snr_db, split_by_event, stack_real_imag, standardize, to_stft,
                    unstack_real_imag)
from model import UNet, count_params, sparse_loss

# Per-noise-level L1 weights from the authors' released STFT configuration.
DEFAULT_ALPHA = {-5: 0.15, 0: 0.10, 5: 0.01}
PAPER = {  # Table II, Gaussian noise: (semiclean, proposed, supervised)
    -5: (-2.52, 4.59, 7.24), 0: (3.87, 8.84, 10.18), 5: (8.83, 12.30, 13.42)}


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_split(clean, snr, tau, rng, chunk=1000):
    """Noisy STFT input, semiclean STFT, clean STFT for one split (chunked to bound memory)."""
    noisy = add_gaussian_noise(clean, snr, rng)
    Y, X = to_stft(noisy), to_stft(clean)
    Sc = np.concatenate([semiclean_svd(Y[i:i + chunk], tau=tau)[0] for i in range(0, len(Y), chunk)])
    return Y, Sc, X


def batches(n, batch_size, shuffle, rng):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def run_epoch(model, X, T, alpha, device, batch_size, rng, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for b in batches(len(X), batch_size, training, rng):
            x = torch.from_numpy(X[b]).to(device)
            t = torch.from_numpy(T[b]).to(device)
            loss = sparse_loss(model(x), t, alpha)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(b)
            count += len(b)
    return total / count


@torch.no_grad()
def predict(model, X, device, batch_size=128):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + batch_size]).to(device)).cpu().numpy()
           for i in range(0, len(X), batch_size)]
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=int, choices=[-5, 0, 5], required=True)
    ap.add_argument("--mode", choices=["semiclean", "supervised", "noisy"], default="semiclean")
    ap.add_argument("--alpha", type=float, default=None, help="L1 weight (default: per-SNR value)")
    ap.add_argument("--tau", type=float, default=0.7, help="SVD energy threshold")
    ap.add_argument("--width", type=int, default=64, help="64 = paper model")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5, help="early stopping (paper: 5)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None, help="use only N traces (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.alpha is not None:
        alpha = args.alpha
    elif args.mode == "supervised":
        alpha = 0.0   # clean targets: plain MSE. The L1 sparsity term belongs to the unsupervised method
    else:
        alpha = DEFAULT_ALPHA[args.snr]
    tag = (f"{args.mode}_snr{args.snr:+d}_a{alpha:g}_t{args.tau:g}_w{args.width}"
           + (f"_lim{args.limit}" if args.limit else ""))

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device()
    t0 = time.time()

    # ---- data -------------------------------------------------------------
    waves, meta = load_subset()
    if args.limit:
        keep = np.random.default_rng(0).choice(len(waves), args.limit, replace=False)
        waves, meta = waves[keep], meta.iloc[keep].reset_index(drop=True)
    train, val, test = split_by_event(meta)
    clean = standardize(waves)
    noise_rng = np.random.default_rng(1000 + args.snr)
    data = {name: build_split(clean[idx], args.snr, args.tau, noise_rng)
            for name, idx in [("train", train), ("val", val), ("test", test)]}

    target_of = {"semiclean": 1, "supervised": 2, "noisy": 0}[args.mode]
    Xtr, Ttr = stack_real_imag(data["train"][0]), stack_real_imag(data["train"][target_of])
    Xva, Tva = stack_real_imag(data["val"][0]), stack_real_imag(data["val"][target_of])
    print(f"Device {device} | {args.mode} @ {args.snr:+d} dB | alpha {alpha} tau {args.tau} | "
          f"train {len(Xtr):,} val {len(Xva):,} test {len(test):,} | data ready in {time.time() - t0:.0f}s")

    # ---- model ------------------------------------------------------------
    model = UNet(width=args.width).to(device)
    print(f"U-Net width {args.width}: {count_params(model):,} parameters")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)   # halve LR every 10 epochs

    ckpt_dir = HERE / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt = ckpt_dir / f"{tag}.pt"
    history, best, wait = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        tr = run_epoch(model, Xtr, Ttr, alpha, device, args.batch, rng, opt)
        va = run_epoch(model, Xva, Tva, alpha, device, args.batch, rng)
        sched.step()
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        improved = va < best
        if improved:
            best, wait = va, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
        print(f"  epoch {epoch:>2}  train {tr:.5f}  val {va:.5f}{'  *' if improved else ''}  "
              f"({time.time() - te:.0f}s)")
        if wait >= args.patience:
            print(f"  early stop: no val improvement for {args.patience} epochs")
            break

    # ---- test (clean data used here only) ---------------------------------
    model.load_state_dict(torch.load(ckpt, map_location=device))
    Y, Sc, X = data["test"]
    clean_ref = from_stft(X)
    denoised = from_stft(unstack_real_imag(predict(model, stack_real_imag(Y), device)))
    s_in = snr_db(clean_ref, from_stft(Y)).mean()
    s_sc = snr_db(clean_ref, from_stft(Sc)).mean()
    s_out_all = snr_db(clean_ref, denoised)
    s_out = s_out_all.mean()

    p_sc, p_prop, p_sup = PAPER[args.snr]
    p_ref = p_sup if args.mode == "supervised" else p_prop
    print(f"\nTest SNR ({len(test):,} traces): input {s_in:.2f} | semiclean {s_sc:.2f} (paper {p_sc}) | "
          f"model {s_out:.2f} +/- {s_out_all.std():.2f} (paper {p_ref}) | gain {s_out - s_in:+.2f} dB")

    # ---- log ------------------------------------------------------------------
    res_dir = HERE / "results"
    res_dir.mkdir(exist_ok=True)
    row = {"time": datetime.now().isoformat(timespec="seconds"), "tag": tag, "mode": args.mode,
           "snr_in_target": args.snr, "alpha": alpha, "tau": args.tau, "width": args.width,
           "params": count_params(model), "limit": args.limit or "", "epochs_run": len(history),
           "best_val_loss": round(best, 6), "test_snr_input": round(s_in, 2),
           "test_snr_semiclean": round(s_sc, 2), "test_snr_model": round(s_out, 2),
           "paper_snr_model": p_ref, "minutes": round((time.time() - t0) / 60, 1), "device": str(device)}
    path = res_dir / "results.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    (res_dir / f"{tag}_history.json").write_text(json.dumps(history, indent=1))
    print(f"Logged to results/results.csv; checkpoint {ckpt.name}")


if __name__ == "__main__":
    main()
