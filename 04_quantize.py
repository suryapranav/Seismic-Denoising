"""Step 4: post-training quantization of a trained U-Net, simulated in PyTorch.

Simulates the arithmetic a fixed-point FPGA datapath would perform ("fake quantization"):
- weights: symmetric integers, one scale per output channel          (--bits)
- activations: the model input and every convolution output are rounded to symmetric
  integers, one scale per tensor, fixed after calibration           (--act-bits)
- the model's input and final output layer can get their own width   (--io-bits)
- biases stay in higher precision (on hardware they live in the wide accumulator)

Calibration uses noisy *training* inputs only: no clean data and no test data are touched
before the final evaluation.

--sensitivity quantizes ONE activation point at a time (everything else float) and reports
how much each costs, to find which layers need more bits.

Examples:
  python 04_quantize.py --ckpt checkpoints/semiclean_snr+0_a0.1_t0.7_w8.pt --snr 0 --width 8
  python 04_quantize.py --ckpt ... --snr 0 --width 8 --sensitivity
  python 04_quantize.py --ckpt ... --snr 0 --width 8 --io-bits 16          # int8 inside, int16 at the edges
  python 04_quantize.py --ckpt ... --snr 0 --width 8 --act-bits 16         # int8 weights, int16 activations
  python 04_quantize.py --ckpt ... --snr 0 --width 8 --pct 100              # range = true max: no clipping
  python 04_quantize.py --ckpt ... --snr 0 --width 8 --pct 100 --io-bits 16 --wide input,enc1,dec4,out
"""
import argparse
import csv
import importlib
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from common import HERE, from_stft, load_subset, snr_db, split_by_event, stack_real_imag, standardize, unstack_real_imag
from model import UNet, count_params

train_mod = importlib.import_module("03_train")      # reuse build_split, predict, pick_device


def fake_quant(x, scale, bits):
    """Round to a symmetric signed integer grid and map back to float."""
    qmax = 2 ** (bits - 1) - 1
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


def quantize_weights(model, bits):
    """In place: per-output-channel symmetric quantization of every conv weight."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                w = m.weight
                out_dim = 0 if isinstance(m, nn.Conv2d) else 1   # ConvTranspose2d: (in, out, k, k)
                reduce = [d for d in range(w.dim()) if d != out_dim]
                amax = w.abs().amax(dim=reduce, keepdim=True).clamp_min(1e-12)
                w.copy_(fake_quant(w, amax / (2 ** (bits - 1) - 1), bits))


def robust_absmax(x, pct, max_samples=1_000_000):
    """pct-th percentile of |x|, on a random subsample if x is large."""
    flat = x.detach().abs().flatten().float()
    if flat.numel() > max_samples:
        flat = flat[torch.randint(0, flat.numel(), (max_samples,), device=flat.device)]
    return torch.quantile(flat.cpu(), pct / 100).item()      # quantile on CPU: not all GPU backends support it


class ActivationQuantizer:
    """Calibrates, then applies, per-tensor rounding at the input and at every conv output.

    Activation points are named "input" and the conv's attribute name (enc1, ..., up1, ..., out).
    `bits[name]` sets each point's width; `enabled` limits which points are quantized.
    """

    def __init__(self, model, pct):
        self.pct = pct
        self.modules = {name: m for name, m in model.named_modules()
                        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))}
        self.names = ["input"] + list(self.modules)
        self.stats = {n: [] for n in self.names}
        self.absmax, self.bits, self.enabled = {}, {}, set()
        self.calibrating = True
        for name, m in self.modules.items():
            m.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if self.calibrating:
                self.stats[name].append(robust_absmax(output, self.pct))
                return output
            return self._q(name, output)
        return hook

    def _q(self, name, x):
        if name not in self.enabled:
            return x
        qmax = 2 ** (self.bits[name] - 1) - 1
        return fake_quant(x, max(self.absmax[name], 1e-12) / qmax, self.bits[name])

    def observe_input(self, x):
        self.stats["input"].append(robust_absmax(x, self.pct))

    def freeze(self, bits):
        # pct < 100: average the per-batch percentiles (robust to outliers, but clips the extreme values)
        # pct = 100: take the largest value seen in any batch (no clipping on data like the calibration set)
        agg = np.max if self.pct >= 100 else np.mean
        self.absmax = {n: float(agg(v)) for n, v in self.stats.items()}
        self.bits = dict(bits)
        self.enabled = set(self.names)
        self.calibrating = False

    def quantize_input(self, x):
        return self._q("input", x)


@torch.no_grad()
def predict_quantized(model, aq, X, device, batch=128):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        x = aq.quantize_input(torch.from_numpy(X[i:i + batch]).to(device))
        out.append(model(x).cpu().numpy())
    return np.concatenate(out)


def test_snr(pred_stacked, clean_ref):
    return snr_db(clean_ref, from_stft(unstack_real_imag(pred_stacked))).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--snr", type=int, choices=[-5, 0, 5], required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--bits", type=int, default=8, help="weight bits")
    ap.add_argument("--act-bits", type=int, default=None, help="activation bits (default: same as --bits)")
    ap.add_argument("--io-bits", type=int, default=None, help="bits for the --wide activation points "
                                                              "(default: same as --act-bits)")
    ap.add_argument("--wide", default="input,out", help="comma-separated activation points that get --io-bits "
                                                         "(e.g. input,enc1,dec4,out)")
    ap.add_argument("--sensitivity", action="store_true", help="quantize one activation point at a time")
    ap.add_argument("--calib-batches", type=int, default=20, help="x64 noisy training inputs")
    ap.add_argument("--pct", type=float, default=99.99, help="percentile of |activation| used as range; "
                                                              "100 = true maximum (no clipping)")
    args = ap.parse_args()
    act_bits = args.act_bits or args.bits
    io_bits = args.io_bits or act_bits
    device = train_mod.pick_device()
    t0 = time.time()

    # Rebuild exactly the noisy data the model was trained/tested on (same seeds, same order)
    waves, meta = load_subset()
    train, val, test = split_by_event(meta)
    clean = standardize(waves)
    noise_rng = np.random.default_rng(1000 + args.snr)
    data = {name: train_mod.build_split(clean[idx], args.snr, args.tau, noise_rng)
            for name, idx in [("train", train), ("val", val), ("test", test)]}
    Xcal = stack_real_imag(data["train"][0][: args.calib_batches * 64])   # noisy inputs only
    Y, _, X = data["test"]
    Xte, clean_ref = stack_real_imag(Y), from_stft(X)
    print(f"Device {device} | data ready in {time.time() - t0:.0f}s | calibration {len(Xcal):,} noisy "
          f"training inputs | test {len(Xte):,}")

    ckpt = HERE / args.ckpt if not args.ckpt.startswith("/") else args.ckpt
    model = UNet(width=args.width).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    n_params = count_params(model)

    # 1) float32 reference
    s_float = test_snr(train_mod.predict(model, Xte, device), clean_ref)

    # 2) quantized weights only
    quantize_weights(model, args.bits)
    s_w = test_snr(train_mod.predict(model, Xte, device), clean_ref)

    # 3) calibrate activation ranges (with quantized weights) on noisy training inputs
    aq = ActivationQuantizer(model, args.pct)
    with torch.no_grad():
        for i in range(0, len(Xcal), 64):
            x = torch.from_numpy(Xcal[i:i + 64]).to(device)
            aq.observe_input(x)
            model(x)
    wide = [w.strip() for w in args.wide.split(",") if w.strip()]
    unknown = set(wide) - set(aq.names)
    assert not unknown, f"unknown --wide points {unknown}; valid: {aq.names}"
    bits = {n: (io_bits if n in wide else act_bits) for n in aq.names}
    aq.freeze(bits)

    # 4) all activation points quantized
    s_wa = test_snr(predict_quantized(model, aq, Xte, device), clean_ref)

    n_weights = sum(m.weight.numel() for m in aq.modules.values())
    n_bias = n_params - n_weights
    size_float = 4 * n_params
    size_q = n_weights * args.bits / 8 + 4 * n_bias
    io_note = f", int{io_bits} at {'/'.join(wide)}" if io_bits != act_bits else ""
    print(f"\nWidth {args.width} ({n_params:,} parameters) @ {args.snr:+d} dB input")
    print(f"  float32                                {s_float:6.2f} dB   weights {size_float / 1024:7.1f} KB")
    print(f"  int{args.bits} weights                          {s_w:6.2f} dB   ({s_w - s_float:+.2f})")
    print(f"  int{args.bits} weights + int{act_bits} activations{io_note}  {s_wa:6.2f} dB   "
          f"({s_wa - s_float:+.2f})   weights {size_q / 1024:7.1f} KB ({size_float / size_q:.1f}x smaller)")

    # 5) optional: cost of each activation point on its own
    sens = {}
    if args.sensitivity:
        print(f"\nSensitivity: only ONE activation point quantized (int{act_bits}), weights int{args.bits}")
        for name in aq.names:
            aq.enabled = {name}
            aq.bits[name] = act_bits
            sens[name] = test_snr(predict_quantized(model, aq, Xte, device), clean_ref) - s_w
            print(f"  {name:<11} {sens[name]:+6.2f} dB  range +/-{aq.absmax[name]:.4g}")
        print(f"  sum of individual costs: {sum(sens.values()):+.2f} dB (all at once: {s_wa - s_w:+.2f})")

    res = HERE / "results"
    res.mkdir(exist_ok=True)
    path = res / "quant_results.csv"
    row = {"time": datetime.now().isoformat(timespec="seconds"), "ckpt": args.ckpt, "snr_in_target": args.snr,
           "width": args.width, "params": n_params, "w_bits": args.bits, "act_bits": act_bits,
           "io_bits": io_bits, "wide": "/".join(wide), "pct": args.pct, "calib_inputs": len(Xcal), "snr_float": round(s_float, 2),
           "snr_w_only": round(s_w, 2), "snr_w_and_act": round(s_wa, 2),
           "size_float_kb": round(size_float / 1024, 1), "size_quant_kb": round(size_q / 1024, 1),
           "device": str(device)}
    new = not path.exists()
    if not new:
        with open(path) as f:
            header = f.readline().strip().split(",")
        if header != list(row):                       # columns changed since an older run: start a new file
            path = res / "quant_results_v3.csv"
            new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    if sens:
        (res / f"quant_sensitivity_w{args.width}_snr{args.snr:+d}_a{act_bits}.csv").write_text(
            "point,snr_change_db,abs_range\n" + "".join(f"{n},{v:.3f},{aq.absmax[n]:.6g}\n" for n, v in sens.items()))
    print(f"Logged to results/{path.name} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
