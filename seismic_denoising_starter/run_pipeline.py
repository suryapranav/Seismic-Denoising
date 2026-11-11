#!/usr/bin/env python3
"""Run project stages 1–3: input, AWGN, STFT + SVD semiclean target."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.seismic_pipeline import (
    add_awgn_at_snr,
    calculate_snr_db,
    generate_synthetic_trace,
    inverse_stft_trace,
    load_trace_csv,
    stft_trace,
    svd_semiclean_target,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "csv"], default="synthetic")
    parser.add_argument("--input", type=str, default=None, help="CSV path when --mode csv")
    parser.add_argument("--column", type=int, default=0, help="0-based CSV column")
    parser.add_argument("--skip-rows", type=int, default=0, help="CSV header rows to skip")
    parser.add_argument("--sample-rate", type=float, default=100.0, help="Samples per second (Hz)")
    parser.add_argument("--duration", type=float, default=10.0, help="Synthetic trace duration in seconds")
    parser.add_argument("--events", type=int, default=8, help="Number of synthetic seismic pulses")
    parser.add_argument("--snr-db", type=float, default=0.0, help="Input SNR in dB")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--energy-threshold", type=float, default=0.95)
    parser.add_argument("--nperseg", type=int, default=128, help="STFT segment length")
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "synthetic":
        time, clean = generate_synthetic_trace(
            duration_s=args.duration,
            sample_rate_hz=args.sample_rate,
            n_events=args.events,
            seed=args.seed,
        )
    else:
        if not args.input:
            raise SystemExit("For --mode csv, provide --input path/to/trace.csv")
        clean = load_trace_csv(args.input, column=args.column, skip_rows=args.skip_rows)
        time = np.arange(clean.size) / args.sample_rate
        # Normalize only for stable plotting and predictable noise scaling.
        peak = np.max(np.abs(clean))
        if peak > 0:
            clean = clean / peak

    noisy, noise = add_awgn_at_snr(clean, args.snr_db, seed=args.seed + 1)
    measured_input_snr = calculate_snr_db(clean, noisy)

    freqs, stft_times, noisy_stft = stft_trace(
        noisy, args.sample_rate, nperseg=args.nperseg
    )
    semiclean_stft, diagnostics = svd_semiclean_target(
        noisy_stft, energy_threshold=args.energy_threshold
    )
    _, semiclean = inverse_stft_trace(
        semiclean_stft,
        sample_rate_hz=args.sample_rate,
        nperseg=args.nperseg,
        expected_length=clean.size,
    )

    # Recalculate output SNR for synthetic data only, where clean ground truth exists.
    output_snr = calculate_snr_db(clean, semiclean)

    np.savez_compressed(
        out_dir / "trace_data.npz",
        time=time,
        clean=clean,
        noisy=noisy,
        noise=noise,
        semiclean=semiclean,
        sample_rate_hz=args.sample_rate,
        requested_input_snr_db=args.snr_db,
        measured_input_snr_db=measured_input_snr,
        output_snr_db=output_snr,
        retained_components=diagnostics["retained_components"],
        retained_energy_fraction=diagnostics["retained_energy_fraction"],
    )

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(time, clean, linewidth=1)
    axes[0].set_title("Clean reference trace")
    axes[1].plot(time, noisy, linewidth=0.8)
    axes[1].set_title(f"Noisy trace (requested SNR {args.snr_db:g} dB; measured {measured_input_snr:.2f} dB)")
    axes[2].plot(time, semiclean, linewidth=0.8)
    axes[2].set_title(
        f"SVD semiclean reconstruction ({diagnostics['retained_components']}/"
        f"{diagnostics['total_components']} components retained)"
    )
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / "pipeline_plot.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    mesh = ax.pcolormesh(
        stft_times, freqs, np.abs(noisy_stft), shading="auto"
    )
    ax.set_title("Magnitude of noisy-trace STFT")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=ax, label="Magnitude")
    fig.savefig(out_dir / "stft_noisy.png", dpi=160)
    plt.close(fig)

    print("Pipeline completed.")
    print(f"Mode: {args.mode}")
    print(f"Samples: {clean.size}; sample rate: {args.sample_rate:g} Hz")
    print(f"Requested input SNR: {args.snr_db:g} dB")
    print(f"Measured input SNR: {measured_input_snr:.2f} dB")
    print(f"SVD components retained: {diagnostics['retained_components']} / {diagnostics['total_components']}")
    print(f"Retained cumulative energy: {diagnostics['retained_energy_fraction']:.4f}")
    print(f"Output SNR against clean reference: {output_snr:.2f} dB")
    print(f"Files saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
