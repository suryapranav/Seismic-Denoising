"""Step 1: pull a clean 10K-trace working set out of STEAD chunk 2.

Selection
- vertical (Z) component only: the paper denoises single traces
- recorded Z-component SNR > 40 dB, so the "clean" reference is genuinely clean
  (inherent noise sits >= 35 dB below the noise we add at -5..+5 dB)
- random 10,000 of the ~23K that qualify, fixed seed

Output (in data/):  stead_z_10k.npy  (10000, 6000) float32,  stead_z_10k_meta.csv
Usage:  python3 01_extract_subset.py
"""
import time

import h5py
import numpy as np
import pandas as pd

from common import DATA, HERE, TRACE_LEN

SRC = HERE / "chunk2"
N_TRACES = 10_000
MIN_SNR_Z = 40.0
Z = 2          # STEAD channel order: E, N, Z
SEED = 42


def main():
    t0 = time.time()
    meta = pd.read_csv(SRC / "chunk2.csv", low_memory=False)
    snr = meta["snr_db"].str.strip("[]").str.split(expand=True).astype(float)
    meta["snr_z_db"] = snr[2]
    ok = (meta["trace_category"] == "earthquake_local") & (meta["snr_z_db"] > MIN_SNR_Z)
    candidates = meta[ok]
    print(f"{len(meta):,} traces in chunk; {len(candidates):,} with Z SNR > {MIN_SNR_Z} dB")

    pick = candidates.sample(n=N_TRACES, random_state=SEED).reset_index(drop=True)

    waves = np.empty((N_TRACES, TRACE_LEN), dtype=np.float32)
    with h5py.File(SRC / "chunk2.hdf5", "r") as f:
        g = f["data"]
        for i, name in enumerate(pick["trace_name"]):
            waves[i] = g[name][:, Z]
            if (i + 1) % 2000 == 0:
                print(f"  read {i + 1:,} / {N_TRACES:,}")

    # Sanity checks: no NaNs, no flat traces
    assert np.isfinite(waves).all(), "NaN/inf in waveforms"
    flat = (waves.std(axis=1) == 0).sum()
    assert flat == 0, f"{flat} flat traces"

    DATA.mkdir(exist_ok=True)
    np.save(DATA / "stead_z_10k.npy", waves)
    cols = ["trace_name", "source_id", "network_code", "receiver_code", "source_magnitude",
            "source_distance_km", "p_arrival_sample", "s_arrival_sample", "snr_z_db"]
    pick[cols].to_csv(DATA / "stead_z_10k_meta.csv", index=False)
    print(f"Saved {waves.shape} to data/ in {time.time() - t0:.0f}s; "
          f"{pick['source_id'].nunique():,} distinct earthquakes")


if __name__ == "__main__":
    main()
