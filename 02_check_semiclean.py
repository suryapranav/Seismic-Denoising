"""Step 2: check the preprocessing + SVD semiclean stage against the paper.

For each input SNR (-5, 0, +5 dB) on the held-out test split:
  clean -> add Gaussian noise -> STFT (78x78) -> SVD semiclean -> inverse STFT
and report per-trace SNR averaged over traces (the paper's metric).

Paper, Table II (pseudosynthetic STEAD, Gaussian noise), self-noise cancellation column:
  -5 dB -> -2.52    0 dB -> 3.87    +5 dB -> 8.83

Usage:  python3 02_check_semiclean.py
"""
import numpy as np

from common import (add_gaussian_noise, from_stft, load_subset, semiclean_svd,
                    snr_db, split_by_event, standardize, to_stft)

PAPER_SEMICLEAN = {-5: -2.52, 0: 3.87, 5: 8.83}


def main():
    waves, meta = load_subset()
    train, val, test = split_by_event(meta)
    print(f"Split by earthquake: train {len(train):,} | val {len(val):,} | test {len(test):,} traces")

    clean = standardize(waves[test])
    clean_ref = from_stft(to_stft(clean))        # clean passed through STFT/ISTFT, as the authors do
    rng = np.random.default_rng(0)

    print(f"\n{'input':>6} {'measured':>9} | {'semiclean tau=0.7':>18} {'(paper rule)':>13} "
          f"{'tau=0.9':>8} | {'paper':>6}  {'kept comps (mean)':>17}")
    for snr_in in (-5, 0, 5):
        noisy = add_gaussian_noise(clean, snr_in, rng)
        Y = to_stft(noisy)
        measured = snr_db(clean_ref, from_stft(Y)).mean()

        sc_code, k_code = semiclean_svd(Y, tau=0.7, rule="code")  # explicit: authors' released code
        sc_paper, _ = semiclean_svd(Y, tau=0.7, rule="paper")
        sc_09, _ = semiclean_svd(Y, tau=0.9, rule="code")

        s_code = snr_db(clean_ref, from_stft(sc_code)).mean()
        s_paper = snr_db(clean_ref, from_stft(sc_paper)).mean()
        s_09 = snr_db(clean_ref, from_stft(sc_09)).mean()
        print(f"{snr_in:>+5} dB {measured:>8.2f}  | {s_code:>18.2f} {s_paper:>13.2f} {s_09:>8.2f} | "
              f"{PAPER_SEMICLEAN[snr_in]:>6.2f}  {k_code.mean():>17.1f}")


if __name__ == "__main__":
    main()
