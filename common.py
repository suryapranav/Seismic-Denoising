"""Shared signal-processing pieces for the noise-to-semiclean denoising pipeline.

Implements Sections II-A and II-B of Almasri et al., "Resilient Unsupervised Deep
Learning for Data Denoising via Noise-to-Semiclean Training", IEEE TGRS 2026.
"""
from pathlib import Path

import numpy as np
from scipy.signal import istft, stft

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

TRACE_LEN = 6000          # STEAD: 60 s at 100 Hz
NPERSEG = 155             # Hann window length -> 78 freq bins x 78 frames (paper: 78 x 78)
WINDOW = "hann"           # scipy default overlap = NPERSEG // 2 (hop of 78 samples)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_subset():
    """Clean Z-component traces (N, 6000) float32 and their metadata."""
    import pandas as pd
    waves = np.load(DATA / "stead_z_10k.npy")
    meta = pd.read_csv(DATA / "stead_z_10k_meta.csv")
    return waves, meta


def standardize(x):
    """Per-trace zero mean, unit variance (same as the authors' StandardScaler use)."""
    x = x - x.mean(axis=-1, keepdims=True)
    return (x / x.std(axis=-1, keepdims=True)).astype(np.float32)


def split_by_event(meta, test_frac=0.15, val_frac=0.20, seed=42):
    """Train / val / test index arrays, grouped by earthquake (source_id).

    The same earthquake is usually recorded by several stations. Splitting by
    trace would let near-identical waveforms land in both train and test, which
    inflates test scores. Splitting by event prevents that leakage.
    Proportions mirror the authors' code: 15% test, then 20% of the rest for val.
    """
    rng = np.random.default_rng(seed)
    events = meta["source_id"].astype(str).to_numpy()
    uniq = rng.permutation(np.unique(events))
    n_test = int(round(test_frac * len(uniq)))
    n_val = int(round(val_frac * (len(uniq) - n_test)))
    test_ev, val_ev = set(uniq[:n_test]), set(uniq[n_test:n_test + n_val])
    idx = np.arange(len(events))
    test = idx[np.isin(events, list(test_ev))]
    val = idx[np.isin(events, list(val_ev))]
    train = idx[~np.isin(events, list(test_ev | val_ev))]
    return train, val, test


# ---------------------------------------------------------------------------
# Noise (Eq. 1: y = x + eta)
# ---------------------------------------------------------------------------
def add_gaussian_noise(x, snr_db, rng):
    """Add white Gaussian noise so each trace has exactly the requested SNR on average."""
    p_signal = np.mean(x ** 2, axis=-1, keepdims=True)
    p_noise = p_signal / (10 ** (snr_db / 10))
    noise = rng.standard_normal(x.shape).astype(np.float32) * np.sqrt(p_noise)
    return (x + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Time-frequency transform (Eq. 2 and Eq. 7)
# ---------------------------------------------------------------------------
def to_stft(x):
    """(N, 6000) real -> (N, 78, 78) complex STFT."""
    _, _, z = stft(x, window=WINDOW, nperseg=NPERSEG)
    return z.astype(np.complex64)


def from_stft(z):
    """(N, 78, 78) complex -> (N, 6000) real via overlap-add inverse STFT."""
    _, x = istft(z, window=WINDOW, nperseg=NPERSEG)
    return x[..., :TRACE_LEN].astype(np.float32)


def stack_real_imag(z):
    """Complex (N, F, T) -> float (N, 2, F, T), channels-first for PyTorch."""
    return np.stack([z.real, z.imag], axis=1).astype(np.float32)


def unstack_real_imag(a):
    """Float (N, 2, F, T) -> complex (N, F, T)."""
    return (a[:, 0] + 1j * a[:, 1]).astype(np.complex64)


# ---------------------------------------------------------------------------
# Self-noise cancellation (Section II-B, Eqs. 3-5)
# ---------------------------------------------------------------------------
def semiclean_svd(Y, tau=0.7, rule="paper"):
    """Project each noisy STFT onto its dominant (signal) subspace.

    Steps from the paper:
      1. C = Y Y^H                      (autocorrelation, F x F)
      2. C = U diag(lambda) V^H         (SVD)
      3. keep the leading components holding a fraction tau of sum(lambda)
      4. P = I - S S^H, S = discarded (noise) vectors  == U_keep U_keep^H
      5. Sc = P Y

    rule="code":  keep k components, k = first index where cumulative energy >= tau
                  (matches the authors' released code: np.argmax(energy >= tau))
    rule="paper": keep j = k + 1 components, the smallest j with E(j) >= tau
                  (the paper's text). The two differ by one component.
    """
    Y = np.asarray(Y, dtype=np.complex64)
    C = Y @ np.conj(np.swapaxes(Y, -1, -2))                    # (N, F, F)
    U, lam, _ = np.linalg.svd(C)                               # batched, lam descending
    energy = np.cumsum(lam, axis=-1) / lam.sum(axis=-1, keepdims=True)
    k = np.argmax(energy >= tau, axis=-1)                      # first index meeting tau
    n_keep = k if rule == "code" else k + 1
    keep = (np.arange(lam.shape[-1])[None, :] < n_keep[:, None]).astype(np.complex64)
    U_keep = U * keep[:, None, :]                              # zero out noise vectors
    P = U_keep @ np.conj(np.swapaxes(U_keep, -1, -2))          # signal-subspace projector
    return (P @ Y).astype(np.complex64), n_keep


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def snr_db(clean, estimate):
    """Per-trace SNR (dB) of an estimate against the clean trace, as in the paper."""
    p_sig = np.mean(clean ** 2, axis=-1)
    p_err = np.mean((estimate - clean) ** 2, axis=-1)
    return 10 * np.log10(p_sig / p_err)
