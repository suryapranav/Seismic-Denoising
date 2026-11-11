"""Reusable functions for the first three stages of seismic denoising."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.signal import stft, istft


def generate_synthetic_trace(
    duration_s: float = 10.0,
    sample_rate_hz: float = 100.0,
    n_events: int = 8,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, clean_trace), made from simple Ricker-like seismic pulses."""
    if duration_s <= 0 or sample_rate_hz <= 0:
        raise ValueError("duration_s and sample_rate_hz must be positive.")
    if n_events < 1:
        raise ValueError("n_events must be at least 1.")

    rng = np.random.default_rng(seed)
    n = int(round(duration_s * sample_rate_hz))
    time = np.arange(n) / sample_rate_hz
    clean = np.zeros(n, dtype=float)

    # Keep events away from the boundaries so each pulse is visible.
    centers = rng.uniform(0.5, max(0.51, duration_s - 0.5), size=n_events)
    frequencies = rng.uniform(3.0, 12.0, size=n_events)
    amplitudes = rng.uniform(0.5, 1.0, size=n_events) * rng.choice([-1.0, 1.0], size=n_events)

    for center, freq, amp in zip(centers, frequencies, amplitudes):
        tau = time - center
        a = (np.pi * freq * tau) ** 2
        pulse = (1.0 - 2.0 * a) * np.exp(-a)  # Ricker wavelet
        clean += amp * pulse

    peak = np.max(np.abs(clean))
    if peak > 0:
        clean = clean / peak
    return time, clean


def load_trace_csv(
    path: str | Path,
    column: int = 0,
    skip_rows: int = 0,
) -> np.ndarray:
    """Load one numeric trace column from a CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    data = np.genfromtxt(path, delimiter=",", skip_header=skip_rows, dtype=float)
    if data.size == 0:
        raise ValueError("CSV contains no numeric data.")
    if data.ndim == 1:
        if column != 0:
            raise ValueError("CSV appears to have one column; use --column 0.")
        trace = data
    else:
        if not 0 <= column < data.shape[1]:
            raise ValueError(f"column must be between 0 and {data.shape[1] - 1}.")
        trace = data[:, column]

    trace = np.asarray(trace, dtype=float).reshape(-1)
    trace = trace[np.isfinite(trace)]
    if trace.size < 16:
        raise ValueError("Need at least 16 finite samples in the input trace.")
    if np.std(trace) == 0:
        raise ValueError("Input trace is constant; cannot add meaningful SNR-controlled noise.")
    return trace


def add_awgn_at_snr(
    clean_trace: np.ndarray,
    snr_db: float,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (noisy_trace, noise) with measured SNR close to snr_db."""
    x = np.asarray(clean_trace, dtype=float).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("clean_trace must be non-empty and finite.")

    signal_power = float(np.mean(x**2))
    if signal_power <= 0:
        raise ValueError("clean_trace must have non-zero power.")

    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=1.0, size=x.size)
    # Rescale this particular random realization to match the requested SNR.
    noise_power = float(np.mean(noise**2))
    noise *= np.sqrt(target_noise_power / noise_power)
    return x + noise, noise


def calculate_snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """SNR in dB, comparing reference with estimate as the error-free signal."""
    ref = np.asarray(reference, dtype=float).reshape(-1)
    est = np.asarray(estimate, dtype=float).reshape(-1)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have the same shape.")
    error_power = float(np.mean((ref - est) ** 2))
    signal_power = float(np.mean(ref**2))
    if error_power == 0:
        return float("inf")
    if signal_power == 0:
        return float("-inf")
    return 10.0 * np.log10(signal_power / error_power)


def stft_trace(
    trace: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the complex STFT; returns (frequencies, times, complex_matrix)."""
    x = np.asarray(trace, dtype=float).reshape(-1)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    if x.size < 16:
        raise ValueError("Trace is too short for STFT.")
    nperseg = int(min(max(16, nperseg), x.size))
    noverlap = min(nperseg // 2, nperseg - 1)
    frequencies, times, matrix = stft(
        x,
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    return frequencies, times, matrix


def svd_semiclean_target(
    noisy_stft: np.ndarray,
    energy_threshold: float = 0.95,
) -> tuple[np.ndarray, dict]:
    """
    Build a semiclean STFT target using a configurable SVD energy heuristic.

    Y has shape [frequency_bins, time_frames]. Form C = Y @ Yᴴ, compute its
    eigenspace via SVD, retain the leading components reaching the requested
    cumulative energy, then project Y into that retained subspace.

    This is a starter implementation. Verify the exact autocorrelation matrix,
    energy criterion, and component-selection convention against the source paper.
    """
    y = np.asarray(noisy_stft)
    if y.ndim != 2 or y.size == 0:
        raise ValueError("noisy_stft must be a non-empty 2D matrix.")
    if not 0 < energy_threshold <= 1:
        raise ValueError("energy_threshold must be in (0, 1].")

    # Frequency-domain autocorrelation matrix. Hermitian transpose is important
    # because the STFT is complex-valued.
    c = y @ y.conj().T
    u, singular_values, _ = np.linalg.svd(c, full_matrices=False)

    energy = np.maximum(singular_values.real, 0.0)
    total_energy = float(np.sum(energy))
    if total_energy <= 0:
        raise ValueError("SVD matrix has zero energy.")

    cumulative = np.cumsum(energy) / total_energy
    k = int(np.searchsorted(cumulative, energy_threshold, side="left") + 1)
    k = min(max(k, 1), u.shape[1])

    dominant_basis = u[:, :k]
    projection = dominant_basis @ dominant_basis.conj().T
    semiclean = projection @ y

    diagnostics = {
        "retained_components": k,
        "total_components": int(len(singular_values)),
        "energy_threshold": float(energy_threshold),
        "retained_energy_fraction": float(cumulative[k - 1]),
        "singular_values": singular_values,
    }
    return semiclean, diagnostics


def inverse_stft_trace(
    matrix: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 128,
    expected_length: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a real-valued time trace from a complex STFT matrix."""
    z = np.asarray(matrix)
    nperseg = int(min(max(16, nperseg), expected_length or max(16, (z.shape[1] - 1) * 2)))
    noverlap = min(nperseg // 2, nperseg - 1)
    time, reconstructed = istft(
        z,
        fs=sample_rate_hz,
        nperseg=nperseg,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    reconstructed = np.real(reconstructed)
    if expected_length is not None:
        if reconstructed.size < expected_length:
            reconstructed = np.pad(reconstructed, (0, expected_length - reconstructed.size))
        reconstructed = reconstructed[:expected_length]
        time = np.arange(expected_length) / sample_rate_hz
    return time, reconstructed
