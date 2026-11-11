# Seismic Denoising — Starter Code (Project Steps 1–3)

This is a runnable foundation for the first three stages of the project:

1. **Input/data preparation:** generate a reproducible synthetic seismic-like trace, or load a one-column CSV containing a real trace.
2. **Noise generation:** add Gaussian noise at a selected SNR (dB).
3. **Time-frequency + SVD preprocessing:** compute an STFT and construct a configurable SVD-based semiclean target.

The code intentionally stops before training a neural network. The SVD energy threshold is an **initial configurable heuristic**; verify it against the paper's exact criterion before treating it as a faithful reproduction.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the synthetic demo

From this folder:

```bash
python run_pipeline.py --mode synthetic --snr-db 0
```

Try the paper's main SNR levels:

```bash
python run_pipeline.py --mode synthetic --snr-db -10
python run_pipeline.py --mode synthetic --snr-db -5
python run_pipeline.py --mode synthetic --snr-db 5
python run_pipeline.py --mode synthetic --snr-db 10
```

Outputs are saved under `outputs/`:
- `trace_data.npz`: clean trace, noisy trace, semiclean trace, sampling rate, and settings
- `pipeline_plot.png`: clean/noisy traces and semiclean trace
- `stft_noisy.png`: magnitude of the noisy trace's STFT

## Use your own trace

Create a CSV with one numeric column (one sample per row), e.g. `my_trace.csv`. If there is a header, specify it with `--skip-rows 1`.

```bash
python run_pipeline.py --mode csv --input data/my_trace.csv --column 0 --skip-rows 0 --sample-rate 100
```

For a CSV with a header:

```bash
python run_pipeline.py --mode csv --input data/my_trace.csv --column 0 --skip-rows 1 --sample-rate 100
```

Arguments:
- `--sample-rate`: samples per second (Hz)
- `--snr-db`: requested input SNR in dB
- `--seed`: random seed for reproducibility
- `--energy-threshold`: cumulative singular-value energy threshold used to retain dominant components (default `0.95`)
- `--nperseg`: STFT window length

## What each stage does

- `src/seismic_pipeline.py`: all reusable functions.
- `run_pipeline.py`: command-line entry point.

The synthetic signal is a simple combination of Ricker wavelets with randomized arrival times. It is useful for testing the pipeline, but it is not a replacement for real seismic datasets.

## Important notes

- The SNR function uses `10 * log10(signal_power / noise_power)`.
- The STFT is computed on the noisy trace. The SVD projection is applied to the complex STFT matrix, and inverse STFT reconstructs a time-domain semiclean trace.
- The current projection uses the left singular vectors of `Y @ Yᴴ` and retains enough leading components to reach the configured cumulative energy threshold. This is a starting point, not a claim that it exactly reproduces every implementation detail in the paper.
- Do not use synthetic results alone to claim real-world denoising performance. The next stage should add dataset handling, baseline comparisons, metrics, and tests.
