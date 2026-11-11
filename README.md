# Seismic Denoising: Noise-to-Semiclean U-Net (reproduction + edge deployment)

Implementation of Almasri, Iqbal & Masood, *"Resilient Unsupervised Deep Learning for Data Denoising via Noise-to-Semiclean Training"*, IEEE TGRS vol. 64, 2026 (doi 10.1109/TGRS.2026.3718758).
Written from the paper; the authors' public code (github.com/YaserAlmasri/Noise_Semi_Clean-Denoising) is used only to cross-check details.

## Status

| Phase | Step | Status |
|---|---|---|
| 1 Reproduce | Extract clean STEAD subset | done |
| | STFT + SVD semiclean, checked vs Table II | done |
| | Train U-Net (noisy → semiclean), evaluate vs Table II | done: matches at −5 and 0 dB, 1.7 dB short at +5 dB |
| | Ablation (Table V) | done: all four settings match |
| | Supervised baseline at 0 dB | done: 10.55 vs paper 10.18 |
| | +5 dB gap | closed as an open discrepancy: τ and α ruled out (see below) |
| 2 Compress | Width sweep at 0 dB (64, 32, 16, 8) | done: no loss down to width 8 (64x fewer parameters) |
| | Find the knee (width 4, 2); confirm width 8 at −5 / +5 dB | done: knee between 8 and 4; width 8 holds at all noise levels |
| | int8 quantization of width 8 (`04_quantize.py`) | done at 0 dB: lossless with int16 at 4 layers |
| | Confirm quantized width 8 at −5 / +5 dB | done: lossless at all three noise levels |
| 3 Deploy | Choose board and toolflow (hls4ml / FINN / hand-written HLS) | next |
| 2 Compress | Narrower U-Net, int8 quantization, size/SNR table | |
| 3 Deploy | CPU baseline on PYNQ ARM, then FPGA accelerator (hls4ml / FINN) | |

## How to run

```bash
cd ~/Projects/"Seismic Denoising"
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy h5py pandas torch
python 01_extract_subset.py      # ~40 s: chunk2.hdf5 (15 GB) -> data/stead_z_10k.npy (240 MB)
python 02_check_semiclean.py     # ~35 s: semiclean SNR at -5/0/+5 dB on the test split

# Training (Apple GPU via MPS, or NVIDIA GPU on Colab)
python 03_train.py --snr 0 --limit 300 --epochs 2         # smoke test first, ~2 min
python 03_train.py --snr 0                                # proposed method at 0 dB (target 8.84 dB)
python 03_train.py --snr 0 --mode supervised              # supervised reference (target 10.18 dB)
python 03_train.py --snr 0 --mode noisy --alpha 0         # Table V setting 1
python 03_train.py --snr 0 --mode noisy                   # Table V setting 2
python 03_train.py --snr 0 --alpha 0                      # Table V setting 3
python 03_train.py --snr 0 --width 8                      # Phase 2: 64x smaller model

# int8 quantization (Phase 2)
python 04_quantize.py --ckpt checkpoints/semiclean_snr+0_a0.1_t0.7_w8.pt --snr 0 --width 8
```

Every run appends a row to `results/results.csv` (test SNR next to the paper's figure), saves the loss curve to `results/<tag>_history.json` and the best weights to `checkpoints/<tag>.pt`.

| File | What it does |
|---|---|
| `common.py` | Standardize, add noise (Eq. 1), STFT/ISTFT (Eqs. 2, 7), SVD semiclean (Eqs. 3–5), SNR metric, event-grouped split |
| `01_extract_subset.py` | Picks 10,000 clean Z-component traces from STEAD chunk 2 |
| `02_check_semiclean.py` | Reproduces the self-noise-cancellation column of Table II |
| `model.py` | U-Net (Fig. 1, as in the released code) with a `width` setting; sparsity loss (Eq. 6) |
| `03_train.py` | Trains one model at one noise level, early-stops on validation loss, evaluates test SNR |
| `04_quantize.py` | Simulated int8 post-training quantization: per-channel weights, calibrated per-tensor activations; reports float vs int8 test SNR and size |

## Training decisions

- **No clean data in training or model selection (semiclean mode).** Early stopping watches validation loss against the *semiclean* targets. Clean traces are touched only for the final test SNR. Picking the epoch by clean-signal SNR would quietly make the method supervised.
- **Optimizer:** Adam, learning rate 1e-3 halved every 10 epochs, up to 30 epochs, early stopping after 5 epochs without improvement (all as in the paper).
- **α (L1 weight):** 0.15 / 0.10 / 0.01 at −5 / 0 / +5 dB, the authors' per-noise-level values. Override with `--alpha`.
- **Batch size 64** (authors' code: 128) to fit a MacBook Air's unified memory. `--batch 128` if memory allows.
- **Noise is regenerated per run** with a seed fixed per noise level, so runs at the same SNR see identical noisy data.

## Data decisions

- **Source:** STEAD chunk 2 (200,000 local-earthquake traces, 60 s at 100 Hz, 3 components). STEAD traces are already detrended and band-pass filtered by the dataset authors.
- **Component:** vertical (Z) only. The method denoises single traces.
- **"Relatively clean" reference:** only traces whose recorded Z-component SNR exceeds 40 dB (23,199 qualify). The noise we add (−5 to +5 dB) is then at least 35 dB stronger than the trace's own noise, so the reference really is clean.
- **Sample:** 10,000 random traces (seed 42), covering 8,143 distinct earthquakes. The paper also uses 10,000 STEAD signals; its exact selection isn't stated.
- **Scaling:** each trace standardized to zero mean, unit variance (as in the authors' code).
- **Split by earthquake, not by trace:** 6,840 train / 1,676 val / 1,484 test. The same quake is recorded at several stations; a random per-trace split would put near-duplicate waveforms in both train and test and inflate scores. *This is stricter than the authors' code, which splits per trace.*

## STFT

Hann window, 155 samples, 50% overlap → 78 frequency bins × 78 frames (matches the paper's 78×78). Inverse STFT reconstructs the input to within 1e-6, so the transform itself loses nothing.

## Semiclean check (Table II, self-noise cancellation column, Gaussian noise, τ = 0.7)

Test split, 1,484 traces, SNR averaged per trace:

| Input SNR | Measured input | Semiclean, code rule | Semiclean, paper rule | Paper |
|---|---|---|---|---|
| −5 dB | −4.99 | **−2.52** | −2.72 | −2.52 |
| 0 dB | 0.00 | 3.55 | 3.26 | 3.87 |
| +5 dB | 5.00 | 7.12 | **8.30** | 8.83 |

- −5 dB matches exactly, and 0 dB is within 0.3 dB, so the preprocessing is right. The remaining gap is most plausibly the different trace selection.
- **Code rule vs paper rule:** the released code keeps *k* components, where *k* is the first index at which cumulative energy reaches τ. The paper's text keeps the smallest *j* with E(j) ≥ τ, which is *k* + 1. That one extra component costs 0.2–0.3 dB at low SNR but adds 1.2 dB at +5 dB, where only about 4 components are kept and dropping one removes real signal.
- **τ = 0.9 at +5 dB** (what the authors' code comment suggests) gives 7.24 dB here, worse than the paper rule at τ = 0.7.
- **Default going forward: the paper rule.** It is the documented method, and it guarantees at least τ of the energy is retained. `semiclean_svd(..., rule="code")` switches back.

## Paper vs released code: discrepancies found

1. U-Net: the paper says 2×2 kernels with stride 2 and filters "64 to 512"; the code uses 3×3 stride-1 kernels with filters 64-128-256-256 and a 512 bottleneck. We follow the code.
2. Loss: the paper's L1 term is a sum, and the code's is a mean. The scale of α depends on this; we follow the code.
3. α and τ are tuned per noise level in the code (e.g. α = 0.15 at −5 dB, 0.01 at +5 dB), which assumes the input SNR is known. The paper states this as a limitation.
4. The signal-subspace size rule differs by one component (see above).
5. Table I: Noise2Noise beats the proposed method at −10 dB and +10 dB on synthetic data, despite the text saying "consistently outperforms".

## Results: full-size U-Net (5.82M parameters), test split of 1,484 traces

All runs on an Apple MacBook Air GPU (MPS), about 30–50 minutes each. SNR in dB, averaged per trace.

### Table II reproduction (Gaussian noise)

| Input SNR | Semiclean (ours / paper) | Proposed, unsupervised (ours / paper) | Gain over input | Supervised (ours / paper) |
|---|---|---|---|---|
| −5 dB | −2.72 / −2.52 | **4.50** / 4.59 | +9.5 | – / 7.24 |
| 0 dB | 3.26 / 3.87 | **8.87** / 8.84 | +8.9 | **10.55** / 10.18 |
| +5 dB | 8.29 / 8.83 | **10.57** / 12.30 | +5.6 | – / 13.42 |

### Table V ablation at 0 dB input

| Setting | Target | L1 sparsity | Ours | Paper |
|---|---|---|---|---|
| 1 | noisy | no | 0.00 | 0.24 |
| 2 | noisy | yes (α = 0.1) | 5.38 | 5.46 |
| 3 | semiclean | no | 5.55 | 6.04 |
| 4 (proposed) | semiclean | yes (α = 0.1) | **8.87** | 8.84 |

### What the results show

- **Reproduced at −5 and 0 dB, within 0.1 dB**, despite the stricter split by earthquake.
- **The network beats its own training target.** At 0 dB the semiclean targets are only 3.26 dB, yet the model reaches 8.87 dB. At −5 dB it turns −2.72 dB targets into 4.50 dB.
- **The two ingredients only work together.** Noisy targets with no penalty learn the identity (setting 1: training loss → 0, no denoising). The sparsity term alone (setting 2) or the semiclean target alone (setting 3) each give about +5.5 dB. Together they give +8.9 dB: more than either alone, but less than their sum.
- **Setting 3 still beats its target (5.55 vs 3.26 dB).** The errors in the SVD targets vary from trace to trace, so a network fitting thousands of them averages part of that error out. This is the same effect that makes Noise2Noise work.
- **The unsupervised model gets within 1.7 dB of the supervised one** at 0 dB (8.87 vs 10.55), without ever seeing clean data. The paper's gap is 1.3 dB (8.84 vs 10.18).
- **The first supervised attempt was invalid.** It used α = 0.1, and the L1 term shrank the outputs to 8.72 dB, below the unsupervised model. With plain MSE (α = 0) it reaches 10.55. A sparsity penalty helps when the targets are noisy and only hurts when they are clean. Repeating the invalid run reproduced every loss value exactly, so training is deterministic on a given device.

### The +5 dB gap: what has been ruled out

At +5 dB the model reaches 10.57 dB against the paper's 12.30. Experiments so far (one change at a time, test split, full-size model):

| Run | τ | α | Semiclean target | Model | Device |
|---|---|---|---|---|---|
| Baseline | 0.7 | 0.01 | 8.29 | **10.57** | MPS |
| Looser target | 0.9 | 0.01 | 7.08 | 9.49 | MPS |
| Stronger sparsity | 0.7 | 0.05 | 8.29 | 9.49 | CUDA |
| Stronger sparsity | 0.7 | 0.10 | 8.29 | 8.42 | CUDA |
| Weaker sparsity | 0.7 | 0.003 | 8.29 | 10.58 | CUDA |
| No sparsity | 0.7 | 0 | 8.29 | 10.53 | CUDA |

- **Hypothesis 1, rejected: "τ = 0.7 discards real signal".** A looser target (τ = 0.9) is noisier, and the model got 1.1 dB worse. The model's output tracks the quality of its target, so τ = 0.7 is the better choice even at +5 dB.
- **Hypothesis 2, rejected: "α = 0.01 is too weak".** At 0 dB the sparsity term gave a large boost (ablation), but at +5 dB raising α makes things steadily worse: about 1.1 dB lost per step from 0.01 to 0.05 to 0.10. At high SNR most STFT coefficients are real signal, so an L1 penalty shrinks signal along with noise. This agrees with the paper's statement that α should scale with the noise level.
- **Hypothesis 3, rejected: "α should be even smaller".** Results plateau at about 10.55 dB for every α from 0 to 0.01. α is not the lever at +5 dB.
- **Finding beyond the paper: the sparsity term's value depends on noise level.** At 0 dB it adds 3.3 dB (5.55 without, 8.87 with). At +5 dB it adds nothing (10.53 without, 10.57 with), and any larger α hurts. The paper's ablation is run at 0 dB only, so it doesn't show this.
- **Conclusion: an open discrepancy.** With τ and α ruled out, the most likely cause is the data. Our semiclean targets at +5 dB are 0.5 dB below the paper's with identical settings (8.29 vs 8.83), which points to differences in the traces themselves; the paper doesn't say how its 10,000 traces were selected. Best result: 10.58 dB vs the paper's 12.30.
- Device note: NVIDIA (CUDA) and Apple (MPS) GPUs do floating-point arithmetic slightly differently, so the same run can differ by about 0.1 dB across them. The 1 dB steps above are far larger than that.

## Phase 2: how small can the model be? (0 dB input, α = 0.1, τ = 0.7)

| Width | Filters (enc1 → bottleneck) | Parameters | Compute / trace | int8 weights | Test SNR | Epoch time | Device |
|---|---|---|---|---|---|---|---|
| 64 (paper) | 64 → 512 | 5,821,250 | ~2.0 GMAC | 5.8 MB | 8.87 | ~90 s | MPS |
| 32 | 32 → 256 | 1,456,546 | ~0.5 GMAC | 1.5 MB | 8.90 | 9 s | CUDA |
| 16 | 16 → 128 | 364,754 | ~0.13 GMAC | 0.36 MB | 8.77 | 5 s | CUDA |
| 8 | 8 → 64 | 91,498 | ~0.03 GMAC | 0.09 MB | **8.83** | 3 s | CUDA |
| 4 | 4 → 32 | 23,030 | ~0.008 GMAC | 0.02 MB | 8.45 | 1–2 s | CUDA |
| 2 | 2 → 16 | 5,836 | ~0.002 GMAC | 0.006 MB | 4.15 | 1 s | CUDA |

- **The knee is between width 8 and width 4.** Width 4 loses about 0.4 dB; width 2 collapses to 4.15 dB, barely above its own 3.26 dB training targets, so it has too little capacity to learn the sparsity-driven denoising.
- **No measurable loss down to width 8.** All four results sit within about 0.1 dB of each other and of the paper's 8.84, which is roughly the size of the MPS-vs-CUDA difference. The paper's model is heavily oversized for this task: 64× fewer parameters and about 64× less compute give the same denoising.
- **Width 8 hadn't fully converged.** Its validation loss was still falling at epoch 30, so more epochs might gain a little.
- **Why this matters for the FPGA:** at width 8 the int8 weights (~90 KB) and the largest stored feature map (8 × 80 × 80 = ~50 KB in int8) together fit in a PYNQ-Z2's ~630 KB of on-chip memory, with room left for the rest of the pipeline.
- **Width 8 holds at every noise level:** 4.60 dB at −5 dB input (full model 4.50, paper 4.59) and 10.68 dB at +5 dB (full model 10.57). The +5 dB shortfall against the paper is identical for a model 64× smaller, which is more evidence that it comes from the data, not from model capacity.
- **Chosen model for deployment: width 8** (91,498 parameters).
- **Caveat:** one run per width, so run-to-run variation (from the random seed) isn't measured yet. Differences under ~0.1 dB shouldn't be read as real.

## Phase 2: int8 quantization of the width-8 model (0 dB input)

Simulated post-training quantization (`04_quantize.py`): per-channel symmetric weights; per-tensor symmetric activations at the model input and every convolution output; ranges calibrated on 1,280 noisy *training* inputs only.

| Weights | Activations | Activation range | Test SNR | Change |
|---|---|---|---|---|
| float32 | float32 | – | 8.83 | – |
| int8 | float32 | – | 8.82 | −0.01 |
| int4 | float32 | – | 8.33 | −0.50 |
| int8 | int8 | 99.99th percentile | 7.97 | −0.86 |
| int8 | int16 | 99.99th percentile | 8.04 | −0.79 |
| int8 | int8, int16 at input/out | 99.99th percentile | 7.97 | −0.86 |
| int4 | int4 | 99.99th percentile | 1.28 | −7.55 |
| int8 | int16 | true maximum | 8.82 | −0.01 |
| int8 | int8 | true maximum | 8.51 | −0.32 |
| **int8** | **int8, int16 at input/enc1/dec4/out** | **true maximum** | **8.82** | **−0.01** |

### How the activation loss was diagnosed

1. **Weights are cheap:** int8 weights cost 0.01 dB, and even int4 only 0.5 dB. The loss was in the activations.
2. **Per-layer sensitivity** (one activation point quantized at a time) showed the cost sat in the four full-resolution layers next to the spectrogram: input, enc1, dec4, out. The deep layers cost nothing.
3. **First hypothesis, partly wrong: not enough resolution at the edges.** Giving input and output 16 bits changed nothing, and 16 bits *everywhere* barely helped (−0.79 dB). Rounding error would have almost vanished with 256× finer steps.
4. **Actual cause: clipping.** Ranges were set at the 99.99th percentile, so the largest 0.01% of values were clamped, and more bits can't fix clamping. In a sparse spectrogram those largest values are the strongest signal coefficients, and SNR weights errors by energy. dec4 shows it most clearly: its 99.99th percentile is ±3.2 but its true maximum is ±10.0, a heavy tail.
5. **Fix: use the true maximum.** int16 activations become lossless (−0.01 dB). int8 everywhere then costs 0.32 dB, almost all of it at dec4 (−0.24), whose wide range makes the int8 steps coarse. Giving the four full-resolution layers 16 bits makes the whole model lossless.

### Phase 2 result

| | Paper model (float32) | Deployment model |
|---|---|---|
| Parameters | 5,821,250 | 91,498 |
| Weight memory | 23.3 MB | **90 KB** (~250× smaller) |
| Compute per trace | ~2.0 GMAC | ~0.03 GMAC (~64× less) |
| Test SNR at 0 dB | 8.87 (paper 8.84) | **8.82** |

- **Hardware fit:** int8 × int16 multiplies fit the PYNQ-Z2's DSP48E1 blocks (25 × 18-bit) at the same cost as int8 × int8. The 16-bit layers double only their feature-map memory: 8 × 80 × 80 values, about 100 KB each. Weights plus the stored skip-connection map still fit the ~630 KB of on-chip memory.
- **Caveat:** ranges come from the largest values in the calibration set. Real field data can exceed them, so a deployed version needs some headroom (or saturation monitoring).
- **Holds at every noise level** (same configuration, width-8 checkpoint trained at each level):

| Input SNR | float32 | Quantized (int8, int16 at 4 layers) | Change | Paper (full model) |
|---|---|---|---|---|
| −5 dB | 4.60 | **4.61** | +0.01 | 4.59 |
| 0 dB | 8.83 | **8.82** | −0.01 | 8.84 |
| +5 dB | 10.68 | **10.66** | −0.02 | 12.30 (open discrepancy) |

### Paper targets still open (Table II, Gaussian noise)

| Input SNR | Proposed | Supervised |
|---|---|---|
| +5 dB | 12.30 (best 10.58, open discrepancy) | 13.42 |
| 0 dB | (done) | (done) |
| −5 dB | (done) | 7.24 |
