# `optimize_hls4ml.py` run report — approximate-multiplier sweep

Date: 2026-08-19
Model: `scripts/tiny_mlp.keras` (Dense(16→32)→ReLU→Dense(32→16)→ReLU→Dense(16→10), untrained/random weights)
Tool: AMD Vitis HLS 2025.2, installed locally at `/home/codedog/2025.2/Vitis`
FPGA part: `xc7a200tfbg484-2` (Artix-7 200T) — see "Environment issues found" below for why this isn't the hls4ml default part.

## What was run

```
python3 scripts/optimize_hls4ml.py --sweep scripts/sweep.yaml
```

`sweep.yaml` applies the multiplier strategy to **only the last Dense layer** (the
classifier head) and runs full C-synthesis (`csynth`) with Vitis HLS for each variant:

| tag                  | last-layer multiplier                              |
|----------------------|-----------------------------------------------------|
| `baseline`           | `standard` (exact signed multiply, DSP-inferred)    |
| `lpor_k4_last`       | `lpor_k4` (Lower-Part-OR approximate, K=4)          |
| `lsb_zero_k4_last`   | `lsb_zero_k4` (truncate low 4 bits, exact multiply) |
| `comp42_k4_last`     | `comp42_k4` (4:2-compressor approximate, K=4)       |
| `random_lsb_k4_last` | `random_lsb_k4` (random low-bit pattern, K=4)       |

K=8 variants (`lpor_k8`, `comp42_k8`, `random_lsb_k8`) were excluded from this run per request.

Every variant used the same model, the same `Precision: fixed<16,6>` /
`ReuseFactor: 1` / `Strategy: Latency` model config, and the same
`part`/`clock_period` (5 ns target), so the numbers below are directly comparable.

## Environment issues found and fixed

Getting a real synthesis run out of this machine's Vitis install required fixing several
things; recorded here so future runs don't have to rediscover them.

1. **No `vivado_hls`/`vitis_hls`/`vitis` on `PATH`.** The actual 2025.2 install lives at
   `/home/codedog/2025.2/{Vivado,Vitis}`, not on `PATH`. Fix: add
   `/home/codedog/2025.2/Vitis/bin` to `PATH`. hls4ml's `vitis` backend (used here, since
   there is no classic standalone `vivado_hls` in this install) shells out to `vitis-run
   --tcl build_prj.tcl --mode hls`, which is on that path.

2. **`XILINX_VIVADO` / `XILINX_HLS` / `XILINX_VITIS` must be exported explicitly.**
   The installed wrapper scripts auto-detect these from an install layout of
   `<root>/Vivado/<version>`, but this machine's layout is `<root>/<version>/Vivado`
   (version-first). Auto-detection silently fails and HLS then reports "failed to find
   any device libraries". Fix: export all three to their correct absolute paths before
   invoking the script (see "How to reproduce" below).

3. **The hls4ml default part is not installed.** `config_from_keras_model`'s default
   part is `xcvu13p-flga2577-2-e` (Virtex UltraScale+). This machine's Vitis install only
   has device libraries for the Artix-7 family (checked via `data/parts/xilinx/*` —
   `virtexuplus`, `kintexuplus`, `zynquplus` etc. are entirely absent, only
   `artix7`/`virtex4-7`/`spartan*`/`versal` folders have per-part device data, and only
   `artix7` actually has fully populated device libraries). Fix: `pipeline.yaml` now
   pins `part: xc7a200tfbg484-2` explicitly.

4. **Wrong `hls4ml` package was being imported.** `pip show hls4ml` resolves to an
   **editable install pointing at a different clone**, `/home/codedog/Coding/hls4ml`,
   which does not have this repo's custom multiplier strategies
   (`comp42_k4/k8`, `random_lsb_k4/k8`, `lpor_k4/k8`, `lsb_zero_k4`). Running the script
   from `scripts/` (as `python3 scripts/optimize_hls4ml.py`) put `scripts/` rather than
   the repo root on `sys.path`, so `import hls4ml` fell through to that other,
   incompatible clone — synthesis failed with
   `no template named 'random_lsb_k4' in namespace 'nnet::product'`.
   Fix: export `PYTHONPATH=<repo root>` before running, so this repo's `hls4ml/` package
   shadows the editable install.

5. **Real bug, fixed in this repo: `nnet::Latency` vs `nnet::latency`.** After fixing
   (4), synthesis still failed for *every* configuration (not just the custom
   multipliers) with:
   ```
   ERROR: [HLS 207-3633] no member named 'Latency' in namespace 'nnet'; did you mean 'latency'?
   ```
   `hls4ml/backends/vivado/passes/core_templates.py` (and the equivalent convolution/
   recurrent/einsum templates) emit `static const unsigned strategy = nnet::{strategy};`
   directly from the layer's `strategy` attribute, which is conventionally PascalCase
   (`'Latency'`, `'Resource'`, ...) throughout hls4ml's config API. The actual C++ enum
   in `nnet_common.h` is lowercase (`enum strategy { latency, resource, ... }`). This
   means synthesis for **any** model/config using this hls4ml version+Vitis HLS 2025.2
   would fail, independent of the approximate-multiplier work.
   **Fix applied**: `hls4ml/backends/template.py`, `_default_config_params()` now
   lowercases `params['strategy']` before it's substituted into any config template
   (single shared fix point for Dense/Conv/recurrent/einsum templates alike).

6. **Benign, left as-is**: every run prints
   `ERROR: [HLS 200-101] config_array_partition: Unknown option '-maximum_size'`.
   The generated `build_prj.tcl` wraps this call in `catch {...}` (see
   `hls4ml/templates/vivado/build_prj.tcl:164`), so it's non-fatal — Vitis HLS 2025.2
   simply removed/renamed that flag. Did not touch this since it doesn't affect results.

7. **Model file needed regenerating.** `scripts/tiny_mlp.keras` had been saved by a
   newer Keras (Keras 3 `InputLayer(batch_shape=...)`) incompatible with the installed
   `tensorflow==2.14.1` (Keras 2). Regenerated it with the same architecture/seeds using
   the installed TF/Keras so `load_model()` works.

## How to reproduce

```bash
cd /home/codedog/Coding/hls4mlrcoem-comp42-randomlsb/hls4mlrcoem
export PATH=/home/codedog/2025.2/Vitis/bin:$PATH
export XILINX_VIVADO=/home/codedog/2025.2/Vivado
export XILINX_HLS=/home/codedog/2025.2/Vitis
export XILINX_VITIS=/home/codedog/2025.2/Vitis
export PYTHONPATH="$(pwd):$PYTHONPATH"

# single config (no optimization / whatever pipeline.yaml currently specifies):
python3 scripts/optimize_hls4ml.py --config scripts/pipeline.yaml

# full sweep (baseline + each K4 approximate multiplier, last layer only):
python3 scripts/optimize_hls4ml.py --sweep scripts/sweep.yaml

# after a run, summarize all hls_output_* csynth reports into a table:
python3 scripts/summarize_reports.py
```

Each variant writes to `./hls_output_<tag>/`, with the actual Vitis HLS project under
`myproject_prj/solution1/` and the parsed report at
`myproject_prj/solution1/syn/report/myproject_csynth.xml`.

## Results

Resource usage and timing come from Vitis HLS's `csynth` report
(`parse_vivado_report`, `EstimatedClockPeriod` / `AreaEstimates`). Accuracy comes from
`hls_model.compile()` + `hls_model.predict()` on 200 random `N(0,1)` input samples
(fixed seed, identical across variants) compared against the float32 Keras model's
output for the same inputs — i.e. purely the *numerical* error introduced by
fixed-point quantization + the chosen multiplier, not a classification accuracy (this
model has random/untrained weights).

| variant              | LUT    | FF     | DSP | BRAM | Latency (cc) | Clock (est., ns) | MSE (vs float) | MAE (vs float) | Max\|err\| |
|-----------------------|--------|--------|-----|------|:---:|-------|-----------|-----------|-----------|
| `baseline` (standard) | 39,496 |73,855 | 606 | 0    | 23  | 3.647 | 1.2547e-05 | 3.120e-03 | 9.217e-03 |
| `lpor_k4_last`        | 49,719 |77,965 | 534 | 0    | 24  | 3.647 | 1.3676e-05 | 3.281e-03 | 9.217e-03 |
| `lsb_zero_k4_last`    | 39,430 |66,791 | 519 | 0    | 22  | 3.647 | 1.2707e-05 | 2.808e-03 | 1.215e-02 |
| `comp42_k4_last`      | 61,604 |81,348 | 534 | 0    | 25  | 3.647 | 1.2697e-05 | 3.141e-03 | 9.217e-03 |
| `random_lsb_k4_last`  | 48,264 |73,818 | 534 | 0    | 24  | 3.647 | 1.3118e-05 | 3.205e-03 | 9.217e-03 |

Percent change vs `baseline`:

| variant              | ΔLUT   | ΔFF   | ΔDSP   | ΔLatency | ΔMSE   |
|-----------------------|--------|-------|--------|----------|--------|
| `lpor_k4_last`        | +25.9% | +5.6% | −11.9% | +1 cc    | +9.0%  |
| `lsb_zero_k4_last`    | −0.2%  | −9.6% | −14.4% | −1 cc    | +1.3%  |
| `comp42_k4_last`      | +56.0% | +10.1%| −11.9% | +2 cc    | +1.2%  |
| `random_lsb_k4_last`  | +22.2% | −0.1% | −11.9% | +1 cc    | +4.6%  |

Full JSON/markdown table also reproducible any time via `python3 scripts/summarize_reports.py`
as long as the `hls_output_*` directories exist.

## Observations

- **All four approximate multipliers reduce DSP usage on the last layer** (−12 to
  −14%, i.e. going from 606 → ~520-534 DSP slices), at the cost of *more* LUTs, since
  the approximation logic is implemented in fabric fabric rather than a DSP48 hard
  block. That's the expected DSP/LUT trade-off these techniques target.
- **`lsb_zero_k4` is the standout here**: it's the only variant that's *cheaper on
  every resource* than baseline (LUT ≈ same, FF −9.6%, DSP −14.4%) and it's also the
  *fastest* (22 cycles vs baseline's 23), while introducing the smallest MSE increase
  (+1.3%) of the group. This makes sense: it just truncates 4 low bits and does an
  exact (smaller) multiply, so there's very little extra logic — unlike the other
  three, which add explicit approximation/compression logic.
- **`comp42_k4` is the most expensive**: +56% LUTs, +2 cycles latency, for a similar
  DSP saving and MSE increase to `random_lsb_k4`/`lpor_k4`. On this tiny model, the
  4:2-compressor structure's fixed LUT overhead dominates.
- **`random_lsb_k4` roughly matches baseline FF count** (−0.1%) while still shedding
  ~12% DSP, at a real but modest LUT cost (+22%) and MSE cost (+4.6%) — consistent with
  the design's stated goal (the lo·lo quadrant collapsing to constants at synthesis
  saves *some* logic relative to lpor/comp42, though not relative to the DSP-based
  exact baseline).
- Since the approximation is only applied to **one Dense layer** (10 outputs × 16
  inputs = 160 multiplies) out of three, all effects are visible but modest relative to
  total design size; applying a technique network-wide (`target: all` instead of
  `target: last_layer`) would amplify all of the above deltas.
- `EstimatedClockPeriod` is identical (3.647 ns) across all variants — the critical
  path is set elsewhere in the design (the two upstream exact-multiply Dense layers),
  so none of these last-layer-only approximations change the achievable clock in this
  particular model.

## Follow-up run: `lsb_zero_k4` vs `lsb_zero_k8` on a real MNIST classifier

The run above used an untrained toy MLP, so "accuracy" only meant numerical error vs.
a float reference. This follow-up trains a real classifier and compares the two
`lsb_zero` truncation depths (K=4 vs K=8) on real classification accuracy, LUT, and
DSP.

**Model**: `scripts/make_mnist_model.py` trains a small MLP,
`784 → Dense(64) → ReLU → Dense(10) → Activation('softmax')`, on the real MNIST
dataset (8 epochs, Adam) and saves it to `scripts/mnist_mlp.keras`. Float32 test
accuracy: **96.9%** (`model.evaluate`); the accuracy check below uses a fixed
1000-sample draw from the real test set and reports **96.4%** on that subset.

**Config**: `scripts/pipeline_mnist.yaml` (`Strategy: Resource`, `ReuseFactor: 64` --
`Strategy: Latency`/`ReuseFactor: 1` from the tiny-model config is not tractable here,
since the first Dense layer alone has 784×64 = 50,176 multiplies). Same
`xc7a200tfbg484-2` part as above. `accuracy: {dataset: mnist, n_samples: 1000}` makes
`optimize_hls4ml.py` report real top-1 classification accuracy (see
`_report_accuracy`/`_load_mnist_test_samples` in the script) instead of the
random-input MSE check used for the tiny model.

**Sweep**: `scripts/sweep_mnist.yaml` -- `baseline` (`standard`, exact multiply),
`lsb_zero_k4_last`, and `lsb_zero_k8_last`, all applied to the last Dense layer (the
classifier head).

```bash
python3 scripts/optimize_hls4ml.py --sweep scripts/sweep_mnist.yaml
```

Added a `lsb_zero_k8` multiplier (it didn't exist yet -- only `_k4` had a
`mult_strategy` shortcut): `hls4ml/templates/vivado/nnet_utils/nnet_mult.h`
(mirrors `lsb_zero_k4`, backed by the existing generic `lsb_zero_base<x_T, w_T, K>`
with `K=8`) plus its `ChoiceAttribute` registration in
`hls4ml/backends/fpga/fpga_backend.py` (without this the attribute is silently
dropped and synthesis quietly falls back to `standard`).

**Another real bug hit along the way**: using `keras.layers.Softmax()` directly
(rather than `Activation('softmax', ...)`) makes `hls4ml`'s Keras converter set the
layer's `activation` attribute to the *class name* `'Softmax'` (PascalCase) --
see `hls4ml/converters/keras/core.py:65-66`. The Softmax layer's config-struct
template names the struct after that same string (`Softmax_config7`), but
`SoftmaxFunctionTemplate` (`hls4ml/backends/vivado/passes/core_templates.py:356`)
hardcodes the lowercase literal `softmax_config7` when calling it, so the two
never match and synthesis fails with "`softmax_config7` was not declared in this
scope; did you mean `Softmax_config7`?". Routed around it in
`scripts/make_mnist_model.py` by using `Activation('softmax', name='softmax')`
instead, which is the form the rest of the hls4ml ecosystem uses. (Left the
converter itself unpatched since it wasn't needed for this task.)

### Results

| variant              | LUT           | FF             | DSP          | BRAM_18K     | float32 acc | HLS acc | Δacc    | float/HLS agreement | softmax MSE |
|-----------------------|---------------|----------------|--------------|--------------|:-----------:|:-------:|:-------:|:--------------------:|-------------|
| `baseline` (standard) | 55,439 (41%)  | 161,836 (60%)  | 916 (124%)   | 410 (56%)    | 0.9640      | 0.9620  | −0.0020 | 0.9970                | 4.49e-04    |
| `lsb_zero_k4_last`    | 55,439 (41%)  | 161,840 (60%)  | 916 (124%)   | 410 (56%)    | 0.9640      | 0.9620  | −0.0020 | 0.9970                | 4.29e-04    |
| `lsb_zero_k8_last`    | 55,439 (41%)  | 161,840 (60%)  | 916 (124%)   | 410 (56%)    | 0.9640      | 0.9000  | −0.0640 | 0.9260                | 9.63e-03    |

(% = utilization of `xc7a200tfbg484-2`'s available LUT=134,600 / FF=269,200 /
DSP=740 / BRAM_18K=730. DSP is **over 100%** -- see caveats.)

Whole-design totals are identical across **all three** variants (FF differs by ≤4,
noise from the truncation logic's registers, not by strategy). Drilling into the
individual last-Dense-layer module's own csynth report
(`dense_resource_..._config5_s_csynth.xml`) confirms it's not just lost in
rounding at the top level -- that module alone reports **exactly** LUT=2576, DSP=10
for `standard`, `lsb_zero_k4`, *and* `lsb_zero_k8` (FF=3321 for `standard` vs 3325
for both `lsb_zero_*`). At `fixed<16,6>` (16-bit operands), an exact multiply and
truncating the bottom 4 or 8 bits before multiplying all still produce a multiply
that comfortably fits in a single Artix-7 DSP48E1 slice (native up to 18×25) -- so
Vitis HLS allocates the same one DSP per multiply regardless of strategy. The
`lsb_zero` technique's resource story only changes once the truncated width crosses
a DSP-cascade or LUT-vs-DSP inference threshold, which doesn't happen at this
precision/device combination.

**What does change: accuracy, and only for K=8.** `lsb_zero_k4` is statistically
indistinguishable from the exact `standard` baseline (same HLS accuracy 0.9620, same
agreement 0.9970, softmax MSE actually *slightly lower*: 4.29e-04 vs baseline's
4.49e-04) -- at this precision, truncating 4 bits is smaller than the fixed-point
quantization noise the design already has from going `float32 → fixed<16,6>`, so it
costs nothing extra. `lsb_zero_k8` zeroes half the bit-width of every multiply
operand in the output layer, and that layer's outputs feed directly into the argmax
classification decision -- so its error is large enough to actually show: 6.4 points
of accuracy lost (92.6% agreement with the float model) vs. baseline, for **zero
measured hardware savings** over either `standard` or `lsb_zero_k4` on this
layer/precision/device combination. On this evidence, `lsb_zero_k4` is a free
substitution for `standard` here, while `lsb_zero_k8` is strictly worse than both --
if K=8 truncation is going to be used, it would need to go on an earlier, less
classification-sensitive layer to be worth its accuracy cost, since it isn't buying
back any resources on this one.

Reproduce the resource comparison at any time:
```python
from hls4ml.report import parse_vivado_report
parse_vivado_report('hls_output_mnist_lsb_zero_k4_last')['CSynthesisReport']
```

## Caveats

- Absolute LUT/FF/DSP counts and their %-of-device would differ on the intended target
  part (only Artix-7 device libraries are installed on this machine); relative
  comparisons *between variants* are unaffected since they all target the same part.
- This is C-synthesis (`csynth`) only — no `cosim`/RTL simulation/`vsynth` was run, so
  these are Vitis HLS's post-synthesis *estimates*, not post-place-and-route numbers.
- In the first (tiny-model) run, "accuracy" is numerical error vs. the float32 model on
  random inputs (untrained weights), not classification accuracy — see the MNIST
  follow-up for real classification accuracy.
- The MNIST design's estimated DSP usage (916) **exceeds** `xc7a200tfbg484-2`'s 740
  available DSPs (124%). `ReuseFactor: 64` was chosen to make `csynth` scheduling
  tractable, not to fit this specific part — a real implementation would need a higher
  `ReuseFactor` (or `io_stream`, or a part with more DSPs) to actually place-and-route.
  This doesn't affect the K4-vs-K8 comparison since both variants overcommit identically.
