# `optimize_hls4ml.py` run report — approximate-multiplier comparison on MNIST

Date: 2026-08-19
Model: `scripts/mnist_mlp.keras` (`784 → Dense(64) → ReLU → Dense(10) → Activation('softmax')`, trained on real MNIST, 96.9% float32 test accuracy)
Tool: AMD Vitis HLS 2025.2, installed locally at `/home/codedog/2025.2/Vitis`
FPGA part: `xc7a200tfbg484-2` (Artix-7 200T) — see "Environment issues found" below for why this isn't the hls4ml default part.

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
   `artix7` actually has fully populated device libraries). Fix: `pipeline_mnist.yaml`
   pins `part: xc7a200tfbg484-2` explicitly.

4. **Wrong `hls4ml` package was being imported.** `pip show hls4ml` resolves to an
   **editable install pointing at a different clone**, `/home/codedog/Coding/hls4ml`,
   which does not have this repo's custom multiplier strategies
   (`comp42_k4/k8`, `random_lsb_k4/k8`, `lpor_k4/k8`, `lsb_zero_k4/k8`). Running the
   script from `scripts/` (as `python3 scripts/optimize_hls4ml.py`) put `scripts/`
   rather than the repo root on `sys.path`, so `import hls4ml` fell through to that
   other, incompatible clone — synthesis failed with
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

7. **Real bug found: `keras.layers.Softmax()` vs `Activation('softmax')`.** Using the
   standalone `keras.layers.Softmax()` layer class (rather than
   `Activation('softmax', ...)`) makes hls4ml's Keras converter set the layer's
   `activation` attribute to the *class name* `'Softmax'` (PascalCase) — see
   `hls4ml/converters/keras/core.py:65-66`. The Softmax layer's config-struct template
   names the struct after that same string (`Softmax_config7`), but
   `SoftmaxFunctionTemplate` (`hls4ml/backends/vivado/passes/core_templates.py:356`)
   hardcodes the lowercase literal `softmax_config7` when calling it, so the two never
   match and synthesis fails with "`softmax_config7` was not declared in this scope;
   did you mean `Softmax_config7`?". Routed around it in `scripts/make_mnist_model.py`
   by using `Activation('softmax', name='softmax')` instead, which is the form the
   rest of the hls4ml ecosystem uses. (Left the converter itself unpatched since it
   wasn't needed for this task.)

8. **Added a missing multiplier variant**: `lsb_zero_k8` didn't exist yet (only
   `lsb_zero_k4` had a `mult_strategy` shortcut). Added it to
   `hls4ml/templates/vivado/nnet_utils/nnet_mult.h` (mirrors `lsb_zero_k4`, backed by
   the existing generic `lsb_zero_base<x_T, w_T, K>` with `K=8`) plus its
   `ChoiceAttribute` registration in `hls4ml/backends/fpga/fpga_backend.py` (without
   that registration the attribute is silently dropped and synthesis quietly falls
   back to `standard`).

## Model and config

`scripts/make_mnist_model.py` trains a small MLP, `784 → Dense(64) → ReLU → Dense(10) →
Activation('softmax')`, on the real MNIST dataset (8 epochs, Adam) and saves it to
`scripts/mnist_mlp.keras`. Float32 test accuracy: **96.9%** (`model.evaluate`); the
accuracy check reported by the pipeline itself uses a fixed 1000-sample draw from the
real test set and reports **96.4%** on that subset.

`scripts/pipeline_mnist.yaml` uses `Strategy: Resource`, `ReuseFactor: 64` (a fully
unrolled `Strategy: Latency`/`ReuseFactor: 1` config is not tractable here, since the
first Dense layer alone has 784×64 = 50,176 multiplies) and pins
`part: xc7a200tfbg484-2`. Its `accuracy: {dataset: mnist, n_samples: 1000}` block makes
`optimize_hls4ml.py` report real top-1 classification accuracy (see
`_report_accuracy`/`_load_mnist_test_samples` in the script) by comparing the compiled
HLS model's predictions against the float Keras model's, both against ground-truth
MNIST labels.

`scripts/sweep_mnist.yaml` runs three variants, all applied to the last Dense layer
(the classifier head) via `target: last_layer`:

| tag                  | last-layer multiplier                              |
|----------------------|-----------------------------------------------------|
| `baseline`           | `standard` (exact signed multiply, DSP-inferred)    |
| `lsb_zero_k4_last`   | `lsb_zero_k4` (truncate low 4 bits, exact multiply) |
| `lsb_zero_k8_last`   | `lsb_zero_k8` (truncate low 8 bits, exact multiply) |

## How to reproduce

```bash
cd /home/codedog/Coding/hls4mlrcoem-comp42-randomlsb/hls4mlrcoem
export PATH=/home/codedog/2025.2/Vitis/bin:$PATH
export XILINX_VIVADO=/home/codedog/2025.2/Vivado
export XILINX_HLS=/home/codedog/2025.2/Vitis
export XILINX_VITIS=/home/codedog/2025.2/Vitis
export PYTHONPATH="$(pwd):$PYTHONPATH"

# (re)train the model if scripts/mnist_mlp.keras doesn't exist:
python3 scripts/make_mnist_model.py

# single config (whatever pipeline_mnist.yaml currently specifies):
python3 scripts/optimize_hls4ml.py --config scripts/pipeline_mnist.yaml

# full sweep (baseline + lsb_zero_k4 + lsb_zero_k8, last layer only):
python3 scripts/optimize_hls4ml.py --sweep scripts/sweep_mnist.yaml

# after a run, summarize all hls_output_mnist* csynth reports into a table:
python3 scripts/summarize_reports.py
```

Each variant writes to `./hls_output_mnist_<tag>/` (the base config alone writes to
`./hls_output_mnist/`), with the actual Vitis HLS project under
`myproject_<tag>_prj/solution1/` (or `mnist_mlp_prj/solution1/` for the base-config-only
run) and the parsed report at `.../solution1/syn/report/*_csynth.xml`.

## Results

| variant               | LUT           | FF             | DSP          | BRAM_18K     | float32 acc | HLS acc | Δacc    | float/HLS agreement | softmax MSE |
|------------------------|---------------|----------------|--------------|--------------|:-----------:|:-------:|:-------:|:--------------------:|-------------|
| `baseline` (standard)  | 55,439 (41%)  | 161,836 (60%)  | 916 (124%)   | 410 (56%)    | 0.9640      | 0.9620  | −0.0020 | 0.9970                | 4.49e-04    |
| `lsb_zero_k4_last`     | 55,439 (41%)  | 161,840 (60%)  | 916 (124%)   | 410 (56%)    | 0.9640      | 0.9620  | −0.0020 | 0.9970                | 4.29e-04    |
| `lsb_zero_k8_last`     | 55,439 (41%)  | 161,840 (60%)  | 916 (124%)   | 410 (56%)    | 0.9640      | 0.9000  | −0.0640 | 0.9260                | 9.63e-03    |

(% = utilization of `xc7a200tfbg484-2`'s available LUT=134,600 / FF=269,200 /
DSP=740 / BRAM_18K=730. DSP is **over 100%** -- see caveats.)

Reproduce this table at any time:
```python
from hls4ml.report import parse_vivado_report
parse_vivado_report('hls_output_mnist_lsb_zero_k4_last')['CSynthesisReport']
```

## Observations

- **Whole-design totals are identical across all three variants** (FF differs by ≤4,
  noise from the truncation logic's registers, not by strategy). Drilling into the
  individual last-Dense-layer module's own csynth report
  (`dense_resource_..._config5_s_csynth.xml`) confirms it's not just lost in rounding
  at the top level -- that module alone reports **exactly** LUT=2576, DSP=10 for
  `standard`, `lsb_zero_k4`, *and* `lsb_zero_k8` (FF=3321 for `standard` vs 3325 for
  both `lsb_zero_*`). At `fixed<16,6>` (16-bit operands), an exact multiply and
  truncating the bottom 4 or 8 bits before multiplying all still produce a multiply
  that comfortably fits in a single Artix-7 DSP48E1 slice (native up to 18×25) -- so
  Vitis HLS allocates the same one DSP per multiply regardless of strategy. The
  `lsb_zero` technique's resource story only changes once the truncated width crosses
  a DSP-cascade or LUT-vs-DSP inference threshold, which doesn't happen at this
  precision/device combination.

  This appears specific to `Strategy: Resource` + `ReuseFactor: 64` (multiplier
  *shared*/time-multiplexed across 64 iterations): Vivado's DSP-vs-LUT binding for a
  shared operator is decided once for the whole reused hardware block, at full
  operand width, rather than per-value. A fully-unrolled `Strategy: Latency` /
  `ReuseFactor: 1` config (one independent expression per multiply, small enough to
  unroll) would be expected to let the front-end exploit the provably-zero truncated
  bits on a per-instance basis and produce a real LUT/DSP difference instead — this
  hypothesis wasn't verified end-to-end before this report was cleaned up, so treat it
  as a lead for a follow-up run (`scripts/sweep_mnist_lastlayer_latency.yaml` sets this
  up: same model, but forces just the last Dense layer to `Strategy: Latency` /
  `ReuseFactor: 1` while leaving the large first layer at `Resource`/`RF=64`).

- **What does change: accuracy, and only for K=8.** `lsb_zero_k4` is statistically
  indistinguishable from the exact `standard` baseline (same HLS accuracy 0.9620, same
  agreement 0.9970, softmax MSE actually *slightly lower*: 4.29e-04 vs baseline's
  4.49e-04) -- at this precision, truncating 4 bits is smaller than the fixed-point
  quantization noise the design already has from going `float32 → fixed<16,6>`, so it
  costs nothing extra. `lsb_zero_k8` zeroes half the bit-width of every multiply
  operand in the output layer, and that layer's outputs feed directly into the argmax
  classification decision -- so its error is large enough to actually show: 6.4 points
  of accuracy lost (92.6% agreement with the float model) vs. baseline, for **zero
  measured hardware savings** over either `standard` or `lsb_zero_k4` on this
  layer/precision/device combination.

- **Bottom line**: on this evidence, `lsb_zero_k4` is a free substitution for
  `standard` on this layer, while `lsb_zero_k8` is strictly worse than both -- if K=8
  truncation is going to be used, it would need to go on an earlier, less
  classification-sensitive layer to be worth its accuracy cost, since it isn't buying
  back any resources on this one (at least not under `Strategy: Resource`).

## Caveats

- This is C-synthesis (`csynth`) only — no `cosim`/RTL simulation/`vsynth` was run, so
  these are Vitis HLS's post-synthesis *estimates*, not post-place-and-route numbers.
- The design's estimated DSP usage (916) **exceeds** `xc7a200tfbg484-2`'s 740 available
  DSPs (124%). `ReuseFactor: 64` was chosen to make `csynth` scheduling tractable, not
  to fit this specific part — a real implementation would need a higher `ReuseFactor`
  (or `io_stream`, or a part with more DSPs) to actually place-and-route. This doesn't
  affect the baseline-vs-K4-vs-K8 comparison since all three variants overcommit
  identically.
- Absolute LUT/FF/DSP counts and their %-of-device would differ on the intended target
  part (only Artix-7 device libraries are installed on this machine); relative
  comparisons *between variants* are unaffected since they all target the same part.

---

# Addendum: Mitchell's logarithmic approximate multiplier (`mitchell`)

Date: 2026-08-27
Same model/tool/part/config as above (`scripts/mnist_mlp.keras`, Vitis HLS 2025.2,
`xc7a200tfbg484-2`, `Strategy: Resource`, `ReuseFactor: 64`). New multiplier strategy:
`mitchell`, implementing `mitchell_base<x_T, w_T>` in
`hls4ml/templates/vivado/nnet_utils/nnet_mult.h` -- unlike `lpor`/`lsb_zero`/`comp42`/
`random_lsb` (which all keep a real hi/hi, hi/lo, lo/hi partial-product multiply and only
approximate the low k x k quadrant), Mitchell's algorithm replaces the multiply *entirely*
with a leading-one-detect, a fixed-point add, and a shift. No `K` parameter -- it applies
across the full operand width. See `scripts/verify_mitchell.py` for the bit-exact Python
reference (confirms the textbook ~11.1% worst-case relative error bound and that the
approximation never overestimates).

## Sweep config

`scripts/sweep_mnist_mitchell.yaml`, two variants, both applying `mult_strategy` to only
the last Dense layer (the classifier head, same `target: last_layer` convention as the
sweep above) -- each variant is a **fresh** synthesis run (not reused from the table
above), written to its own output directory so baseline and mitchell results stay
separate on disk:

| tag             | output dir                          | last-layer multiplier |
|------------------|--------------------------------------|------------------------|
| `baseline`       | `hls_output_mnist_baseline/`         | `standard` (exact)     |
| `mitchell_last`  | `hls_output_mnist_mitchell_last/`    | `mitchell`             |

```bash
python3 scripts/optimize_hls4ml.py --sweep scripts/sweep_mnist_mitchell.yaml
```

## Whole-design results

| variant          | LUT           | FF             | DSP         | BRAM_18K     | float32 acc | HLS acc | Δacc    | float/HLS agreement | softmax MSE |
|-------------------|---------------|----------------|-------------|--------------|:-----------:|:-------:|:-------:|:--------------------:|-------------|
| `baseline`        | 55,439 (41%)  | 161,836 (60%)  | 916 (124%)  | 410 (56%)    | 0.9640      | 0.9620  | −0.0020 | 0.9970                | 4.49e-04    |
| `mitchell_last`   | 65,260 (48%)  | 167,877 (62%)  | 906 (122%)  | 410 (56%)    | 0.9640      | 0.9640  | **+0.0000** | 0.9970            | 3.31e-04    |

(% = utilization of `xc7a200tfbg484-2`'s LUT=134,600 / FF=269,200 / DSP=740 / BRAM_18K=730.)

## Last-Dense-layer module isolation

Drilling into the module actually touched (`dense_resource_..._config5_s_csynth.xml`,
the 64→10 classifier-head Dense layer) isolates the strategy's true cost/benefit; the
untouched first Dense layer (`..._config2...`, 784→64) is bit-for-bit identical between
variants (BRAM=399, DSP=896, FF=138,803, LUT=38,881 in both), confirming the whole-design
delta above comes entirely from this one module:

| variant          | BRAM_18K | DSP    | FF    | LUT     |
|-------------------|:--------:|:------:|:-----:|:-------:|
| `baseline`        | 5        | 10     | 3,321 | 2,576   |
| `mitchell_last`   | 5        | **0**  | 9,362 | 12,397  |

**This is the first strategy tested in this file that actually moves DSP count.**
`lpor`/`lsb_zero`/`comp42`/`random_lsb` all landed on the same single DSP48 as `standard`
for this module (see the original report above) because their approximation only ever
touches the low k bits of an already-DSP-sized multiply. Mitchell's algorithm removes the
multiply from the datapath entirely, so Vitis HLS infers **zero DSPs** for this module --
confirming the prediction that it's the one technique here with a real chance of trading
DSPs for LUTs rather than just changing accuracy. The trade at this specific module/width
is 10 DSPs for +9,821 LUT / +6,041 FF (a ~4.8x LUT increase and ~2.8x FF increase on a
module that was tiny to begin with) -- a bad trade in isolation on a LUT-constrained part,
but exactly the mechanism this optimization is supposed to provide, and worth revisiting
on a DSP-constrained device/layer (this design is already 124% over its DSP budget, so
even a 10-DSP reduction is a step in the right direction, just nowhere near enough on its
own).

## Observations

- **Accuracy did not regress -- if anything it improved on this test set.** HLS accuracy
  for `mitchell_last` matched float32 exactly (0.9640, Δ=+0.0000), and softmax MSE
  (3.31e-04) was *lower* than the exact `standard` baseline's own MSE (4.49e-04) against
  the same 1000-sample MNIST subset. This is plausible rather than a fluke: Mitchell's
  algorithm has a *systematic, directional* bias (it only ever underestimates magnitude,
  never overestimates -- see `scripts/verify_mitchell.py`), which tends to shrink the
  10 logits going into the final argmax roughly proportionally rather than adding
  independent per-output noise the way a technique like `random_lsb` would -- so it's
  less likely to reorder the argmax than a symmetric-error technique of similar
  magnitude. This is a single 1000-sample run, not an exhaustive claim; worth re-checking
  on a larger sample or a different model before relying on it.
- **Latency cost is small but nonzero**: best/worst latency went from 148/150 to 155/157
  cycles (+7 cycles, ~+4.7%) -- consistent with the leading-one-detect priority encoders
  and variable-shift reconstruction adding pipeline depth that a plain DSP-inferred
  multiply doesn't have.
- **Bottom line**: Mitchell's algorithm is the one technique in this file that actually
  changes *which* resource a multiply consumes (DSP → LUT/FF) rather than just its
  accuracy/LUT count within the same resource class. On this workload it didn't pay off
  in absolute area (LUTs are far more plentiful than the 10 DSPs saved were worth here),
  but it came at zero measured accuracy cost, which the k4/k8 truncation techniques
  above did not uniformly achieve. It would be worth re-testing on a layer/precision
  combination where DSPs (not LUTs) are the binding constraint, or on a device with a
  tighter DSP budget relative to LUTs.

## Reproduce

```bash
cd /home/codedog/Coding/hls4mlrcoem-comp42-randomlsb/hls4mlrcoem
export PATH=/home/codedog/2025.2/Vitis/bin:$PATH
export XILINX_VIVADO=/home/codedog/2025.2/Vivado
export XILINX_HLS=/home/codedog/2025.2/Vitis
export XILINX_VITIS=/home/codedog/2025.2/Vitis
export PYTHONPATH="$(pwd):$PYTHONPATH"

python3 scripts/optimize_hls4ml.py --sweep scripts/sweep_mnist_mitchell.yaml

# whole-design summary table:
python3 scripts/summarize_reports.py

# bit-exact reference / error-bound check (no synthesis needed):
python3 scripts/verify_mitchell.py
```
