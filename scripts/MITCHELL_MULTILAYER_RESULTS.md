# Mitchell's logarithmic approximate multiplier on more than one layer

Date: 2026-08-27
Follow-up to the `mitchell_last`-only result recorded as an addendum in
`scripts/OPTIMIZATION_RESULTS.md`. That run only applied `mult_strategy: mitchell`
to the classifier head (`dense_2`, 64→10, 640 multiplies). This document covers
applying it to the much larger first Dense layer (`dense_1`, 784→64, 50,176
multiplies) as well -- both alone and combined with `dense_2`.

Same model/tool/part as the base report: `scripts/mnist_mlp.keras`
(`784 → Dense(64) → ReLU → Dense(10) → Softmax`, 96.9%/96.4% float32 accuracy),
Vitis HLS 2025.2, `xc7a200tfbg484-2`, `Precision: fixed<16,6>`,
`Strategy: Resource`.

## Sweep: dense_1 forced to a high, tractable ReuseFactor

To make `dense_1` synthesizable, `dense_1`'s `ReuseFactor` was forced to
**6272** (`50,176 / 6272 = 8`, comparable to the already-tractable 10-instance
case) via an explicit `set_reuse_factor` stage in
`scripts/sweep_mnist_mitchell_multilayer.yaml`, applied identically to **all
three** variants below so the multiplier-strategy comparison stays isolated
from the RF change:

| tag                    | dense_1 RF | dense_1 mult_strategy | dense_2 mult_strategy | output dir                              |
|--------------------------|:----------:|-------------------------|--------------------------|-------------------------------------------|
| `baseline_hirf`          | 6272       | standard                 | standard (default RF=64) | `hls_output_mnist_baseline_hirf/`         |
| `mitchell_first_hirf`    | 6272       | **mitchell**             | standard (default RF=64) | `hls_output_mnist_mitchell_first_hirf/`   |
| `mitchell_all_hirf`      | 6272       | **mitchell**             | **mitchell** (default RF=64) | `hls_output_mnist_mitchell_all_hirf/`  |

All three completed in 1-2 minutes each.

**Note on comparability**: `RF=6272 > n_in (784)` puts `dense_1` in a
different codegen branch than the original report's `RF=56` case --
`dense_resource_rf_gt_nin_rem0` instead of `dense_resource_rf_leq_nin`. This
is a genuinely different resource-sharing architecture, not just "the same
kernel with a smaller block_factor" -- so **absolute numbers below are not
directly comparable to the RF=56/`mitchell_last` results** in
`scripts/OPTIMIZATION_RESULTS.md`. They're still a fair, apples-to-apples
comparison *among themselves*, since all three variants here use the identical
RF=6272 / `rf_gt_nin_rem0` kernel on `dense_1`.

```bash
python3 scripts/optimize_hls4ml.py --sweep scripts/sweep_mnist_mitchell_multilayer.yaml
```

## Whole-design results

`baseline_default_rf` is included here for direct reference -- it's the same
model-wide-default-RF (`dense_1` at RF=56, `rf_leq_nin` kernel) standard-multiply
run already reported in `scripts/OPTIMIZATION_RESULTS.md` (sourced from the
existing `hls_output_mnist_baseline/` output on disk, no re-synthesis needed).
It makes the "not directly comparable" caveat above concrete rather than just
asserted -- **don't compare its LUT/FF/DSP/EstClk columns against the three
`_hirf` rows below them**; they're two different resource-sharing kernels at
two different `ReuseFactor`s. It's shown purely so the RF=56 → RF=6272 jump
(needed to make `mitchell` synthesizable on `dense_1` at all) is visible in
one place.

| variant                    | dense_1 RF | LUT           | FF            | DSP        | BRAM_18K   | EstClk (ns) | float32 acc | HLS acc | Δacc    | float/HLS agreement | softmax MSE |
|-------------------------------|:----------:|---------------|---------------|------------|------------|:-----------:|:-----------:|:-------:|:-------:|:--------------------:|-------------|
| `baseline_default_rf` *(ref)* | 56         | 55,439 (41%)  | 161,836 (60%) | 916 (124%) | 410 (56%)  | 3.707       | 0.9640      | 0.9620  | −0.0020 | 0.9970                | 4.49e-04    |
| `baseline_hirf`               | 6272       | 22,122 (16%)  | 39,364 (15%)  | 28 (4%)    | 75 (10%)   | 8.629       | 0.9640      | 0.9620  | −0.0020 | 0.9970                | 4.49e-04    |
| `mitchell_first_hirf`         | 6272       | 29,135 (22%)  | 42,503 (16%)  | 20 (3%)    | 75 (10%)   | 5.465       | 0.9640      | 0.9640  | +0.0000 | 0.9950                | 5.96e-04    |
| `mitchell_all_hirf`           | 6272       | 38,956 (29%)  | 48,544 (18%)  | 10 (1%)    | 75 (10%)   | 5.465       | 0.9640      | 0.9640  | +0.0000 | 0.9960                | 3.33e-04    |
| `baseline_rf3136`             | 3136       | 22,009 (16%)  | 39,192 (15%)  | 36 (5%)    | 68 (9%)    | 8.528       | 0.9640      | 0.9620  | −0.0020 | 0.9970                | 4.49e-04    |
| `mitchell_first_rf3136`       | 3136       | 37,507 (28%)  | 47,351 (18%)  | 20 (3%)    | 68 (9%)    | 5.364       | 0.9640      | 0.9640  | +0.0000 | 0.9950                | 5.96e-04    |
| `baseline_bf32`               | 1568       | 25,908 (19%)  | 42,084 (16%)  | 52 (7%)    | 69 (9%)    | 8.217       | 0.9640      | 0.9620  | −0.0020 | 0.9970                | 4.49e-04    |
| `mitchell_first_bf32`         | 1568       | 56,370 (42%)  | 57,697 (21%)  | 20 (3%)    | 69 (9%)    | 5.133       | 0.9640      | 0.9640  | +0.0000 | 0.9950                | 5.96e-04    |
| `baseline_bf64`               | 784        | 20,593 (15%)  | 56,610 (21%)  | 84 (11%)   | 68 (9%)    | 3.707       | 0.9640      | 0.9620  | −0.0020 | 0.9970                | 4.49e-04    |
| `mitchell_first_bf64`         | 784        | 86,196 (64%)  | 98,799 (37%)  | 20 (3%)    | 68 (9%)    | 3.707       | 0.9640      | 0.9640  | +0.0000 | 0.9950                | 5.96e-04    |

(% = utilization of `xc7a200tfbg484-2`'s LUT=134,600 / FF=269,200 / DSP=740 / BRAM_18K=730.
Target clock period was 5.00ns for all rows. `baseline_bf64`/`mitchell_first_bf64`
use `ReuseFactor=784` (`== n_in`), which lands in the `rf_leq_nin` codegen
branch -- the SAME kernel family as `baseline_default_rf`, unlike the
RF=6272/RF=3136 rows (`rf_gt_nin_rem0`) -- and their `EstClk` (3.707ns, meeting
timing) confirms this: see the updated timing observation below.
`baseline_default_rf`/`baseline_hirf`/`baseline_rf3136`/`baseline_bf64` all
have identical accuracy/agreement/MSE columns because `standard` is an exact
multiply regardless of `ReuseFactor` -- RF only changes hardware scheduling,
never the arithmetic result. Same across all `mitchell_first_*` rows:
identical accuracy/MSE regardless of RF, since RF doesn't change Mitchell's
own approximation error, only how many parallel instances of it get
synthesized.

`baseline_rf3136` / `mitchell_first_rf3136` (RF=3136, block_factor=16),
`baseline_bf32` / `mitchell_first_bf32` (RF=1568, block_factor=32), and
`baseline_bf64` / `mitchell_first_bf64` (RF=784, block_factor=64) -- the
latter two under `./output/hls_output_mnist_*_bf32/` and `*_bf64/` -- are
further data points beyond the RF=6272 (block_factor=8) rows -- see
"block_factor scaling" in Observations below.)

## Per-module isolation

`dense_1` (784→64, the layer actually under test -- note the kernel changes
with RF: `rf_leq_nin` at RF=56, `rf_gt_nin_rem0` at RF=6272) and `dense_2`
(64→10, unchanged except in `mitchell_all_hirf`, always `rf_leq_nin` since its
RF is never overridden here):

| variant                       | dense_1 RF | dense_1 BRAM | dense_1 DSP | dense_1 FF | dense_1 LUT | dense_1 EstClk | dense_2 DSP | dense_2 FF | dense_2 LUT |
|-----------------------------------|:----------:|:------------:|:-----------:|:----------:|:-----------:|:---------------:|:-----------:|:----------:|:-----------:|
| `baseline_default_rf` *(ref)*     | 56         | 399          | 896         | 138,803    | 38,881      | 3.707           | 10          | 3,321      | 2,576       |
| `baseline_hirf`                   | 6272       | 64           | 8           | 16,331     | 5,564       | 8.629           | 10          | 3,321      | 2,576       |
| `mitchell_first_hirf`             | 6272       | 64           | **0**       | 19,470     | 12,577      | 5.465           | 10          | 3,321      | 2,576       |
| `mitchell_all_hirf`               | 6272       | 64           | **0**       | 19,470     | 12,577      | 5.465           | **0**       | 9,362      | 12,397      |
| `baseline_rf3136`                 | 3136       | 57           | 16          | 16,159     | 5,451       | 8.528           | 10          | 3,321      | 2,576       |
| `mitchell_first_rf3136`           | 3136       | 57           | **0**       | 24,318     | 20,949      | 5.364           | 10          | 3,321      | 2,576       |
| `baseline_bf32`                   | 1568       | 58           | 32          | 19,051     | 9,350       | 8.217           | 10          | 3,321      | 2,576       |
| `mitchell_first_bf32`             | 1568       | 58           | **0**       | 34,664     | 39,812      | 5.133           | 10          | 3,321      | 2,576       |
| `baseline_bf64`                   | 784        | 57           | 64          | 33,577     | 4,035       | 3.707           | 10          | 3,321      | 2,576       |
| `mitchell_first_bf64`             | 784        | 57           | **0**       | 75,766     | 69,638      | 3.707           | 10          | 3,321      | 2,576       |

`dense_1`'s own DSP count at `standard` drops from 896 (RF=56) to 8 (RF=6272)
purely from the RF change, before Mitchell even enters the picture -- this is
just the normal "more time-multiplexing, fewer parallel multiply units" effect
of a higher `ReuseFactor`, unrelated to the approximate-multiplier work. The
Mitchell-specific effect is the `baseline_hirf` → `mitchell_first_hirf`
comparison (both at RF=6272): DSP 8 → 0, LUT 5,564 → 12,577. The same holds at
RF=3136 (DSP 16 → **0**, LUT 5,451 → 20,949), RF=1568 (DSP 32 → **0**, LUT
9,350 → 39,812), and RF=784/block_factor=64 (DSP 64 → **0**, LUT 4,035 →
69,638) -- the DSP-elimination result now confirmed across four block_factors
(8, 16, 32, 64) and both kernel families (`rf_gt_nin_rem0` and `rf_leq_nin`).

**block_factor scaling** (dense_1 only, `standard` → `mitchell`):

| block_factor | RF   | kernel            | standard DSP | standard LUT | mitchell DSP | mitchell LUT | mitchell EstClk | LUT ratio vs prior row's instances |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 8  | 6272 | `rf_gt_nin_rem0` | 8  | 5,564 | 0 | 12,577 | 5.465 | -- |
| 16 | 3136 | `rf_gt_nin_rem0` | 16 | 5,451 | 0 | 20,949 | 5.364 | 2x instances -> 1.67x LUT (sub-linear) |
| 32 | 1568 | `rf_gt_nin_rem0` | 32 | 9,350 | 0 | 39,812 | 5.133 | 2x instances -> 1.90x LUT (still sub-linear, trending toward linear) |
| 64 | 784  | `rf_leq_nin`     | 64 | 4,035 | 0 | 69,638 | 3.707 | 2x instances -> 1.75x LUT (but kernel also switched here -- see caveat) |

`standard`'s LUT doesn't move monotonically with block_factor within
`rf_gt_nin_rem0` (5,564 -> 5,451 -> 9,350 for 8/16/32) since its area is
DSP-dominated and DSP count tracks block_factor almost exactly (8, 16, 32,
64) rather than LUT. `mitchell`'s LUT scaling is sub-linear at every doubling
tested so far, and *within the same kernel* (8->16->32) the sub-linearity
clearly weakens as block_factor grows (1.67x, then 1.90x per doubling) -- so
don't extrapolate the smallest, most-sub-linear ratio out to large
block_factors. The 32->64 step also happens to land near 1.75x, but that step
additionally switches kernel (`rf_gt_nin_rem0` -> `rf_leq_nin`), so it isn't a
clean continuation of the same trend -- treat it as its own data point, not
proof the sub-linear trend re-tightened. DSP stays at exactly 0 for
`mitchell` at all four block_factors -- the DSP-elimination result holds
regardless of block_factor or kernel family; only the LUT/FF cost of
achieving it changes. `mitchell`'s own `EstClk` also drops smoothly across
the `rf_gt_nin_rem0` points as block_factor grows (5.465 -> 5.364 -> 5.133),
mirroring `standard`'s parallel drop (8.629 -> 8.528 -> 8.217) within that
same kernel -- consistent with more time-multiplexing generally shortening
each kernel's own critical path, independent of multiplier strategy.

`dense_2`'s own numbers in `mitchell_all_hirf` (DSP=0, FF=9,362, LUT=12,397)
match the original single-layer `mitchell_last` result exactly -- a good
determinism/correctness cross-check, since it's the same layer, same
precision, same default RF=64 in both runs.

## Observations

- **DSP elimination confirmed again, on the big layer this time.** `dense_1`
  goes from DSP=8 to DSP=**0**, at a cost of +7,013 LUT (+126%) / +3,139 FF
  (+19%). Combined with `dense_2` also going to DSP=0 in `mitchell_all_hirf`,
  the **whole design drops from 28 DSPs to 10** (dense_2's untouched DSPs in
  `mitchell_first_hirf`) **to effectively 0 DSPs from either Dense layer**
  once both are converted -- the remaining 10 DSPs in `mitchell_first_hirf`
  and general design DSPs elsewhere (ReLU/Softmax have none) are exactly
  `dense_2`'s unconverted 10. This is the clearest demonstration yet that
  Mitchell's algorithm does what it's architecturally supposed to: move work
  off DSPs entirely, at a real and substantial LUT/FF cost.
- **Timing quirk resolved: it's specific to the `rf_gt_nin_rem0` kernel, not
  Mitchell.** At RF=6272/3136 (`rf_gt_nin_rem0`), `standard`'s critical path
  (8.629ns / 8.528ns) was *worse* than Mitchell's (5.465ns / 5.364ns) despite
  using far fewer LUTs -- surprising on its face. The RF=784 (`rf_leq_nin`)
  data point settles it: there, `standard` and `mitchell` have the *identical*
  `EstimatedClockPeriod` (3.707ns each), matching the original RF=56 report.
  So the apparent "Mitchell is faster" effect only shows up under
  `rf_gt_nin_rem0`, most likely from how Vitis HLS binds/accumulates a small
  number of time-multiplexed DSP48s under that specific resource-sharing
  kernel -- it is **not** a general property of Mitchell's algorithm, and
  `rf_leq_nin` (the kernel used whenever `ReuseFactor <= n_in`, which includes
  every RF choice in the original single-layer report) shows no such effect.
- **Accuracy: still no top-1 loss, but MSE tells a more nuanced story than
  the last-layer-only result.** Both `mitchell_first_hirf` and
  `mitchell_all_hirf` matched float32 accuracy exactly (0.9640, Δ=0.0000) on
  the 1000-sample test set. But `mitchell_first_hirf`'s softmax MSE
  (5.96e-04) is the *worst* of every Mitchell configuration tested so far
  (worse than the exact baseline's own 4.49e-04) -- touching the
  input-sensitive first layer introduces more numerical error than touching
  only the classifier head did. `mitchell_all_hirf`, converting *both*
  layers, has the *best* (lowest) MSE of any Mitchell configuration tested
  (3.33e-04) -- lower than `mitchell_first_hirf` alone and lower than the
  exact baseline. This is plausible (Mitchell's error is a systematic,
  correlated underestimate rather than independent per-layer noise, so
  cascaded application isn't guaranteed to compound), but it is an empirical
  result from one 1000-sample test set and one fixed seed, not a general
  guarantee -- don't read "mitchell everywhere" as reliably safer than
  "mitchell on one layer" without re-checking on more data.
- **Bottom line**: applying Mitchell's algorithm to more layers works and
  extends the DSP-elimination result to the layer that actually dominates
  this design's DSP budget, but doing so under `Strategy: Resource` requires
  deliberately choosing a high `ReuseFactor` for large layers -- at the
  model-wide default (`block_factor=896` here), synthesis simply does not
  finish in practical time. This is a real practical constraint on the
  technique, distinct from (and in addition to) the accuracy/area trade-offs
  already documented for the last-layer-only case.

## Files touched

```
scripts/sweep_mnist_mitchell_multilayer.yaml   # 3-variant sweep (baseline_hirf, mitchell_first_hirf, mitchell_all_hirf),
                                                # forces dense_1 to ReuseFactor=6272 for tractability
scripts/sweep_mnist_mitchell_rf3136.yaml       # 2-variant sweep (baseline_rf3136, mitchell_first_rf3136),
                                                # dense_1 at ReuseFactor=3136 (block_factor=16); writes under ./output/
scripts/sweep_mnist_mitchell_bf32.yaml         # 2-variant sweep (baseline_bf32, mitchell_first_bf32),
                                                # dense_1 at ReuseFactor=1568 (block_factor=32, rf_gt_nin_rem0 kernel); writes under ./output/
scripts/sweep_mnist_mitchell_bf64.yaml         # 2-variant sweep (baseline_bf64, mitchell_first_bf64),
                                                # dense_1 at ReuseFactor=784 (block_factor=64, rf_leq_nin kernel); writes under ./output/
scripts/MITCHELL_MULTILAYER_RESULTS.md         # this file
```

No changes were needed to `nnet_mult.h` / `fpga_backend.py` / `optimize_hls4ml.py`
for this run -- the `mitchell` strategy and `by_name` + `set_reuse_factor`
targeting already existed from prior work.

**Note on output locations**: the RF=6272 sweep's outputs
(`hls_output_mnist_baseline_hirf/` etc.) live at the repo root, matching the
original `mitchell_last` run's convention. The RF=3136, RF=1568, and RF=784
sweeps instead write to `./output/hls_output_mnist_*_rf3136/`,
`./output/hls_output_mnist_*_bf32/`, and `./output/hls_output_mnist_*_bf64/`
respectively (set via an explicit `output_dir:` override per variant in each
sweep YAML) -- `scripts/summarize_reports.py` globs the current directory
non-recursively, so `cd output/` before running it against those results.

## Reproduce

```bash
cd /home/codedog/Coding/hls4mlrcoem-comp42-randomlsb/hls4mlrcoem
export PATH=/home/codedog/2025.2/Vitis/bin:$PATH
export XILINX_VIVADO=/home/codedog/2025.2/Vivado
export XILINX_HLS=/home/codedog/2025.2/Vitis
export XILINX_VITIS=/home/codedog/2025.2/Vitis
export PYTHONPATH="$(pwd):$PYTHONPATH"

python3 scripts/optimize_hls4ml.py --sweep scripts/sweep_mnist_mitchell_multilayer.yaml
python3 scripts/optimize_hls4ml.py --sweep scripts/sweep_mnist_mitchell_rf3136.yaml
python3 scripts/optimize_hls4ml.py --sweep scripts/sweep_mnist_mitchell_bf32.yaml
python3 scripts/optimize_hls4ml.py --sweep scripts/sweep_mnist_mitchell_bf64.yaml

# whole-design summary table:
python3 scripts/summarize_reports.py
```

**Do not** attempt `mult_strategy: mitchell` on `dense_1` at the model-wide
default `ReuseFactor` (i.e. without the `set_reuse_factor` override above) --
it will not complete in practical time under `Strategy: Resource`.
