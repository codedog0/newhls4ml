# Deploying `baseline_hirf` on real Artix-7 hardware

Date: 2026-09-04

End-to-end path from the trained MNIST MLP (`scripts/mnist_mlp.keras`,
784 → Dense(64) → ReLU → Dense(10) → Softmax, `fixed<16,6>`) to a working
inference demo on a **Digilent Nexys A7-100T** (`xc7a100tcsg324-1`), driven
directly from a host PC over JTAG -- no embedded processor on the FPGA at
all. Confirmed working on real hardware: correct predictions, LEDs lighting
the right digit, repeatable over many runs.

## Architecture

```
 Host PC (Python) ──JTAG──► jtag_axi_0 ──AXI4-Lite──► AXI SmartConnect ──┬──► mnist_axilite_top_0 (MNIST core)
                                                                          └──► axi_gpio_0 (LD0-LD9)
```

No MicroBlaze, no Vitis/BSP/ELF build step. The whole runtime is: a Vivado
bitstream, plus a Python script that pokes AXI4-Lite registers straight over
the JTAG debug bridge.

This is the result of working through -- and ruling out -- two dead ends,
documented in [Lessons learned](#lessons-learned) below:
1. A narrow `io_stream` interface trick to get a MicroBlaze-friendly AXI4-Stream
   port turned out to have a **real bug in the HLS-generated RTL** (confirmed
   via a live ILA capture: `layer10_out_TVALID` never asserted, even after
   6+ minutes of real traffic).
2. MicroBlaze itself, once added to work around that, turned out to be
   unnecessary -- `jtag_axi` gives a host PC a real AXI4-Lite master over
   JTAG with no processor involved.

## Prerequisites

- Vivado + Vitis HLS 2025.2 (`/home/codedog/2025.2/`)
- A Digilent Nexys A7-100T connected via USB (JTAG + power)
- This repo, with `hls4ml/` and `scripts/` present
- Python env with `tensorflow` (pyenv 3.11.9 in this environment: `export PATH="$HOME/.pyenv/shims:$PATH"`)

## Step 1 -- Generate the HLS core (`baseline_hirf`)

The core is a standard hls4ml/Vitis HLS build using `io_parallel` /
`ap_vld` -- the well-trodden, never-shown-broken interface (unlike the
`io_stream` variant, see below). It's produced by the existing sweep
pipeline in this repo, tag `baseline_hirf`:

```bash
cd /home/codedog/Coding/hls4mlrcoem-comp42-randomlsb/hls4mlrcoem
export PATH=/home/codedog/2025.2/Vitis/bin:$PATH
export XILINX_VIVADO=/home/codedog/2025.2/Vivado
export XILINX_HLS=/home/codedog/2025.2/Vitis
export XILINX_VITIS=/home/codedog/2025.2/Vitis
export PYTHONPATH="$(pwd):$PYTHONPATH"

python3 scripts/optimize_hls4ml.py --sweep scripts/sweep_mnist_mitchell_multilayer.yaml
```

Relevant config for the `baseline_hirf` tag (see
`scripts/sweep_mnist_mitchell_multilayer.yaml`): `dense_1` forced to
`ReuseFactor=6272` (needed to keep synthesis tractable at all -- the
model-wide default block_factor does not finish in practical time),
`mult_strategy: standard` on both Dense layers, part `xc7a200tfbg484-2`,
`Precision: fixed<16,6>`, `clock_period: 5` (ns).

Output: `output/hls_output_mnist_baseline_hirf/firmware/` -- the generated
C++ (`myproject_baseline_hirf.cpp/.h`), weights, and `nnet_utils`. This
directory is the input to Step 2. Its top-level interface (from the csynth
report) is:

| Port | Bits | Protocol |
|---|---|---|
| `dense_1_input` | 12544 (784×16, one wide bus) | `ap_vld`, one shared valid bit |
| `layer7_out_0` .. `layer7_out_9` | 16 each | `ap_vld`, individually valid-flagged |
| `ap_clk`/`ap_rst`/`ap_start`/`ap_done`/`ap_idle`/`ap_ready` | -- | block control |

## Step 2 -- Wrap it as an AXI4-Lite IP

`ap_vld` isn't directly usable by an external AXI master -- it needs a
memory-mapped wrapper. Rather than routing through hls4ml's
`VivadoAccelerator` writer (which is board-catalog-gated to Zynq/Alveo
parts only, and doesn't support a bare Artix-7), this is a **hand-written**
plain Vitis HLS wrapper -- `s_axilite` is a standard Vitis HLS feature, not
tied to that writer or any board catalog.

Files (`firmware/mnist_axilite_ip/src/`):
- `myproject_baseline_hirf_core.{h,cpp}` -- a copy of the original
  `myproject_baseline_hirf` function, renamed, with the top-level
  `#pragma HLS INTERFACE ap_vld ...` line removed (that pragma is only
  valid on an actual top-level function; this one is now a sub-function).
  `ARRAY_RESHAPE`/`ARRAY_PARTITION`/`DATAFLOW` pragmas are kept.
- `mnist_axilite_top.{h,cpp}` -- the new top-level function. Declares local
  `ARRAY_PARTITION`-complete buffers, copies `in_data`/`out_data` in and out
  via explicit `#pragma HLS UNROLL` loops (required: an `s_axilite`-mapped
  array is a serial, address-indexed port, so the fully-parallel compute
  core needs its own local complete-partitioned copy fed from it), and
  applies `#pragma HLS INTERFACE s_axilite port=... bundle=CTRL_BUS` to
  `in_data`, `out_data`, and `return` (block control) -- all three folded
  into one AXI4-Lite bus.
- `nnet_utils/`, `ap_types/`, `weights/`, `parameters.h`, `defines.h` --
  copied unchanged from Step 1's output.

Build (`run_hls.tcl`, invoked via `vitis-run` -- Vitis 2025.2 has no
standalone `vitis_hls` binary):

```bash
cd firmware/mnist_axilite_ip
export PATH=/home/codedog/2025.2/Vitis/bin:$PATH
export XILINX_VIVADO=/home/codedog/2025.2/Vivado
export XILINX_HLS=/home/codedog/2025.2/Vitis
export XILINX_VITIS=/home/codedog/2025.2/Vitis
vitis-run --tcl run_hls.tcl --mode hls
```

`run_hls.tcl` targets `xc7a100tcsg324-1` (the Nexys A7-100T's actual part;
Step 1's `xc7a200tfbg484-2` was just a synthesis-capacity check on a
bigger dev part -- the real deployment target is the 100T), 5ns clock,
and ends with `export_design -format ip_catalog -version 1.0`.

Result: a packaged Vivado IP at
`firmware/mnist_axilite_ip/mnist_axilite_ip_prj/solution1/impl/ip/`
(`xilinx.com:hls:mnist_axilite_top:1.0`). Utilization on the 100T: 78% LUT,
42% FF, 11% DSP, 28% BRAM (higher than the raw core -- the AXI4-Lite
address-decode/mux logic for 794 mapped registers adds real overhead, but
it fits).

### Register map

(from the HLS-generated driver header, `xmnist_axilite_top_hw.h`)

| Offset | Purpose |
|---|---|
| `0x000` | `AP_CTRL`: bit0=`ap_start` (R/W), bit1=`ap_done` (R, clear-on-read), bit2=`ap_idle` (R), bit7=`auto_restart` |
| `0x020`-`0x03f` | `out_data[10]`, packed 2×`int16` per 32-bit word (word *n*: bits[15:0]=`out_data[2n]`, bits[31:16]=`out_data[2n+1]`) |
| `0x800`-`0xfff` | `in_data[784]`, same 2-per-word packing |

`ap_fixed<16,6>`: 16 total bits, 6 integer bits → **10 fractional bits**
(scale = 2¹⁰ = 1024). Convert float ↔ fixed with `round(v * 1024)` /
`raw / 1024.0` (sign-extend from bit 15 first when going raw → float).

Protocol: write 392 packed words to `in_data`, write `1` to `AP_CTRL`
(start), poll `AP_CTRL` bit1 until set (also clears it), read 5 packed
words from `out_data`. No FIFO reset or trigger-length dance needed --
much simpler than an AXI4-Stream FIFO protocol would be.

## Step 3 -- Vivado block design

Cells (final, no-MicroBlaze topology):

| Cell | Role |
|---|---|
| `clk_wiz_1` | 100MHz board oscillator → system clock |
| `rst_clk_wiz_1_100M` | `proc_sys_reset`, standard reset network |
| `jtag_axi_0` | `xilinx.com:ip:jtag_axi:1.2` -- AXI4 master driven over JTAG, **no processor** |
| `microblaze_0_axi_periph` | AXI SmartConnect (name is a historical leftover from when MicroBlaze fed it) |
| `mnist_axilite_top_0` | the IP from Step 2 |
| `axi_gpio_0` | 10-bit all-output GPIO → LD0-LD9 |

### IP connections

Pulled directly from the built design's `.hwh` (net-by-net, not
transcribed from memory) so this is exact, not approximate.

**Clocking/reset fan-out** -- every clocked cell uses the same two nets:

| Net | Driven by | Fans out to |
|---|---|---|
| `clk_wiz_1_clk_out1` (100MHz) | `clk_wiz_1/clk_out1` | `jtag_axi_0/aclk`, `mnist_axilite_top_0/ap_clk`, `axi_gpio_0/s_axi_aclk`, `microblaze_0_axi_periph/aclk` |
| `rst_clk_wiz_1_100M_peripheral_aresetn` | `rst_clk_wiz_1_100M/peripheral_aresetn` | `jtag_axi_0/aresetn`, `mnist_axilite_top_0/ap_rst_n`, `axi_gpio_0/s_axi_aresetn`, `microblaze_0_axi_periph/aresetn` |

`clk_wiz_1/clk_in1` ← external port `clk_100mhz` (Pin E3); `resetn` ←
external port `reset_rtl_0` (Pin C12). `rst_clk_wiz_1_100M/ext_reset_in` ←
`reset_rtl_0` directly, `aux_reset_in` ← `clk_wiz_1/locked` (the standard
`proc_sys_reset` pattern: hold reset until the MMCM/PLL actually locks).

**AXI4 fabric** -- `microblaze_0_axi_periph` (AXI SmartConnect) is 1 slave ×
3 master ports, though only 2 masters are actually used:

| SmartConnect port | Connects to | Notes |
|---|---|---|
| `S00_AXI` (slave) | `jtag_axi_0/M_AXI` | the only master driving the bus |
| `M00_AXI` | *(nothing -- dangling)* | leftover from this SmartConnect's earlier 3-peripheral configuration (before MicroBlaze was removed); harmless, just unused |
| `M01_AXI` | `mnist_axilite_top_0/s_axi_CTRL_BUS` | full AXI4-Lite channel set (AR/AW/W/R/B) |
| `M02_AXI` | `axi_gpio_0/S_AXI` | full AXI4-Lite channel set |

`mnist_axilite_top_0/interrupt` is genuinely unconnected (confirmed: a
self-closing `<PORT .../>` with no `<CONNECTIONS>` in the `.hwh` at all) --
by design, since the control script polls `ap_done` instead of using
interrupts.

`axi_gpio_0/gpio_io_o` → top-level external port `gpio_rtl_0_tri_o[9:0]` →
board pins per the XDC below (LD0-LD9).

Address Editor (Vivado does not auto-assign addresses for a master wired in
via Tcl the way Connection Automation would -- run **Assign All** manually):

| Peripheral | Base address |
|---|---|
| `mnist_axilite_top_0/s_axi_CTRL_BUS` | `0x0001_0000` |
| `axi_gpio_0/S_AXI` | `0x0000_0000` |

(If you rebuild from scratch, these may come out differently -- always
re-check the Address Editor and update the Python scripts' `MNIST_BASEADDR`/
`GPIO_BASEADDR` accordingly.)

### Constraints (Nexys A7-100T)

```tcl
# 100 MHz System Clock (Pin E3)
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets design_1_i/clk_wiz_1/inst/clk_in1_design_1_clk_wiz_1_0]
set_property -dict {PACKAGE_PIN E3 IOSTANDARD LVCMOS33} [get_ports clk_100mhz]
create_clock -period 10.000 -name sys_clk_pin -waveform {0.000 5.000} -add [get_ports clk_100mhz]

# CPU Reset Button (Pin C12, Active Low)
set_property -dict {PACKAGE_PIN C12 IOSTANDARD LVCMOS33} [get_ports reset_rtl_0]

# LD0-LD9 (verified from the installed Nexys A7-100T board file, not guessed)
set_property -dict {PACKAGE_PIN H17 IOSTANDARD LVCMOS33} [get_ports {gpio_rtl_0_tri_o[0]}]
set_property -dict {PACKAGE_PIN K15 IOSTANDARD LVCMOS33} [get_ports {gpio_rtl_0_tri_o[1]}]
set_property -dict {PACKAGE_PIN J13 IOSTANDARD LVCMOS33} [get_ports {gpio_rtl_0_tri_o[2]}]
set_property -dict {PACKAGE_PIN N14 IOSTANDARD LVCMOS33} [get_ports {gpio_rtl_0_tri_o[3]}]
set_property -dict {PACKAGE_PIN R18 IOSTANDARD LVCMOS33} [get_ports {gpio_rtl_0_tri_o[4]}]
set_property -dict {PACKAGE_PIN V17 IOSTANDARD LVCMOS33} [get_ports {gpio_rtl_0_tri_o[5]}]
set_property -dict {PACKAGE_PIN U17 IOSTANDARD LVCMOS33} [get_ports {gpio_rtl_0_tri_o[6]}]
set_property -dict {PACKAGE_PIN U16 IOSTANDARD LVCMOS33} [get_ports {gpio_rtl_0_tri_o[7]}]
set_property -dict {PACKAGE_PIN V16 IOSTANDARD LVCMOS33} [get_ports {gpio_rtl_0_tri_o[8]}]
set_property -dict {PACKAGE_PIN T15 IOSTANDARD LVCMOS33} [get_ports {gpio_rtl_0_tri_o[9]}]
```

Note: `gpio_rtl_0_tri_o` (not `gpio_io_o`) is the actual port name Vivado's
"Make External" gives an AXI GPIO output-only interface -- confirmed from a
real `[DRC NSTD-1]` error message, not assumed.

## Step 4 -- Synthesize, implement, generate bitstream

Standard Vivado flow: Generate Block Design Wrapper → Run Synthesis → Run
Implementation → Generate Bitstream. Output:
`<project>.runs/impl_1/design_1_wrapper.bit`, and export hardware
(`File → Export → Export Hardware`, include bitstream) for
`design_1_wrapper.xsa` (only needed if you want the `.hwh` for inspecting
the address map programmatically -- the Python control scripts don't need
it at runtime).

## Step 5 -- Program the board

```bash
export PATH=/home/codedog/2025.2/Vivado/bin:$PATH
# hw_server must be running -- start it if it isn't:
#   nohup hw_server > /tmp/hw_server.log 2>&1 &
xsdb firmware/mnist_microblaze/debug/xsdb_program_bit.tcl
```

(That script is just `connect` + `targets -set -filter {name =~ "xc7a100t"}`
+ `fpga -file <path-to-.bit>` -- update the `.bit` path inside it if your
project lives elsewhere.) This is a **volatile** JTAG configuration: it's
lost on power-cycle and safe to redo any time.

## Step 6 -- Run inference from the host

```bash
cd firmware/mnist_microblaze
export PATH="$HOME/.pyenv/shims:$PATH"
python3 send_mnist_input.py --index 42 --vivado-path /home/codedog/2025.2/Vivado/bin/vivado
```

Loads MNIST test-set image `#42`, converts to `fixed<16,6>`, drives the
core's registers directly over JTAG via Vivado's `hw_axi` debug feature
(`create_hw_axi_txn` / `run_hw_axi` / `get_property DATA` -- **not**
`xsdb mrd/mwr`, since there's no MicroBlaze/MDM debug bridge in this design
to ride on), reads back the 10 logits, argmaxes, and lights the
corresponding LED.

For an interactive demo: `python3 serve_draw.py --vivado-path ...`, then
open `http://localhost:8765` in a browser -- draws a digit on an HTML
canvas and sends it to the FPGA the same way, over HTTP.

## Lessons learned

- **`io_stream` + a hand-reshaped `(784,1)` input is a genuine, hls4ml/Vitis
  HLS-idiomatic way to get a narrow, MicroBlaze/DMA-friendly AXI4-Stream
  interface** (`hls4ml`'s stream packing rule is `beat_width = shape[-1]`,
  confirmed from `fpga_types.py:StreamVariableConverter`) -- but the
  resulting core, on this specific configuration, **never asserted its
  output `TVALID`**, confirmed with a live Vivado ILA capture (6+ minutes of
  real traffic, zero pulses -- ruling out both a wiring mistake, which was
  checked net-by-net against the exported `.hwh`, and a latency issue).
  RTL co-simulation would have caught this earlier, but was blocked by an
  unrelated `xelab`/`gcc` toolchain mismatch in this environment (Xilinx's
  generated DPI glue file needs a C++ compiler; several fix attempts --
  compiler version pinning, `PATH` shims, clean cache -- all failed
  identically, suggesting `xelab` wasn't respecting any of them). The
  `io_parallel`/`ap_vld` core was never shown broken and is what's actually
  deployed.
- **`s_axilite` doesn't require hls4ml's `VivadoAccelerator` writer or its
  board catalog.** It's a plain Vitis HLS pragma; hand-writing a thin
  wrapper works on any part, including a bare Artix-7 with no PS.
- **MicroBlaze was unnecessary.** `jtag_axi` gives a real AXI4 master driven
  from the host over JTAG. Removing MicroBlaze meant losing the `xsdb
  mrd/mwr`-via-MDM control path (that rides on the processor's debug
  bridge) in favor of Vivado's `hw_axi` API -- a different mechanism, not
  just an address change.
- Watch for board USB/JTAG dropping intermittently (`lsusb` losing the FTDI
  device) -- not a tooling bug, just check the physical cable.
- Several Vivado/xsdb Tcl property and command-syntax guesses were wrong on
  the first try (`TRIGGER_COMPARE_VALUE` format, a nonexistent `STATUS`
  property on both `hw_ila` and `hw_axi_txn` objects, `connect_hw_server
  -url` needing a bare `host:port` with no `tcp:` prefix). All were caught
  by testing against real hardware in an isolated script before trusting
  the final deliverable -- worth budgeting for when driving these APIs.

## Reproduce

```bash
cd /home/codedog/Coding/hls4mlrcoem-comp42-randomlsb/hls4mlrcoem

# 1. HLS core
export PATH=/home/codedog/2025.2/Vitis/bin:$PATH
export XILINX_VIVADO=/home/codedog/2025.2/Vivado
export XILINX_HLS=/home/codedog/2025.2/Vitis
export XILINX_VITIS=/home/codedog/2025.2/Vitis
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 scripts/optimize_hls4ml.py --sweep scripts/sweep_mnist_mitchell_multilayer.yaml

# 2. AXI4-Lite wrapper IP
cd firmware/mnist_axilite_ip
vitis-run --tcl run_hls.tcl --mode hls
cd ../..

# 3-4. Vivado block design + bitstream: see Step 3/4 above (GUI or Tcl,
#      not scripted end-to-end here since it depends on your board/project layout)

# 5. Program
export PATH=/home/codedog/2025.2/Vivado/bin:$PATH
xsdb firmware/mnist_microblaze/debug/xsdb_program_bit.tcl

# 6. Run
cd firmware/mnist_microblaze
export PATH="$HOME/.pyenv/shims:$PATH"
python3 send_mnist_input.py --index 42 --vivado-path /home/codedog/2025.2/Vivado/bin/vivado
```
