"""Generate a hand-written AXI4-Lite wrapper IP project around an hls4ml
io_parallel/ap_vld core.

`ap_vld` isn't directly usable by an external AXI master (jtag_axi, an AXI
SmartConnect, MicroBlaze, ...) -- it needs a memory-mapped wrapper. This
script automates the pattern documented in firmware/FPGA_DEPLOYMENT.md
("Step 2 -- Wrap it as an AXI4-Lite IP") and first hand-written in
firmware/mnist_axilite_ip/ (for the `baseline_hirf` core) and
firmware/mnist_over100_baseline_axilite_ip/ (for `over100_baseline`):

  1. The original top function (from `optimize_hls4ml.py`'s hls4ml output)
     becomes a sub-function, `<top>_core`, with its top-level
     `#pragma HLS INTERFACE ap_vld` stripped (only valid on an actual
     top-level function).
  2. A new top function, `<wrapper-name>`, declares local
     ARRAY_PARTITION-complete buffers, copies the input/output arrays in
     and out via `#pragma HLS UNROLL` loops, and maps `in_data`, `out_data`,
     and `return` (block control) onto one `s_axilite` bundle.
  3. `nnet_utils/`, `ap_types/`, `weights/`, `parameters.h`, `defines.h` are
     copied unchanged from the hls4ml output.
  4. A `run_hls.tcl` is generated to build + `export_design -format
     ip_catalog` the wrapper via `vitis-run` (Vitis 2025.2 has no
     standalone `vitis_hls` binary).

This only writes the wrapper project -- it does not invoke the HLS
toolchain. Build it yourself:

    cd <out-dir>
    export PATH=/home/codedog/2025.2/Vitis/bin:$PATH
    export XILINX_VIVADO=/home/codedog/2025.2/Vivado
    export XILINX_HLS=/home/codedog/2025.2/Vitis
    export XILINX_VITIS=/home/codedog/2025.2/Vitis
    vitis-run --tcl run_hls.tcl --mode hls

Usage:
    python3 scripts/make_axilite_wrapper.py --hls-dir output/hls_output_mnist_over100_baseline
    python3 scripts/make_axilite_wrapper.py --hls-dir output/hls_output_mnist_baseline_hirf \
        --out-dir firmware/mnist_axilite_ip --wrapper-name mnist_axilite_top
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import yaml

# hls4ml_config.yml embeds a custom `!keras_model` tag (see hls4ml/utils/link.py);
# register a no-op constructor for it so a plain PyYAML safe_load can read the
# rest of the file without needing to import hls4ml (and its tensorflow chain).
yaml.add_multi_constructor('!keras_model', lambda loader, suffix, node: None, Loader=yaml.SafeLoader)


def _read_project_name(hls_dir: Path) -> str:
    cfg_path = hls_dir / 'hls4ml_config.yml'
    if not cfg_path.exists():
        raise SystemExit(f"--top not given and {cfg_path} not found; pass --top explicitly")
    cfg = yaml.safe_load(cfg_path.read_text())
    name = cfg.get('ProjectName')
    if not name:
        raise SystemExit(f"'ProjectName' not found in {cfg_path}; pass --top explicitly")
    return name


def _parse_ports(header_src: str, top: str) -> list[tuple[str, str, str]]:
    """Extract (type, name, size) for each array parameter of the top function."""
    m = re.search(rf'void\s+{re.escape(top)}\s*\((.*?)\)\s*;', header_src, re.DOTALL)
    if not m:
        raise SystemExit(f"Could not find prototype for '{top}' in its header")
    ports = []
    for param in m.group(1).split(','):
        param = param.strip()
        if not param:
            continue
        pm = re.match(r'(\w+)\s+(\w+)\s*\[\s*(\d+)\s*\]', param)
        if not pm:
            raise SystemExit(f"Could not parse parameter '{param}' -- expected 'type name[size]'")
        ports.append((pm.group(1), pm.group(2), pm.group(3)))
    if len(ports) != 2:
        raise SystemExit(
            f"Expected exactly 2 array parameters (input, output) on '{top}', found {len(ports)}. "
            "This script only supports a single-input/single-output ap_vld top."
        )
    return ports


_NOISE_LINES = {
    '// hls-fpga-machine-learning insert IO',
    '// hls-fpga-machine-learning insert load weights',
    '// hls-fpga-machine-learning insert layers',
    '// hls-fpga-machine-learning insert emulator-defines',
    '// ****************************************',
    '// NETWORK INSTANTIATION',
}


def _make_core_cpp(src: str, top: str, core: str) -> str:
    src = re.sub(r'#include <iostream>\n', '', src)
    src = src.replace(f'#include "{top}.h"', f'#include "{core}.h"')
    src = re.sub(rf'\bvoid\s+{re.escape(top)}\s*\(', f'void {core}(', src, count=1)
    # Only the wrapper top gets a top-level interface pragma.
    src = re.sub(r'[ \t]*#pragma HLS INTERFACE ap_vld[^\n]*\n', '', src)
    # csim-only weight loading (guarded by __SYNTHESIS__) has no business in
    # a sub-function -- weights are already embedded via parameters.h.
    src = re.sub(r'#ifndef __SYNTHESIS__.*?#endif\n', '', src, count=1, flags=re.DOTALL)
    lines = [line for line in src.splitlines() if line.strip() not in _NOISE_LINES]
    src = '\n'.join(lines)
    src = re.sub(r'\n{3,}', '\n\n', src)
    return src.strip() + '\n'


def _make_core_h(top: str, core: str, ports: list[tuple[str, str, str]]) -> str:
    guard = core.upper() + '_H_'
    (t1, n1, s1), (t2, n2, s2) = ports
    return f'''#ifndef {guard}
#define {guard}

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

/* Same network as {top} (hls4ml output), minus the top-level
 * `INTERFACE ap_vld` pragma -- this version is called as a sub-function
 * from the AXI4-Lite top, which owns the actual top-level interface. */
void {core}(
    {t1} {n1}[{s1}],
    {t2} {n2}[{s2}]
);

#endif
'''


def _make_wrapper_h(core: str, wrapper: str, ports: list[tuple[str, str, str]]) -> str:
    guard = wrapper.upper() + '_H_'
    (t1, _, s1), (t2, _, s2) = ports
    return f'''#ifndef {guard}
#define {guard}

#include "{core}.h"

/* AXI4-Lite memory-mapped top for {core} (io_parallel/ap_vld). An AXI4-Lite
 * master writes {s1} values into in_data, pulses ap_start (via the standard
 * block-level control register in this same s_axilite bundle), polls
 * ap_done, then reads back {s2} values from out_data. See
 * firmware/FPGA_DEPLOYMENT.md's "Register map" section for the offsets. */
void {wrapper}({t1} in_data[{s1}], {t2} out_data[{s2}]);

#endif
'''


def _make_wrapper_cpp(core: str, wrapper: str, ports: list[tuple[str, str, str]]) -> str:
    (t1, _, s1), (t2, _, s2) = ports
    return f'''#include "{wrapper}.h"

void {wrapper}({t1} in_data[{s1}], {t2} out_data[{s2}]) {{
#pragma HLS INTERFACE s_axilite port=in_data bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=out_data bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS

    {t1} local_in[{s1}];
    {t2} local_out[{s2}];
#pragma HLS ARRAY_PARTITION variable=local_in complete dim=0
#pragma HLS ARRAY_PARTITION variable=local_out complete dim=0

copy_in:
    for (int i = 0; i < {s1}; i++) {{
#pragma HLS UNROLL
        local_in[i] = in_data[i];
    }}

    {core}(local_in, local_out);

copy_out:
    for (int i = 0; i < {s2}; i++) {{
#pragma HLS UNROLL
        out_data[i] = local_out[i];
    }}
}}
'''


def _make_run_hls_tcl(project: str, wrapper: str, core: str, part: str, clock_period: float) -> str:
    return f'''open_project -reset {project}
set_top {wrapper}
add_files src/{wrapper}.cpp -cflags {{-std=c++0x}}
add_files src/{core}.cpp -cflags {{-std=c++0x}}

open_solution -reset solution1
set_part {{{part}}}
create_clock -period {clock_period} -name default

csynth_design
export_design -format ip_catalog -version 1.0

exit
'''


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hls-dir', type=Path, required=True,
                    help='hls4ml output dir (e.g. output/hls_output_mnist_over100_baseline)')
    p.add_argument('--top', help='Top-level function/project name; default: read ProjectName from hls4ml_config.yml')
    p.add_argument('--out-dir', type=Path, help='Wrapper project dir; default: firmware/<tag>_axilite_ip')
    p.add_argument('--wrapper-name', help='AXI4-Lite top function name; default: mnist_<tag>_axilite_top')
    p.add_argument('--part', default='xc7a100tcsg324-1', help='Target part (default: Nexys A7-100T)')
    p.add_argument('--clock-period', type=float, default=5, help='Clock period in ns (default: 5)')
    p.add_argument('--force', action='store_true', help='Overwrite an existing --out-dir')
    args = p.parse_args()

    hls_dir: Path = args.hls_dir
    fw_dir = hls_dir / 'firmware'
    if not fw_dir.is_dir():
        raise SystemExit(f"{fw_dir} not found -- is --hls-dir an hls4ml output dir?")

    top = args.top or _read_project_name(hls_dir)
    top_cpp = fw_dir / f'{top}.cpp'
    top_h = fw_dir / f'{top}.h'
    if not top_cpp.exists() or not top_h.exists():
        raise SystemExit(f"Expected {top_cpp} and {top_h} -- check --top matches the hls4ml ProjectName")

    # Derive a short tag from the hls_dir name for default naming, e.g.
    # "hls_output_mnist_over100_baseline" -> "over100_baseline".
    tag = hls_dir.name
    for prefix in ('hls_output_mnist_', 'hls_output_'):
        if tag.startswith(prefix):
            tag = tag[len(prefix):]
            break

    out_dir = args.out_dir or Path('firmware') / f'mnist_{tag}_axilite_ip'
    wrapper = args.wrapper_name or f'mnist_{tag}_axilite_top'
    core = f'{top}_core'

    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} already exists -- pass --force to overwrite its src/")
        shutil.rmtree(out_dir / 'src', ignore_errors=True)

    src_dir = out_dir / 'src'
    src_dir.mkdir(parents=True, exist_ok=True)

    for name in ('nnet_utils', 'ap_types', 'weights'):
        shutil.copytree(fw_dir / name, src_dir / name, dirs_exist_ok=True)
    for name in ('parameters.h', 'defines.h'):
        shutil.copy2(fw_dir / name, src_dir / name)

    ports = _parse_ports(top_h.read_text(), top)

    (src_dir / f'{core}.h').write_text(_make_core_h(top, core, ports))
    (src_dir / f'{core}.cpp').write_text(_make_core_cpp(top_cpp.read_text(), top, core))
    (src_dir / f'{wrapper}.h').write_text(_make_wrapper_h(core, wrapper, ports))
    (src_dir / f'{wrapper}.cpp').write_text(_make_wrapper_cpp(core, wrapper, ports))

    project = f'{out_dir.name}_prj'
    (out_dir / 'run_hls.tcl').write_text(
        _make_run_hls_tcl(project, wrapper, core, args.part, args.clock_period)
    )

    print(f"Wrapper project written to {out_dir}/")
    print(f"  top (source core):  {top}")
    print(f"  core sub-function:  {core}")
    print(f"  AXI4-Lite wrapper:  {wrapper}")
    print(f"  part:               {args.part}")
    print()
    print("Build it:")
    print(f"  cd {out_dir}")
    print("  export PATH=/home/codedog/2025.2/Vitis/bin:$PATH")
    print("  export XILINX_VIVADO=/home/codedog/2025.2/Vivado")
    print("  export XILINX_HLS=/home/codedog/2025.2/Vitis")
    print("  export XILINX_VITIS=/home/codedog/2025.2/Vitis")
    print("  vitis-run --tcl run_hls.tcl --mode hls")


if __name__ == '__main__':
    main()
