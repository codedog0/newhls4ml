"""
Plug-and-play optimization driver for hls4ml + Vivado.

Reads a YAML config file that describes a pipeline of optimization stages
applied sequentially to an hls4ml model. Each stage sets one attribute (or a
small group of attributes) on a chosen set of layer types. This lets you
swap multiplier strategies, reuse factors, precisions, etc. without touching
Python code -- just edit the YAML and re-run.

Usage:
    python optimize_hls4ml.py --config pipeline.yaml
    python optimize_hls4ml.py --config pipeline.yaml --sweep sweep.yaml

The script:
  1. Loads the source model (Keras / ONNX / PyTorch -- autodetected).
  2. Builds a base hls4ml config from the YAML's `model_config`.
  3. Walks the `pipeline` list and applies each stage to the matching layer
     types of the hls4ml model graph.
  4. Writes the HLS C++ project to `output_dir`.
  5. (Optional) Invokes Vivado/Vitis HLS for csim / csynth / cosim / export.

See `pipeline.yaml` (next to this script) for a fully-commented example.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

VALID_MULT_STRATEGIES = {
    'standard', 'lpor_k4', 'lpor_k8', 'lsb_zero_k4', 'lsb_zero_k8', 'comp42_k4', 'comp42_k8',
    'random_lsb_k4', 'random_lsb_k8',
}
VALID_STRATEGIES = {'Latency', 'Resource', 'ResourceUnrolled', 'DistributedArithmetic'}

# Layer types that can have a `mult_strategy` attribute (i.e. layers that
# actually perform a multiply). Used by the `target: last_layer` resolver
# to skip input/flatten/reshape/etc. layers when looking for "the last layer
# of the network".
MULT_LAYER_TYPES = {'Dense', 'Conv1D', 'Conv2D', 'SeparableConv1D', 'SeparableConv2D',
                    'DepthwiseConv1D', 'DepthwiseConv2D', 'SimpleRNN', 'LSTM', 'GRU',
                    'Bidirectional', 'MatMul', 'Dot', 'LayerNormalization'}


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _load_model(cfg: dict[str, Any]):
    """Load a Keras / ONNX / PyTorch model from the path described by `cfg`."""
    path = Path(cfg['path'])
    framework = cfg.get('framework', 'auto').lower()

    if framework == 'auto':
        ext = path.suffix.lower()
        if ext in {'.h5', '.keras', '.json'}:
            framework = 'keras'
        elif ext == '.onnx':
            framework = 'onnx'
        elif ext in {'.pt', '.pth'}:
            framework = 'pytorch'
        else:
            raise ValueError(f"Cannot auto-detect framework for {path}; please set `framework` explicitly.")

    print(f"[load_model] framework={framework} path={path}")
    if framework == 'keras':
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        from tensorflow.keras.models import load_model
        return load_model(str(path)), framework
    if framework == 'onnx':
        import onnx
        return onnx.load(str(path)), framework
    if framework == 'pytorch':
        import torch
        return torch.load(str(path), map_location='cpu'), framework
    raise ValueError(f"Unknown framework: {framework}")


def _convert(model, framework: str, hls_config: dict, output_dir: str, project_name: str,
             backend: str, **kwargs):
    """Dispatch to the right hls4ml converter."""
    import hls4ml
    if framework == 'keras':
        return hls4ml.converters.convert_from_keras_model(
            model, output_dir=output_dir, project_name=project_name,
            backend=backend, hls_config=hls_config, **kwargs,
        )
    if framework == 'onnx':
        return hls4ml.converters.convert_from_onnx_model(
            model, output_dir=output_dir, project_name=project_name,
            backend=backend, hls_config=hls_config, **kwargs,
        )
    if framework == 'pytorch':
        return hls4ml.converters.convert_from_pytorch_model(
            model, output_dir=output_dir, project_name=project_name,
            backend=backend, hls_config=hls_config, **kwargs,
        )
    raise ValueError(f"Unknown framework: {framework}")


def _load_mnist_test_samples(n_samples: int, seed: int = 1234):
    """Load `n_samples` MNIST test images (flattened, normalized to [0,1])
    plus their true labels, using the same preprocessing as make_mnist_model.py."""
    import numpy as np
    from tensorflow.keras.datasets import mnist

    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_test), size=min(n_samples, len(x_test)), replace=False)
    return x_test[idx], y_test[idx]


def _report_accuracy(hls_model, model, framework: str, accuracy_cfg: dict | None = None,
                      n_samples: int = 200, seed: int = 1234):
    """Compare the compiled HLS (fixed-point) model's predictions against the
    original float model's predictions, and print the resulting error.
    Requires `hls_model.compile()` to have succeeded.

    If `accuracy_cfg` names a known labeled `dataset` (currently just
    'mnist'), this reports real top-1 classification accuracy (float model
    vs HLS model, both vs ground truth) on real test samples. Otherwise it
    falls back to a numerical-error check (MSE/MAE/MaxAbsErr) between the
    float and HLS model's raw outputs on random N(0,1) input, which only
    quantifies quantization/approximation error, not classification accuracy.

    Only supports the `keras` framework for now (the source model object is
    used directly to get a float reference prediction).
    """
    if framework != 'keras':
        print(f"  [accuracy] skipped -- not implemented for framework={framework!r}")
        return
    import numpy as np

    accuracy_cfg = accuracy_cfg or {}
    n_samples = accuracy_cfg.get('n_samples', n_samples)
    dataset = accuracy_cfg.get('dataset')

    if dataset == 'mnist':
        x, y_true = _load_mnist_test_samples(n_samples, seed=seed)

        y_ref = model.predict(x, verbose=0)
        y_hls = hls_model.predict(x)

        ref_labels = np.argmax(y_ref, axis=1)
        hls_labels = np.argmax(y_hls, axis=1)

        float_acc = float(np.mean(ref_labels == y_true))
        hls_acc = float(np.mean(hls_labels == y_true))
        agreement = float(np.mean(ref_labels == hls_labels))
        err = y_ref - y_hls
        mse = float(np.mean(err**2))
        print(f"  [accuracy] MNIST top-1, {len(x)} test samples: "
              f"float32_acc={float_acc:.4f} hls_acc={hls_acc:.4f} "
              f"(delta={hls_acc - float_acc:+.4f}) float_vs_hls_agreement={agreement:.4f} "
              f"softmax_output_MSE={mse:.6e}")
        return

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        print("  [accuracy] skipped -- multi-input models not supported")
        return
    n_features = input_shape[1:]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(size=(n_samples, *n_features)).astype('float32')

    y_ref = model.predict(x, verbose=0)
    y_hls = hls_model.predict(x)

    err = y_ref - y_hls
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))
    print(f"  [accuracy] vs float32 keras reference over {n_samples} random samples: "
          f"MSE={mse:.6e} MAE={mae:.6e} MaxAbsErr={max_abs:.6e}")


# -----------------------------------------------------------------------------
# Pipeline stages
# -----------------------------------------------------------------------------

def _layer_mro_names(layer) -> set[str]:
    """Return the set of class names in the layer's MRO (so `Dense` matches `VivadoDense`)."""
    return {cls.__name__ for cls in type(layer).__mro__}


def _resolve_layers(hls_model, stage: dict) -> list:
    """Resolve which layers a stage should apply to.

    Supported targeting modes (mutually exclusive -- pick one per stage):

      target: all                     (default) -- every layer in the graph
      target: last_layer                       -- the last multiply-having layer
                                                 (skips Input/Flatten/Reshape/
                                                 Activation/etc.). If `layer_type`
                                                 is also given, the search is
                                                 restricted to layers of that
                                                 type (e.g. `last Dense layer`).
      target: by_name                          -- the layers whose .name matches
                                                 one of the entries in `names`
      layers: [Dense, Conv1D]                  -- every layer whose class name (or
                                                 any ancestor in its MRO) matches
                                                 one of the entries (legacy mode)
    """
    target = stage.get('target', 'all')
    layers_filter = stage.get('layers')
    layer_type_filter = stage.get('layer_type')
    names = stage.get('names')

    all_layers = list(hls_model.get_layers())

    if target == 'all':
        if layers_filter:
            return [l for l in all_layers
                    if _layer_mro_names(l) & set(layers_filter) or l.name in set(layers_filter)]
        return all_layers

    if target == 'last_layer':
        candidates = all_layers
        # If a layer_type filter is given, restrict to layers whose MRO contains it.
        if layer_type_filter:
            candidates = [l for l in candidates if layer_type_filter in _layer_mro_names(l)]
        else:
            # Otherwise restrict to layers that perform a multiply (have a mult_strategy attr).
            candidates = [l for l in candidates if _layer_mro_names(l) & MULT_LAYER_TYPES]
        if not candidates:
            return []
        last = candidates[-1]
        return [last]

    if target == 'by_name':
        if not names:
            raise ValueError("target: by_name requires a `names: [...]` list in the stage")
        name_set = set(names)
        return [l for l in all_layers if l.name in name_set]

    if layers_filter:
        # Legacy `layers:` mode
        return [l for l in all_layers
                if _layer_mro_names(l) & set(layers_filter) or l.name in set(layers_filter)]

    return all_layers


def stage_set_mult_strategy(hls_model, config: dict, stage: dict):
    """Set the `mult_strategy` attribute on the chosen layer(s).

    Allowed values: standard | lpor_k4 | lpor_k8 | lsb_zero_k4 | comp42_k4 | comp42_k8
                    | random_lsb_k4 | random_lsb_k8

    Targeting is controlled by the `target` field of the stage (see
    `_resolve_layers` for the full taxonomy).
    """
    value = config['mult_strategy']
    if value not in VALID_MULT_STRATEGIES:
        raise ValueError(f"mult_strategy={value!r} not in {sorted(VALID_MULT_STRATEGIES)}")
    matched = 0
    matched_names = []
    for layer in _resolve_layers(hls_model, stage):
        try:
            layer.set_attr('mult_strategy', value)
            matched += 1
            matched_names.append(layer.name)
        except Exception:
            continue
    print(f"  [set_mult_strategy] value={value!r} matched {matched} layer(s): {matched_names}")


def stage_set_random_lsb(hls_model, config: dict, stage: dict):
    """Convenience stage for the random-LSB approximate multiplier.

    This is a thin wrapper around `set_mult_strategy` that picks
    `random_lsb_k4` (or `random_lsb_k8` if `k: 8` is set in the config).

    The reason this is a separate stage is that it has an explicit `enabled`
    on/off switch in the config so the user can toggle the technique from a
    single YAML line without touching the rest of the pipeline.

    Config:
      enabled: true          # set to false to skip this stage entirely
      k: 4                   # 4 or 8 (controls random_lsb_k4 vs random_lsb_k8)
    Targeting:
      Use `target: last_layer` (recommended for the user's use case) or any
      other mode supported by `_resolve_layers`.
    """
    enabled = config.get('enabled', True)
    if not enabled:
        print("  [set_random_lsb] enabled=False -- skipped")
        return
    k = int(config.get('k', 4))
    if k not in (4, 8):
        raise ValueError(f"set_random_lsb: k must be 4 or 8, got {k}")
    value = f'random_lsb_k{k}'
    matched = 0
    matched_names = []
    for layer in _resolve_layers(hls_model, stage):
        try:
            layer.set_attr('mult_strategy', value)
            matched += 1
            matched_names.append(layer.name)
        except Exception:
            continue
    print(f"  [set_random_lsb] value={value!r} (enabled={enabled}) "
          f"matched {matched} layer(s): {matched_names}")


def stage_set_reuse_factor(hls_model, config: dict, stage: dict):
    """Set the reuse factor on the chosen layer(s)."""
    value = int(config['reuse_factor'])
    matched = 0
    matched_names = []
    for layer in _resolve_layers(hls_model, stage):
        try:
            layer.set_attr('reuse_factor', value)
            matched += 1
            matched_names.append(layer.name)
        except Exception:
            continue
    print(f"  [set_reuse_factor] value={value} matched {matched} layer(s): {matched_names}")


def stage_set_strategy(hls_model, config: dict, stage: dict):
    """Set the implementation strategy: Latency | Resource | ResourceUnrolled | DistributedArithmetic."""
    value = config['strategy']
    if value not in VALID_STRATEGIES:
        raise ValueError(f"strategy={value!r} not in {sorted(VALID_STRATEGIES)}")
    matched = 0
    matched_names = []
    for layer in _resolve_layers(hls_model, stage):
        try:
            layer.set_attr('strategy', value)
            matched += 1
            matched_names.append(layer.name)
        except Exception:
            continue
    print(f"  [set_strategy] value={value!r} matched {matched} layer(s): {matched_names}")


def stage_set_precision(hls_model, config: dict, stage: dict):
    """Set precision for io/weight/bias/accum on the chosen layer(s).

    All four keys are optional. Format examples: 'fixed<16,6>', 'ap_fixed<24,8>',
    'ac_fixed<32,16>'.
    """
    matched = 0
    matched_names = []
    for layer in _resolve_layers(hls_model, stage):
        changed = False
        for kind in ('io', 'weight', 'bias', 'accum'):
            key = f'{kind}_precision'
            if key in config:
                try:
                    layer.set_attr(kind, config[key])
                    changed = True
                except Exception:
                    continue
        if changed:
            matched += 1
            matched_names.append(layer.name)
    print(f"  [set_precision] config={config!r} matched {matched} layer(s): {matched_names}")


def stage_set_attr(hls_model, config: dict, stage: dict):
    """Generic fallthrough: set any attribute named by the dict keys to its value."""
    matched = 0
    matched_names = []
    for layer in _resolve_layers(hls_model, stage):
        changed = False
        for key, value in config.items():
            try:
                layer.set_attr(key, value)
                changed = True
            except Exception:
                continue
        if changed:
            matched += 1
            matched_names.append(layer.name)
    print(f"  [set_attr] config={config!r} matched {matched} layer(s): {matched_names}")


STAGE_REGISTRY = {
    'set_mult_strategy': stage_set_mult_strategy,
    'set_random_lsb': stage_set_random_lsb,
    'set_reuse_factor': stage_set_reuse_factor,
    'set_strategy': stage_set_strategy,
    'set_precision': stage_set_precision,
    'set_attr': stage_set_attr,
}


def apply_pipeline(hls_model, pipeline: list[dict]):
    """Walk the pipeline stages and apply each one in order."""
    backend = hls_model.config.backend.name.lower() if hasattr(hls_model.config, 'backend') else 'vivado'
    for i, stage in enumerate(pipeline):
        name = stage.get('name')
        if name not in STAGE_REGISTRY:
            raise ValueError(f"Unknown stage name={name!r}. Available: {sorted(STAGE_REGISTRY)}")
        config = stage.get('config', {})
        # Format the stage header with target info so it's clear in the log which
        # layers a stage is aiming at.
        target = stage.get('target', 'all' if not stage.get('layers') else 'all-filtered')
        layers_filter = stage.get('layers')
        layer_type_filter = stage.get('layer_type')
        names = stage.get('names')
        header_bits = [f"target={target}"]
        if layers_filter: header_bits.append(f"layers={layers_filter}")
        if layer_type_filter: header_bits.append(f"layer_type={layer_type_filter}")
        if names: header_bits.append(f"names={names}")
        print(f"[pipeline {i+1}/{len(pipeline)}] {name} {' '.join(header_bits)} config={config}")
        STAGE_REGISTRY[name](hls_model, config, stage)

    # Re-run the backend's templates flow so the generated parameters.h /
    # config_cpp snippets reflect any attributes we just changed (e.g.
    # `mult_strategy`). Without this, the writer would emit the
    # pre-pipeline C++ that was baked in during convert_from_keras_model.
    apply_templates_flow = f'{backend}:apply_templates'
    try:
        hls_model.apply_flow(apply_templates_flow)
        print(f"  [apply_flow] {apply_templates_flow} re-applied to refresh generated C++")
    except Exception as e:
        print(f"  [apply_flow] WARNING: could not re-apply {apply_templates_flow}: {e}")


# -----------------------------------------------------------------------------
# Synthesis
# -----------------------------------------------------------------------------

def run_synthesis(hls_model, synth_cfg: dict):
    """Optionally invoke Vivado/Vitis HLS for csim/csynth/cosim/export."""
    if not synth_cfg.get('enabled', False):
        print("[synthesis] skipped (enabled=False)")
        return
    try:
        hls_model.build(
            csim=bool(synth_cfg.get('csim', False)),
            synth=bool(synth_cfg.get('csynth', False)),
            cosim=bool(synth_cfg.get('cosim', False)),
            export=bool(synth_cfg.get('export', False)),
        )
        print(f"[synthesis] done. Reports under {hls_model.config.get_output_dir()}")
    except Exception as e:  # pragma: no cover
        print(f"[synthesis] FAILED: {e}")
        traceback.print_exc()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_pipeline(config_path: Path, sweep_overrides: dict | None = None):
    cfg = _load_yaml(config_path)
    if sweep_overrides:
        # Apply top-level overrides from the sweep config
        for k, v in sweep_overrides.items():
            cfg[k] = v

    model_cfg = cfg['model']
    backend = cfg.get('backend', 'vivado')
    output_dir = cfg.get('output_dir', './hls_output')
    project_name = cfg.get('project_name', 'myproject')

    print(f"\n=== Running pipeline: {config_path.name} ===")
    print(f"  backend:    {backend}")
    print(f"  output_dir: {output_dir}")
    print(f"  project:    {project_name}")

    # 1) Load the source model.
    model, framework = _load_model(model_cfg)

    # 2) Build a base hls4ml config (mirrors config_from_keras_model etc.).
    import hls4ml
    model_config = cfg.get('model_config', {})
    if framework == 'keras':
        hls_cfg = hls4ml.utils.config_from_keras_model(
            model,
            backend=backend,
            default_precision=model_config.get('Precision', 'fixed<16,6>'),
            default_reuse_factor=model_config.get('ReuseFactor', 1),
        )
    elif framework == 'onnx':
        hls_cfg = hls4ml.utils.config_from_onnx_model(
            model,
            backend=backend,
            default_precision=model_config.get('Precision', 'fixed<16,6>'),
            default_reuse_factor=model_config.get('ReuseFactor', 1),
        )
    elif framework == 'pytorch':
        hls_cfg = hls4ml.utils.config_from_pytorch_model(
            model,
            backend=backend,
            default_precision=model_config.get('Precision', 'fixed<16,6>'),
            default_reuse_factor=model_config.get('ReuseFactor', 1),
        )
    else:
        raise ValueError(f"Unknown framework: {framework}")

    # Apply model-wide settings (strategy, etc.)
    for key, value in model_config.items():
        hls_cfg['Model'][key] = value

    # 3) Convert model to hls4ml. We pass extra kwargs from cfg (io_type, clock, etc.).
    extra_kwargs = {}
    for k in ('io_type', 'clock_period', 'clock_uncertainty', 'part', 'board'):
        if k in cfg:
            extra_kwargs[k] = cfg[k]

    # For per-layer-type config (Dense, etc.) we apply via the hls_config dict;
    # hls4ml reads these into the corresponding layer attributes during conversion.
    # The pipeline stages below refine those further on the already-built graph.
    hls_model = _convert(model, framework, hls_cfg, output_dir, project_name, backend, **extra_kwargs)

    # 4) Apply the pipeline of optimization stages on the built graph.
    print("\n--- Applying optimization pipeline ---")
    apply_pipeline(hls_model, cfg.get('pipeline', []))

    # 5) Re-write the HLS C++ project so the generated parameters.h reflects
    # the new attribute values (apply_flow above refreshed config_cpp; the
    # writer now needs to be re-invoked to actually emit the files).
    # `compile()` does forward inference; if the backend HLS tool is not
    # installed (no vivado/vitis in PATH) it will fail, but we still want
    # to emit the project files so the user can run synthesis later.
    try:
        hls_model.compile()
        _report_accuracy(hls_model, model, framework, accuracy_cfg=cfg.get('accuracy'))
    except Exception as e:
        print(f"  [compile] WARNING: {e} (continuing -- HLS toolchain may be missing)")
    hls_model.write()
    print(f"\n[done] HLS project generated at: {output_dir}")

    # 6) Optional synthesis step.
    run_synthesis(hls_model, cfg.get('synthesis', {}))

    return hls_model


def run_sweep(sweep_path: Path):
    """Run multiple variants of a pipeline, each with its own overrides.

    Each entry in `sweep:` is a dict whose keys are top-level config overrides
    applied on top of the base pipeline. Useful for benchmarking different
    multiplier strategies or reuse factors side by side.
    """
    sweep_cfg = _load_yaml(sweep_path)
    base_config = Path(sweep_cfg['base_config'])
    variants = sweep_cfg['sweep']
    print(f"\n=== Sweep: {len(variants)} variants on top of {base_config.name} ===")
    summary = []
    for i, v in enumerate(variants):
        tag = v.get('tag', f'variant_{i+1}')
        # Each variant gets its own output dir + project name so we can
        # compare reports across variants.
        overrides = {
            'output_dir': f"{_load_yaml(base_config).get('output_dir', './hls_output')}_{tag}",
            'project_name': f"myproject_{tag}",
        }
        # Merge in user-provided overrides (e.g. mult_strategy).
        for k, val in v.get('overrides', {}).items():
            overrides[k] = val
        # Replace the pipeline if the variant specifies one.
        if 'pipeline' in v:
            overrides['pipeline'] = v['pipeline']
        try:
            run_pipeline(base_config, sweep_overrides=overrides)
            summary.append({'tag': tag, 'status': 'ok', **v.get('overrides', {})})
        except Exception as e:
            print(f"[sweep {tag}] FAILED: {e}")
            traceback.print_exc()
            summary.append({'tag': tag, 'status': f'error: {e}', **v.get('overrides', {})})

    print("\n=== Sweep summary ===")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='Path to pipeline.yaml')
    p.add_argument('--sweep', type=Path, help='Path to sweep.yaml (runs multiple variants)')
    args = p.parse_args()

    if args.sweep:
        run_sweep(args.sweep)
    elif args.config:
        run_pipeline(args.config)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
