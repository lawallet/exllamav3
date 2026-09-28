"""Inspect loaded Gemma weights and audit the unmodified v1 block-plan contract.

This is a negative eligibility gate, not a stock/plan correctness benchmark.
Uses checkout Python and an existing isolated extension, never installs packages.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(
        r'D:\text-generation-webui\user_data\models\turboderp_gemma-4-31b-it-exl3_6.00bpw'))
    parser.add_argument('--extension-dir', type=Path,
                        default=ROOT / 'native-results/deepseek-v4/extension')
    parser.add_argument('--budgets', type=float, nargs='+', default=[20, 40])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(1, str(args.extension_dir.resolve()))
    # Fail instead of silently using/installing/building another extension.
    import importlib.util
    spec = importlib.util.find_spec('exllamav3_ext')
    if spec is None or Path(spec.origin).parent != args.extension_dir.resolve():
        raise RuntimeError('Expected prebuilt isolated extension in --extension-dir')
    import torch
    from exllamav3 import Config, Model, Cache
    from exllamav3.modules import TransformerBlock, native_plan
    from exllamav3.ext import exllamav3_ext as ext

    def tensor(t):
        return None if t is None else dict(shape=list(t.shape), dtype=str(t.dtype),
            device=str(t.device), contiguous=t.is_contiguous())

    def module(m):
        if m is None:
            return None
        record = dict(cls=type(m).__name__, key=m.key, device=str(m.device))
        for key in ('in_features', 'out_features', 'is_quantized', 'out_dtype',
                    'rms_norm_eps', 'constant_bias', 'constant_scale', 'groups',
                    'span_heads', 'unweighted', 'activation_fn', 'interm_dtype'):
            if hasattr(m, key):
                value = getattr(m, key)
                record[key] = value if isinstance(value, (str, bool, int, float, type(None))) else str(value)
        if hasattr(m, 'weight'):
            record['weight'] = tensor(m.weight)
        return record

    started = time.monotonic()
    cfg = Config.from_directory(str(args.model))
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens=4096, max_batch_size=4)
    data = dict(python=platform.python_version(), torch=torch.__version__,
        cuda=torch.version.cuda, visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
        checkpoint=str(args.model), extension=str(spec.origin),
        extension_sha256=hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest(),
        plan_available=native_plan.available, components=list(cfg.model_classes),
        budgets=args.budgets, blocks=[], rejection_cases=[])
    try:
        model.load(use_per_device=args.budgets, max_chunk_size=256,
                   max_batch_size=4, progressbar=True, verbose=True)
        data['placement'] = [module(m) for m in model.modules]
        data['recurrent_state_cls'] = model.recurrent_state_cls.__name__
        data['caps'] = model.caps
        data['embedding_multiplier'] = model.modules[0].multiplier
        blocks = [m for m in model.modules if type(m) is TransformerBlock]
        with torch.inference_mode():
            for block in blocks:
                a = block.attn
                record = dict(key=block.key, layer=block.layer_idx,
                    device=str(block.device), attention=module(a), mlp=module(block.mlp),
                    attn_hc=module(block.attn_hc), mlp_hc=module(block.mlp_hc),
                    layer_scalar=tensor(block.layer_scalar_t), layer_scalar_f=block.layer_scalar_f,
                    registered=block._native_plan_get is not None,
                    norms={key: module(getattr(block, key)) for key in
                        ('attn_norm', 'attn_post_norm', 'mlp_norm', 'mlp_post_norm')},
                    projections={key: module(getattr(a, key, None)) for key in
                        ('q_proj', 'k_proj', 'v_proj', 'o_proj')},
                    qkv_norms={key: module(getattr(a, key, None)) for key in
                        ('q_norm', 'k_norm', 'v_norm')})
                for key in ('head_dim', 'num_q_heads', 'num_kv_heads', 'sliding_window',
                            'use_k_as_v', 'sm_scale'):
                    record[key] = getattr(a, key, None)
                data['blocks'].append(record)
                assert block.attn_hc is None and block.mlp_hc is None
                assert block._native_plan_get is None
                # Try a standalone norm using real loaded weights. v1 does not
                # admit a norm site, so rejection must precede any execution.
                for b, q in ((1, 1), (2, 1), (3, 1), (4, 1), (1, 5), (2, 6)):
                    for kind in (native_plan.RMS_NORM, native_plan.HC_MIX, native_plan.GR_MIX):
                        try:
                            ext.BC_BlockPlan([[block.attn_norm.weight]], [kind],
                                [block.attn_norm.rms_norm_eps], [0.], [0],
                                [[(native_plan.RMS_NORM, 0)], [], [(native_plan.APPLY, 0)]],
                                [], None, b, q)
                        except RuntimeError as exc:
                            message = str(exc).splitlines()[0]
                            assert 'invalid mix bundle' in message, message
                        else:
                            raise AssertionError('v1 contract changed: revisit Gemma mapping')
                        data['rejection_cases'].append(dict(layer=block.layer_idx,
                            batch=b, query=q, kind=kind, reason=message))
        data['native_invocations'] = 0
        data['outcome'] = 'unsupported_by_v1_site_and_residual_contract'
        participating = {torch.device(m.device).index for m in model.modules
                         if m.device is not None and torch.device(m.device).type == 'cuda'}
        data['memory'] = [dict(device=i, participating=i in participating,
            peak_bytes=torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())]
        assert all(m['peak_bytes'] == 0 for m in data['memory'] if not m['participating'])
    finally:
        model.unload()
    data['elapsed_seconds'] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(f"PASS negative contract gate: {len(blocks)} loaded blocks, "
          f"{len(data['rejection_cases'])} rejected constructions; no native execution", flush=True)


if __name__ == '__main__':
    main()
