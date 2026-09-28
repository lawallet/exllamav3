"""Inspect the actual checkpoint in the pinned runtime without installing anything."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'native-results/block-ir/extension'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=r'D:\tabbyAPI\models\turboderp_DeepSeek-V4-Flash-0731-exl3_3.04bpw')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    from exllamav3 import Config, Model, Cache
    from exllamav3.modules import TransformerBlock
    import exllamav3_ext
    config = Config.from_directory(args.model)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(use_per_device=[85, 40], max_chunk_size=256, max_batch_size=4,
               progressbar=True, verbose=True)
    def tensor(t):
        return dict(shape=list(t.shape), dtype=str(t.dtype), device=str(t.device), contiguous=t.is_contiguous()) if t is not None else None
    data = dict(extension=exllamav3_ext.__file__, plan_available=hasattr(exllamav3_ext, 'BC_BlockPlan'),
                components=list(config.model_classes), placement=[], blocks=[])
    for m in model.modules:
        data['placement'].append(dict(key=m.key, cls=type(m).__name__, device=str(m.device)))
        if isinstance(m, TransformerBlock):
            data['blocks'].append(dict(key=m.key, attention=type(m.attn).__name__, layer_type=m.attn.layer_type,
                router=m.mlp.router_type, sites=[dict(eps=h.rms_eps, norm_eps=n.rms_norm_eps,
                    tensors={k:tensor(t) for k,t in dict(fn=h.fn, fn_h=h.fn_h, base=h.base, scale=h.scale, norm=n.weight).items()})
                    for h,n in ((m.attn_hc,m.attn_norm),(m.mlp_hc,m.mlp_norm))]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))
    model.unload()


if __name__ == '__main__':
    main()
