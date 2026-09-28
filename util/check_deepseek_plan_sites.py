"""Exact existing-IR stages using actual DeepSeek trunk and DSpark weights."""
import json
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BOOT=argparse.ArgumentParser(add_help=False)
BOOT.add_argument('--extension-dir',type=Path,default=ROOT/'native-results/block-ir/extension')
sys.path.insert(1,str(BOOT.parse_known_args()[0].extension_dir.resolve()))
import torch
from exllamav3 import Config, Model
from exllamav3.modules import TransformerBlock, native_plan


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__,parents=[BOOT])
    parser.add_argument('--output',type=Path,default=ROOT/'native-results/deepseek-v4/sites.json')
    args=parser.parse_args()
    cfg=Config.from_directory(r'D:\tabbyAPI\models\turboderp_DeepSeek-V4-Flash-0731-exl3_3.04bpw')
    records=[]
    torch.manual_seed(721)
    for component in ('text','mtp'):
        model=Model.from_config(cfg,component=component)
        for block in model.modules:
            if not isinstance(block,TransformerBlock): continue
            # Exercise both participating validation GPUs, without enumerating
            # or allocating on the unrelated visible device.
            dev=torch.device('cuda',block.layer_idx%2)
            sites=((block.attn_hc,block.attn_norm),(block.mlp_hc,block.mlp_norm))
            for h,n in sites:
                h.load(dev); n.load(dev)
                h.fn_h=h.fn.half()
            for b,q in ((1,1),(2,1),(3,1),(4,1),(1,5),(2,5),(1,6),(2,6),(3,4)):
                native=native_plan.deepseek(sites,b,q)
                x=torch.randn((b,q,4,4096),device=dev)
                z=x.clone()
                a,m=sites
                post,comb,mixed=a[0].mix(x,{})
                expected=a[1].forward(mixed,{},out_dtype=torch.half)
                assert torch.equal(expected,native.begin(z)), (component,block.key,b,q,'begin')
                y=torch.randn((b,q,4096),device=dev)
                a[0].apply_(x,y,post,comb,{})
                post,comb,mixed=m[0].mix(x,{})
                expected=m[1].forward(mixed,{},out_dtype=torch.half)
                assert torch.equal(expected,native.middle(z,y)), (component,block.key,b,q,'middle')
                assert torch.equal(x,z)
                y=torch.randn((b,q,4096),device=dev)
                m[0].apply_(x,y,post,comb,{})
                native.finish(z,y)
                assert torch.equal(x,z), (component,block.key,b,q,'finish')
                prior=z.clone()
                try: native.middle(z,y[:,:,:32])
                except RuntimeError: pass
                else: raise AssertionError('invalid boundary accepted')
                assert torch.equal(z,prior)
                records.append(dict(component=component,key=block.key,batch=b,query=q,device=str(dev),exact=True))
            for h,n in sites: h.unload(); n.unload()
    out=args.output
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(cases=len(records),records=records,
        peak_bytes={str(i):torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())}),indent=2))
    print(f'PASS {len(records)} actual-weight stage specializations')


if __name__=='__main__': main()
