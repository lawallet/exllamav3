"""Same-input stock/native parity using every target/draft checkpoint HC site."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--extension-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--graphs',action='store_true')
a=p.parse_args()
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(a.extension_dir.resolve()))
import torch
from exllamav3 import Config,Model
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules import TransformerBlock,native_plan
from exllamav3.modules.hyperconnections import hc_flush

@torch.inference_mode()
def main():
    assert Path(ext.__file__).parent==a.extension_dir.resolve()
    cfg=Config.from_directory('D:/tabbyAPI/models/DeepSeek-V4.1-Flash-1017-Heretic-EXL3-1.68bpw')
    records=[];torch.manual_seed(941)
    for component in ('text','mtp'):
        model=Model.from_config(cfg,component=component)
        for block in model.modules:
            if type(block) is not TransformerBlock:continue
            device=torch.device('cuda',block.layer_idx%2)
            sites=((block.attn_hc,block.attn_norm),(block.mlp_hc,block.mlp_norm))
            for h,n in sites:h.load(device);n.load(device);h.fn_h=h.fn.half()
            for q in (1,2,3,4,5):
                plan=native_plan.deepseek_carry(sites,1,q)
                plan.set_graphs(a.graphs)
                retained=[]
                for pre_present,scale,bdtype in ((False,1.,torch.float),(True,1.,torch.half),
                                                (False,100000.,torch.float),(True,100000.,torch.float)):
                    x=torch.randn(1,q,4,5120,device=device)*scale;z=x.clone()
                    pre=torch.rand(1,q,4,device=device) if pre_present else None
                    params={'hc_pre':pre}
                    h,n=sites[0]
                    post,comb,y,_=h.mix_norm(x,params,n)
                    expected=n.forward(y,params,out_dtype=torch.half)
                    assert torch.equal(plan.begin_carry(z,pre.view(q,4) if pre is not None else None),expected),(component,block.key,q,'begin')
                    assert torch.equal(plan.carry_output,params['hc_pre'].view(q,4))
                    y=torch.randn(1,q,5120,device=device,dtype=bdtype)
                    h.apply_(x,y,post,comb,params);hc_flush(params)
                    h,n=sites[1];post,comb,mixed,_=h.mix_norm(x,params,n)
                    expected=n.forward(mixed,params,out_dtype=torch.half)
                    assert torch.equal(plan.middle(z,y),expected),(component,block.key,q,'middle')
                    assert torch.equal(x,z)
                    assert torch.equal(plan.carry_output,params['hc_pre'].view(q,4))
                    out=torch.randn_like(y)
                    h.apply_(x,out,post,comb,params);hc_flush(params);plan.finish(z,out)
                    assert torch.equal(x,z),(component,block.key,q,'finish')
                    retained.extend((x,z,pre,y,out))
                if a.graphs:assert plan.graph_replays>0
                records.append(dict(component=component,key=block.key,query=q,device=str(device),
                    exact=True,captures=plan.graph_captures,replays=plan.graph_replays))
            for h,n in sites:h.unload();n.unload()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(extension=str(ext.__file__),specializations=len(records),
                                       cases=4*len(records),records=records),indent=2))
    print('PASS',len(records),'actual-weight specializations;',4*len(records),'exact cases')

if __name__=='__main__':main()
