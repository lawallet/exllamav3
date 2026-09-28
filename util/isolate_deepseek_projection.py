"""Actual-weight unequal-width MGEMM reproducer, without attention or cache."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extension-dir',type=Path,default=ROOT/'native-results/block-ir/extension')
    p.add_argument('--devices',type=int,nargs='+',default=[0,1])
    p.add_argument('--reference',type=Path,help='Check invariance and byte-exact unaffected outputs against a pre-fix result')
    args=p.parse_args()
    sys.path.insert(1,str(args.extension_dir.resolve()))
    import torch
    from exllamav3 import Config,Model
    from exllamav3.ext import exllamav3_ext as ext
    from util.validate_deepseek_plan import digest
    args.output.mkdir(parents=True,exist_ok=False)
    samples=torch.load(args.samples,weights_only=True)
    xcpu=next(v for k,v in samples.items() if ':attention-input:' in k)
    cfg=Config.from_directory(r'D:\tabbyAPI\models\turboderp_DeepSeek-V4-Flash-0731-exl3_3.04bpw')
    model=Model.from_config(cfg)
    a=next(b.attn for b in model.modules if getattr(b,'layer_idx',None)==36)
    lins=[a.q_a,a.wkv,a.compressor.wkv,a.compressor.wgate,a.indexer.wkv,a.indexer.wgate]
    rows=[]
    tensors={}
    with torch.inference_mode():
        for device in args.devices:
            dev=torch.device('cuda',device)
            for lin in lins: lin.load(dev)
            inner=[l.inner for l in lins]
            ns=[l.out_features for l in lins]
            assert len({(i.K,i.mcg,i.mul1) for i in inner})==1
            x=xcpu.to(dev)
            def ptr(attr): return torch.tensor([getattr(i,attr).data_ptr() for i in inner],dtype=torch.long,device=dev)
            b,su,sv=ptr('trellis'),ptr('suh'),ptr('svh')
            widths=torch.tensor(ns,dtype=torch.int32,device=dev)
            indices=torch.arange(6,dtype=torch.long,device=dev).unsqueeze(0)
            had=torch.empty((6,1,4096),dtype=torch.half,device=dev)
            outs=[torch.empty((1,n),dtype=torch.half,device=dev) for n in ns]
            def run(shape):
                cp=torch.tensor([o.data_ptr() for o in outs],dtype=torch.long,device=dev)
                return ext.exl3_mgemm(x,b,outs[0].view(1,1,-1),su,had,sv,indices,None,
                    inner[0].K,shape,inner[0].mcg,inner[0].mul1,-1,-1,0,1,widths,cp)
            # Identical input/weights with deliberately different prior output
            # contents, fresh/reused scratch and default/alternate streams.
            for shape in (-1,1,2,3,4):
                for alternate in (False,True):
                    stream=torch.cuda.Stream(device=dev) if alternate else torch.cuda.current_stream(dev)
                    stream.wait_stream(torch.cuda.current_stream(dev))
                    for fresh in (False,True):
                        for repeat in range(6):
                            fill=repeat%3
                            with torch.cuda.stream(stream):
                                if fresh:
                                    outs=[torch.empty((1,n),dtype=torch.half,device=dev) for n in ns]
                                    had=torch.empty((6,1,4096),dtype=torch.half,device=dev)
                                for o in outs: o.fill_(fill)
                                had.zero_()
                                stream.synchronize()
                                selected=run(shape)
                                stream.synchronize()
                                vals=[t.cpu().clone() for t in outs]
                            tag=f'd{device}-s{shape}-a{int(alternate)}-f{int(fresh)}-r{repeat}'
                            tensors[tag]=vals[4]
                            rows.append(dict(tag=tag,device=device,physical_name=torch.cuda.get_device_name(dev),
                                requested=shape,selected=selected,alternate=alternate,fresh=fresh,fill=fill,
                                hashes=[digest(v) for v in vals],indexer_zero=bool(torch.count_nonzero(vals[4])==0)))
            for lin in lins: lin.unload()
    (args.output/'results.json').write_text(json.dumps(dict(extension=ext.__file__,widths=ns,records=rows),indent=2))
    torch.save(tensors,args.output/'indexer-projections.pt')
    if args.reference:
        old=json.loads(args.reference.read_text())['records']
        previous={r['tag']:r for r in old}
        for d in args.devices:
            for shape in (-1,1,2,3,4):
                group=[r for r in rows if r['device']==d and r['requested']==shape]
                assert len({tuple(r['hashes']) for r in group})==1,(d,shape,'prior-output dependency')
        for row in rows:
            if row['requested']<1: continue  # Auto-tuning may choose a different valid shape after the fix.
            n=4 if row['requested']==4 else 6
            assert row['hashes'][:n]==previous[row['tag']]['hashes'][:n],row['tag']
        (args.output/'passed.json').write_text(json.dumps(dict(calls=len(rows),deterministic=True,unaffected_exact=True)))
    print(f'Captured {len(rows)} isolated projection calls',flush=True)


if __name__=='__main__': main()
