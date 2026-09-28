"""GPU regression: every matrix must be written, including sub-tile/tail widths.

No model required. Run in the pinned runtime with --extension-dir and --device.
The pre-fix extension fails for shape 3/128 and shape 4/128,256,384,768.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extension-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    sys.path.insert(0,str(args.extension_dir.resolve()))
    import torch
    import exllamav3_ext as ext
    torch.manual_seed(318)
    dev=torch.device(args.device)
    widths=[1024,128,256,384,512,768]
    k,bits=1024,5
    records=[]
    with torch.inference_mode():
        matrices=[torch.randint(0,65536,(k//16,n//16,16*bits),device=dev,dtype=torch.int32).to(torch.uint16) for n in widths]
        su=[torch.ones(k,device=dev,dtype=torch.half) for _ in widths]
        sv=[torch.ones(n,device=dev,dtype=torch.half) for n in widths]
        def ptr(ts): return torch.tensor([t.data_ptr() for t in ts],dtype=torch.long,device=dev)
        bp,sup,svp=ptr(matrices),ptr(su),ptr(sv)
        ns=torch.tensor(widths,dtype=torch.int32,device=dev)
        idx=torch.arange(len(widths),device=dev).unsqueeze(0)
        for rows in (1,2,4,5,12,17):
            x=torch.randn((1,rows,k),device=dev,dtype=torch.half)*0.05
            had=torch.empty((len(widths),rows,k),device=dev,dtype=torch.half)
            for dtype in (torch.half,torch.float):
                outputs=[torch.empty((rows,n),device=dev,dtype=dtype) for n in widths]
                cp=ptr(outputs)
                for shape in (1,2,3,4,-1):
                    baseline=None
                    for fill in (0,1,-3,0):
                        for t in outputs: t.fill_(fill)
                        selected=ext.exl3_mgemm(x,bp,outputs[0].view(1,rows,-1),sup,had,svp,idx,None,
                            bits,shape,0,1,-1,-1,0,1,ns,cp)
                        current=[t.cpu().clone() for t in outputs]
                        for n,t in zip(widths,current):
                            assert torch.isfinite(t).all() and torch.count_nonzero(t)>0, (args.device,rows,str(dtype),shape,selected,n,fill,'unwritten/nonfinite')
                        if baseline is None: baseline=current
                        else:
                            for n,a,b in zip(widths,baseline,current):
                                assert torch.equal(a,b),(args.device,rows,str(dtype),shape,n,fill)
                    records.append(dict(rows=rows,dtype=str(dtype),requested=shape,selected=selected,exact=True))
        # Column slices share an input transform and use a wider output row
        # stride. Verify the fallback covers each slice but never its neighbours.
        slice_widths=[128,256,384]
        offsets=[0,128,384]
        ns=torch.tensor(slice_widths,dtype=torch.int32,device=dev)
        strides=torch.full((3,),1024,dtype=torch.int32,device=dev)
        sources=torch.zeros(3,dtype=torch.int32,device=dev)
        bp=torch.tensor([matrices[0].data_ptr()+off*bits*2 for off in offsets],dtype=torch.long,device=dev)
        svp=torch.tensor([sv[0].data_ptr()+off*2 for off in offsets],dtype=torch.long,device=dev)
        sup=ptr(su[:1])
        for rows in (1,5,17):
            x=torch.randn((1,rows,k),device=dev,dtype=torch.half)*0.05
            had=torch.empty((1,rows,k),device=dev,dtype=torch.half)
            output=torch.empty((rows,1024),device=dev,dtype=torch.half)
            carrier=torch.empty((1,rows,384),device=dev,dtype=torch.half)
            cp=torch.tensor([output.data_ptr()+off*2 for off in offsets],dtype=torch.long,device=dev)
            for shape in (1,2,3,4,-1):
                baseline=None
                for fill in (0,1,-3,0):
                    output.fill_(fill)
                    selected=ext.exl3_mgemm(x,bp,carrier,sup,had,svp,None,None,bits,shape,0,1,-1,-1,0,1,
                                          ns,cp,strides,sources,1)
                    current=output.cpu().clone()
                    assert torch.equal(current[:,768:],torch.full_like(current[:,768:],fill)), 'slice wrote past its end'
                    assert torch.isfinite(current[:,:768]).all() and torch.count_nonzero(current[:,:768])>0
                    if baseline is None: baseline=current[:,:768]
                    else: assert torch.equal(baseline,current[:,:768]),(rows,shape,'slice prior-output dependency')
                records.append(dict(rows=rows,dtype='torch.float16',sliced=True,requested=shape,selected=selected,exact=True))
    args.output.write_text(json.dumps(dict(extension=ext.__file__,device=args.device,widths=widths,cases=records),indent=2))
    print(f'PASS {len(records)} variable-width MGEMM cases on {args.device}')


if __name__=='__main__': main()
