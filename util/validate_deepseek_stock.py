"""Post-fix stock repetition gate, followed only by the original short plan gate."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extension-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    sys.path.insert(1,str(args.extension_dir.resolve()))
    import torch
    from exllamav3 import Config,Model,Cache,Tokenizer
    from exllamav3.modules import dsv4,native_plan
    from exllamav3.ext import exllamav3_ext as ext
    from util.validate_deepseek_plan import direct
    args.output.mkdir(parents=True,exist_ok=False)
    def save(n,v): (args.output/n).write_text(json.dumps(v,indent=2))
    save('extension.json',dict(extension=ext.__file__))
    with torch.inference_mode():
        native_plan.enabled=False
        cfg=Config.from_directory(r'D:\tabbyAPI\models\turboderp_DeepSeek-V4-Flash-0731-exl3_3.04bpw')
        model=Model.from_config(cfg)
        cache=Cache(model,max_num_tokens=8192,max_batch_size=4,max_history=5)
        tokenizer=Tokenizer.from_config(cfg)
        model.load(use_per_device=[85,40],max_chunk_size=256,max_batch_size=4)
        for graph in (True,False):
            dsv4.bc_dsa_enable=graph
            baseline=None
            for repeat in range(args.repeats):
                r=direct(model,cache,tokenizer,1,1,8,300,False)
                save(f'stock-graph{int(graph)}-{repeat}.json',r)
                if baseline is None: baseline=r
                else: assert r==baseline,f'stock state mismatch graph={graph} repeat={repeat}'
            print(f'PASS {args.repeats} exact stock runs graph={graph}',flush=True)
        dsv4.bc_dsa_enable=True
        stock=direct(model,cache,tokenizer,1,1,32,300,False)
        save('short-stock.json',stock)
        plan=direct(model,cache,tokenizer,1,1,32,300,True)
        save('short-plan.json',plan)
        assert stock['records']==plan['records']
        assert stock['boundaries']==plan['boundaries']
        save('passed.json',dict(stock_repetitions=args.repeats,graph_and_eager=True,short_steps=32,
             plan_calls=plan['counts'],exact=True))
        print('PASS original 32-step stock/plan persistent-state gate',flush=True)
        model.unload()


if __name__=='__main__': main()
