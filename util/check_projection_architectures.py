"""Exact Qwen/GLM correctness regressions for the shared MGEMM fix; no benchmark."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--architecture',choices=['qwen','glm'],required=True)
    p.add_argument('--extension-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    a=p.parse_args()
    sys.path.insert(1,str(a.extension_dir.resolve()))
    import torch
    from exllamav3 import Config,Model,Cache,Tokenizer
    from exllamav3.modules import native_plan
    from exllamav3.ext import exllamav3_ext as ext
    with torch.inference_mode():
        native_plan.enabled=True
        name='Qwen3.8-Flash-Next-Uncensored-exl3-4bpw' if a.architecture=='qwen' else 'GLM-5.3-Flash-Uncensored-EXL3-3.05bpw'
        cfg=Config.from_directory(str(Path(r'D:\tabbyAPI\models')/name))
        model=Model.from_config(cfg)
        tokenizer=Tokenizer.from_config(cfg)
        prompt=(ROOT/f'doc/native_{"qwen" if a.architecture=="qwen" else "glm53"}_prompt.txt').read_text(encoding='utf-8')
        cache=Cache(model,max_num_tokens=2048,max_batch_size=1)
        if a.architecture=='qwen':
            from util.check_qwen_native import check
            model.load(device='cuda:0',max_chunk_size=256,max_batch_size=1)
        else:
            from util.check_glm_native import check
            model.load(use_per_device=[90,44],max_chunk_size=256,max_batch_size=1)
        result=check(model,cache,tokenizer,prompt,steps=128)
        a.output.write_text(json.dumps(dict(extension=ext.__file__,result=result),indent=2))
        if a.reference:
            previous=json.loads(a.reference.read_text())['result']
            # GLM includes placement tuples, which JSON reloads as lists.
            assert json.loads(json.dumps(result))==previous,'Existing architecture differs from pre-fix extension'
        model.unload()
        print(f'PASS {a.architecture} exact stock/bespoke/plan' + (' and pre-fix reference' if a.reference else ''),flush=True)


if __name__=='__main__': main()
