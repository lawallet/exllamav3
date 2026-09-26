"""Trace the unmodified generator's compact batch, per-job slots and MTP windows."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path('native-results/glm53-mtp/wheel-stage').resolve()))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler
from exllamav3.modules import transformer

PROMPTS = [
    'Write a Python function that computes prime numbers. Explain the sieve algorithm step by step.',
    'Describe how to bake a chocolate cake, including ingredients and detailed preparation instructions.',
    'Explain how ocean tides arise from gravity. Describe the roles of the Moon and Sun.',
    'Write a fantasy story about a traveling musician who discovers a hidden city in the mountains.',
    'Explain how a relational database index speeds up queries, with a concrete SQL example.',
]


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2,0'
    torch.cuda.set_device(0)
    cfg = Config.from_directory(r'D:\tabbyAPI\models\GLM-5.3-Flash-Uncensored-EXL3-3.05bpw')
    model, draft = Model.from_config(cfg), Model.from_config(cfg, component='mtp')
    cache, dc = Cache(model, 5120, max_batch_size=4, max_history=0), Cache(draft, 5120, max_batch_size=4)
    tokenizer = Tokenizer.from_config(cfg)
    model.load(use_per_device=[90,44],max_batch_size=4,max_chunk_size=256)
    transformer._native_glm_enabled = False
    trace = []
    shapes=[]
    from exllamav3.modules import TransformerBlock
    for mod in model.modules:
        if isinstance(mod,TransformerBlock):
            original=mod.forward
            def block(x,params,_f=original,_i=mod.layer_idx):
                if _i in (0,35): shapes.append(dict(layer=_i,shape=list(x.shape),device=str(x.device),history=params.get('recurrent_history')))
                return _f(x,params)
            mod.forward=block
    for mtp in (False,):
        gen = Generator(model, cache, tokenizer, max_batch_size=4, draft_model=draft if mtp else None,
                        draft_cache=dc if mtp else None, num_draft_tokens=3 if mtp else 0,
                        dynamic_draft_tokens=False, record_draft_stats=True)
        jobs = [Job(tokenizer.encode(p), max_new_tokens=n, min_new_tokens=n, stop_conditions=[],
                    sampler=GreedySampler(), seed=1234+i, identifier=str(i))
                for i, (p, n) in enumerate(zip(PROMPTS, [6, 10, 14, 18, 8]))]
        old = gen.iterate_gen
        def verify(results, draft_tokens=None):
            active = [j for j in gen.active_jobs if j.is_prefill_done()]
            before = [(j.accepted_draft_tokens, j.rejected_draft_tokens) for j in active]
            item = dict(mtp=mtp, jobs=[j.identifier for j in active],
                        q=1 if draft_tokens is None else draft_tokens.shape[1]+1,
                        slots=[j.recurrent_state.slot for j in active],
                        positions=[j.sequences[0].kv_position for j in active],
                        pages=[[p.page_index for p in j.sequences[0].allocated_pages] for j in active])
            old(results, draft_tokens)
            item['accept_reject'] = [(j.accepted_draft_tokens-a, j.rejected_draft_tokens-r)
                                     for j,(a,r) in zip(active,before)]
            trace.append(item)
        with patch.object(gen, 'iterate_gen', verify):
            gen.enqueue(jobs[:4])
            joined = False
            while gen.num_remaining_jobs():

                for r in gen.iterate():
                    if r.get('stage')=='error':raise r['error']
                if not joined and jobs[1].is_finished:
                    gen.enqueue(jobs[4]); joined = True
        gen.filter_pool.shutdown(wait=True)
        print('MTP', mtp, 'shapes', sorted({(len(r['jobs']),r['q']) for r in trace if r['mtp']==mtp}), flush=True)
    out = Path('native-results/glm53-concurrency/trace-ordinary.json')
    out.write_text(json.dumps(dict(events=trace,blocks=shapes,placement=[dict(key=m.key,device=str(m.device)) for m in model.modules]),indent=2))
    print('Saved',out)


if __name__ == '__main__':
    main()
