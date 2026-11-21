"""Installed-wheel validation through TabbyAPI's real backend (run with its Python -I).

Modes: memory, minimal, full. Writes evidence under native-results/wddm-autosplit.
Does not change Tabby's saved configuration or checkpoint files.
"""
import asyncio
from collections import Counter
import importlib.metadata as md
from functools import partial
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
TABBY=Path('D:/tabbyAPI')
OUT=ROOT/'native-results/wddm-autosplit'
MODE=sys.argv[1]
assert MODE in ('memory','minimal','full')
LABEL=MODE+('-native' if os.environ.get('EXL3_NATIVE_PLAN')=='1' else '')
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1,2'
assert os.environ.get('EXL3_WDDM_MEMORY')=='1'
assert os.environ.get('EXL3_AUTOSPLIT_WORSTCASE','1')=='1'
sys.path.insert(0,str(TABBY))
import torch
import exllamav3
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.util import memory
from exllamav3.util.wddm_memory import _nvml

assert Path(exllamav3.__file__).is_relative_to(TABBY/'venv')
assert Path(ext.__file__).is_relative_to(TABBY/'venv')
assert torch.cuda.device_count()==2
identities=[]
for i in range(2):
    p=torch.cuda.get_device_properties(i)
    assert 'RTX PRO 6000' in p.name and 95*2**30<p.total_memory<97*2**30
    identities.append(dict(logical=i,name=p.name,uuid=str(p.uuid),total=p.total_memory,
                           pci=f'{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0'))


def snapshot():
    rows=[]
    for i in range(2):
        n=_nvml().memory(('GPU-'+identities[i]['uuid']).encode())
        rows.append(dict(device=i,cuda_free=torch.cuda.mem_get_info(i)[0],
                         physical_free=n.free,physical_used=n.used,driver_reserved=n.reserved,
                         corrected_free=memory.device_mem_info(i)[0],
                         allocated=torch.cuda.memory_allocated(i),reserved=torch.cuda.memory_reserved(i)))
    return rows


report=dict(mode=MODE,python=sys.executable,torch=torch.__version__,version=md.version('exllamav3'),
            package=exllamav3.__file__,extension=ext.__file__,devices=identities,start=snapshot())
OUT.mkdir(parents=True,exist_ok=True)


def save():
    (OUT/(LABEL+'.json')).write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')


def memory_test():
    buffers=[]
    for i,gib in enumerate((.25,.5)):
        t=torch.empty(int(gib*2**30),device=f'cuda:{i}',dtype=torch.uint8)
        t.fill_(17);torch.cuda.synchronize(i);buffers.append(t)
    report['with_allocations']=snapshot()
    report['caps']=[]
    for i in range(2):
        memory.touch_device(i)
        torch.cuda.synchronize(i)
        current=torch.cuda.memory_reserved(i)
        cap=memory.set_memory_fraction_use(32<<20,i)
        assert abs(cap-current-(32<<20))<1<<20
        try:
            torch.empty(128<<20,device=f'cuda:{i}',dtype=torch.uint8)
        except torch.cuda.OutOfMemoryError:
            report['caps'].append(dict(device=i,current=current,cap=cap,low_budget_rejected=True))
        else:
            raise AssertionError('Explicit low memory cap was not enforced')
        finally:
            memory.unset_memory_fraction([i])
    del t
    buffers.clear();memory.free_mem()


async def model_test():
    from backends.exllamav3.model import ExllamaV3Container
    from common.transformers_utils import HFModel
    from common.sampling import BaseSamplerRequest
    from common.networking import DisconnectHandler
    from starlette.requests import Request
    from exllamav3.model import model_ls
    from exllamav3.modules import native_moe, native_plan, native_deepseek_v41
    model_path=TABBY/'models/DeepSeek-V4.1-Flash-2.1bpw'
    assert model_path.is_dir()
    report['checkpoint']=str(model_path.resolve())
    full=MODE=='full'
    context=65536 if full else 8192
    split=[60,94] if full else [90,90]
    kwargs=dict(max_seq_len=context,cache_size=context,cache_mode='FP16',max_batch_size=4,
                chunk_size=4096,tensor_parallel=False,gpu_split_auto=False,gpu_split=split,
                vision=False,ngram_ram=False,cpu_moe_offload_layers=0,cpu_moe_split_experts=0,
                draft_model=dict(draft_mode='mtp' if full else 'disabled',draft_num_tokens=4,
                                 dynamic_draft=False,draft_cache_mode='FP16',draft_gpu_split=[90,90]))
    report['configuration']=kwargs
    report['budgets']=[]
    original=model_ls.set_memory_fraction_use
    def observed(use,device):
        before=snapshot()
        result=original(use,device)
        report['budgets'].append(dict(device=device,use=use,cap=result,before=before))
        save()
        print('BUDGET',device,use/2**30,result/2**30,flush=True)
        return result
    model_ls.set_memory_fraction_use=observed
    container=await ExllamaV3Container.create(model_path,await HFModel.from_directory(model_path),**kwargs)
    for model in (container.model,container.draft_model):
        if model is not None:
            model.load_gen=partial(model.load_gen,verbose=True)
    assert container.config.infer_params.ngram_stream_from_disk
    assert not container.use_tp
    begin=time.monotonic()
    async for progress in container.load_gen():
        pass
    report['load_seconds']=time.monotonic()-begin
    report['loaded']=container.loaded
    report['after_load']=snapshot()
    report['placement']=dict(Counter(str(m.device) for m in container.model.modules))
    report['cpu_modules']=[m.key for m in container.model.modules if str(m.device)=='cpu']
    assert not any(m.key.startswith('layers.') for m in container.model.modules if str(m.device)=='cpu')
    g=container.generator.generator
    report['generator']=dict(drafts=g.num_draft_tokens,dynamic=g.dynamic_draft,
                             confidence_calibrator=g.draft_calibrator,mtp=g.mtp_draft,
                             dflash=g.dflash_draft,
                             dspark_confidence=getattr(container.draft_model,'draft_conf_threshold',None),
                             native_moe=native_moe.enabled,native_plan=native_plan.enabled,
                             native_graphs=native_deepseek_v41.graphs_enabled)
    assert g.draft_calibrator is None  # no confidence truncation with dynamic draft disabled
    if full:
        # This checkpoint's native MTP component is a DSpark block drafter,
        # dispatched through the generator's DFlash path, not AR-MTP.
        assert container.draft_model.component=='mtp' and g.dflash_draft
        assert g.num_draft_tokens==4 and not g.dynamic_draft
        assert container.draft_model.draft_conf_threshold==0.0
    counts=Counter()
    original_forward=container.model.forward
    def forward(input_ids,params=None):
        counts[str(tuple(input_ids.shape))]+=1
        return original_forward(input_ids,params)
    container.model.forward=forward
    native_calls=Counter()
    original_native=native_moe.run
    def native_run(*args,**kw):
        result=original_native(*args,**kw)
        native_calls['executed' if result is not None else 'fallback']+=1
        return result
    native_moe.run=native_run
    prompt='User: What is the capital of France? Answer in one short sentence.\nAssistant:'
    params=BaseSamplerRequest(max_tokens=32,min_tokens=16,temperature=0,top_k=1,top_p=1,min_p=0,
                              repetition_penalty=1,frequency_penalty=0,presence_penalty=0,
                              token_healing=False,add_bos_token=False)
    report['generations']=[]
    save()
    for i in range(2):
        async def receive():
            return await asyncio.Future()
        handler=DisconnectHandler(Request({'type':'http'},receive=receive),f'wddm-{MODE}-{i}')
        chunks=[]
        try:
            async for chunk in container.stream_generate(f'wddm-{MODE}-{i}',prompt,params,handler):
                if chunk is not None:
                    chunks.append(chunk)
        finally:
            await handler.cleanup()
        result=dict(text=''.join(c.get('text') or '' for c in chunks),chunks=chunks)
        assert result['text'].strip(),result
        report['generations'].append(result)
        save()
    assert report['generations'][0]['text']==report['generations'][1]['text']
    report['forward_shapes']=dict(counts)
    report['native_moe_calls']=dict(native_calls)
    report['native_plan_cache_entries']=sum(len(getattr(m,'_native_deepseek',{}))
                                           for model in (container.model,container.draft_model) if model is not None
                                           for m in model.modules)
    if native_plan.enabled:
        assert report['native_plan_cache_entries']>0
    report['after_generation']=snapshot()
    print('GENERATION',report['generations'][0]['text'],flush=True)
    await container.unload()


stop=threading.Event()
def monitor():
    with (OUT/(LABEL+'-memory.jsonl')).open('w') as f:
        while not stop.wait(2):
            f.write(json.dumps(dict(time=time.time(),devices=snapshot()))+'\n');f.flush()


if __name__=='__main__':
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    save()
    try:
        if MODE=='memory':memory_test()
        else:asyncio.run(model_test())
        report['success']=True
    except BaseException:
        report['error']=traceback.format_exc()
        raise
    finally:
        stop.set();watcher.join(timeout=5)
        report['end']=snapshot();save()
