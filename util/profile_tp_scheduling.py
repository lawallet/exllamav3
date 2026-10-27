"""R3 diagnostic observer for profile_final_ir; never used in throughput runs.

Use --host-timers --profile-tokens N --runs 0. EXL3_R3_OBSERVER selects
torch (CUPTI + host scopes), host (host-only timers), helper (CPU reducer only), nsight (NVTX + cudaProfilerApi), or events
(CUDA event brackets without CUPTI). Runtime wrappers preserve all calls.
"""
import ctypes
import json
import os
from pathlib import Path
import sys
import time

import profile_final_ir as driver

ROOT=Path(__file__).resolve().parents[1]
STATE=None


class Processor(ctypes.Structure):
    _fields_=[('group',ctypes.c_ushort),('number',ctypes.c_ubyte),('reserved',ctypes.c_ubyte)]


def cpu_number():
    if os.name=='nt':
        p=Processor()
        ctypes.windll.kernel32.GetCurrentProcessorNumberEx(ctypes.byref(p))
        return [p.group,p.number]
    return -1


def placement():
    result=dict(pid=os.getpid(),processor=cpu_number())
    if os.name=='nt':
        from ctypes import wintypes as w
        class Affinity(ctypes.Structure):
            _fields_=[('mask',ctypes.c_size_t),('group',w.WORD),('reserved',w.WORD*3)]
        k=ctypes.windll.kernel32
        k.GetCurrentThread.restype=w.HANDLE
        k.GetThreadGroupAffinity.argtypes=[w.HANDLE,ctypes.POINTER(Affinity)]
        a=Affinity()
        if k.GetThreadGroupAffinity(k.GetCurrentThread(),ctypes.byref(a)):
            result.update(tid=k.GetCurrentThreadId(),group=a.group,mask=hex(a.mask))
    return result


def helper_trace(context, start, path):
    from exllamav3.ext import exllamav3_ext as ext
    if not hasattr(ext,'tp_profile_begin'):
        return dict(available=False)
    if start:
        context['r3_helper_placement']=placement()
        ext.tp_profile_begin(32768)
        return dict(available=True)
    calibration=[]
    for _ in range(9):
        before=time.perf_counter_ns()
        clock=ext.tp_profile_clock_ns()
        after=time.perf_counter_ns()
        calibration.append(dict(python_before=before,cpp=clock,python_after=after))
    rows=ext.tp_profile_end()
    fields=['queue_begin','ready','done','bytes','mask','wire','math_ns','published',
            'sleeps','sleep_ns','chunks','tier']+['arrival_'+str(i) for i in range(16)]
    data=dict(fields=fields,rows=rows,dropped=ext.tp_profile_dropped(),
              placement_start=context.pop('r3_helper_placement'),placement_end=placement(),
              clock=min(calibration,key=lambda c:c['python_after']-c['python_before']),
              clock_calibration=calibration)
    Path(path).write_text(json.dumps(data),encoding='utf-8')
    return dict(available=True,records=len(rows),dropped=data['dropped'])


def scoped(label, fn, *args, metadata=None, **kwargs):
    import torch
    state=STATE
    if state is None:
        return fn(*args,**kwargs)
    if len(state['events'])>=100000:
        state['dropped']+=1
        return fn(*args,**kwargs)
    event_pair=None
    cpu=cpu_number()
    if state['observer']=='events' and label.startswith(('R3.collective.', 'R3.attn.', 'R3.mlp.')):
        event_pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
        event_pair[0].record(torch.cuda.current_stream(state['device']))
    scope=None
    if state['observer']=='torch':
        scope=torch.profiler.record_function(label)
        scope.__enter__()
    elif state['observer']=='nsight':
        torch.cuda.nvtx.range_push(label)
    start=time.perf_counter_ns()
    try:
        return fn(*args,**kwargs)
    finally:
        end=time.perf_counter_ns()
        if scope is not None:
            scope.__exit__(None,None,None)
        elif state['observer']=='nsight':
            torch.cuda.nvtx.range_pop()
        if event_pair:
            event_pair[1].record(torch.cuda.current_stream(state['device']))
            state['cuda_events'].append((len(state['events']),event_pair))
        state['events'].append(dict(name=label,start_ns=start,end_ns=end,
                                    cpu=cpu,cpu_end=cpu_number(),metadata=metadata))


class ObservedPlan:
    def __init__(self,plan,key): self.plan,self.key=plan,key
    def __getattr__(self,name): return getattr(self.plan,name)
    def begin(self,x): return scoped('R3.ir.begin.'+self.key,self.plan.begin,x)
    def middle(self,x,y): return scoped('R3.ir.middle.'+self.key,self.plan.middle,x,y)
    def finish(self,x,y): return scoped('R3.ir.finish.'+self.key,self.plan.finish,x,y)


def anchor():
    import torch
    for _ in range(5):
        before=time.perf_counter_ns()
        name='R3.anchor:'+str(before)
        if STATE['observer']=='torch':
            with torch.profiler.record_function(name):
                inside=time.perf_counter_ns()
                before_exit=time.perf_counter_ns()
            after=time.perf_counter_ns()
            STATE['anchors'].append(dict(name=name,before=before,inside=inside,before_exit=before_exit,after=after))
        elif STATE['observer']=='nsight':
            torch.cuda.nvtx.range_push(name)
            torch.cuda.nvtx.range_pop()


def rank_observer(context,start):
    global STATE
    import torch
    from exllamav3.modules import TransformerBlock
    observer=os.environ.get('EXL3_R3_OBSERVER','torch')
    mode='graphs' if context.get('final_ir_graphs') else 'native'
    path=Path(os.environ['EXL3_R3_OUTPUT'])/('r3-'+mode)
    if not start:
        state=STATE
        stopped=time.perf_counter_ns()
        anchor()
        if state['profiler'] is not None:
            state['profiler'].__exit__(None,None,None)
            state['profiler'].export_chrome_trace(str(path)+f'-rank{context["rank"]}.json')
        elif observer=='nsight' and context['device']==context['output_device']:
            torch.cuda.cudart().cudaProfilerStop()
        if state['cuda_events']:
            torch.cuda.synchronize(context['device'])
            for index,(a,b) in state['cuda_events']:
                state['events'][index]['gpu_ms_inclusive']=a.elapsed_time(b)
        STATE=None
        for obj,name,original in reversed(state['saved']): setattr(obj,name,original)
        data=dict(rank=context['rank'],device=context['device'],observer=observer,
            pid=os.getpid(),events=state['events'],dropped=state['dropped'],
            anchors=state['anchors'],
            output_device=context['output_device'],window_ns=[state['started_ns'],stopped],
            placement_start=state['placement'],placement_end=placement(),
            clock=dict(perf_ns=time.perf_counter_ns(),epoch_ns=time.time_ns()))
        Path(str(path)+f'-rank{context["rank"]}-host.json').write_text(json.dumps(data),encoding='utf-8')
        totals={}
        for e in data['events']:
            v=totals.setdefault(e['name'],[0,0])
            v[0]+=1; v[1]+=e['end_ns']-e['start_ns']
        return dict(rank=context['rank'],device=context['device'],totals=totals,dropped=data['dropped'])
    assert STATE is None
    STATE=dict(observer=observer,device=context['device'],saved=[],events=[],cuda_events=[],anchors=[],dropped=0,profiler=None,placement=placement())
    if observer=='torch':
        STATE['profiler']=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
        STATE['profiler'].__enter__()
    elif observer=='nsight' and context['device']==context['output_device']:
        torch.cuda.cudart().cudaProfilerStart()
    anchor()
    if observer=='helper':
        STATE['started_ns']=time.perf_counter_ns()
        return
    def wrap(obj,name,label):
        original=getattr(obj,name)
        STATE['saved'].append((obj,name,original))
        def call(*args,**kwargs):
            metadata=None
            if label=='R3.collective.all_reduce':
                t=args[0]
                metadata=dict(shape=list(t.shape),dtype=str(t.dtype),wire_bytes=t.numel()*2,
                              contributor=kwargs.get('contribution',args[1] if len(args)>1 else True))
            return scoped(label,original,*args,metadata=metadata,**kwargs)
        setattr(obj,name,call)
    for block in context['modules']:
        if not isinstance(block,TransformerBlock): continue
        for name in ('attn','mlp'):
            child=getattr(block,name)
            if child is not None: wrap(child,'forward','R3.'+name+'.'+block.key)
        get=block._native_plan_get
        if get is None: continue
        STATE['saved'].append((block,'_native_plan_get',get))
        plans={}
        def select(b,x,params,get=get,key=block.key,plans=plans):
            p=scoped('R3.ir.select.'+key,get,b,x,params)
            if p is None: return None
            if id(p) not in plans: plans[id(p)]=ObservedPlan(p,key)
            return plans[id(p)]
        block._native_plan_get=select
    for name in ('all_reduce','broadcast','gather','gather_small','fwd_barrier','end_cpu_reduce_jobs'):
        wrap(context['backend'],name,'R3.collective.'+name)
    if context['device']==context['output_device']:
        from exllamav3.model.model_tp import Model_TPMixin as ModelTP
        from multiprocessing.connection import _ConnectionBase
        for name in ('forward_tp','tp_drain_acks','prepare_inputs_for_tp','restore_tp_params',
                     'tp_worker_dispatch','tp_worker_result'):
            wrap(ModelTP,name,'R3.parent.'+name)
        # Windows PipeConnection inherits this implementation; Connection is a
        # different transport class and would miss native Windows named pipes.
        wrap(_ConnectionBase,'send_bytes','R3.parent.pipe_send_bytes')
        wrap(_ConnectionBase,'send','R3.parent.pipe_send')
    STATE['started_ns']=time.perf_counter_ns()


def install():
    from exllamav3.model.model_tp import Model_TPMixin as ModelTP
    original_dispatch=ModelTP.tp_dispatch_all
    original_wait=ModelTP.tp_worker_dispatch_wait_multi
    def dispatch(self,func,args):
        if func is rank_observer and args==(True,):
            mode='graphs' if getattr(self,'_r3_graph_mode',False) else 'native'
            path=str(Path(os.environ['EXL3_R3_OUTPUT'])/('r3-'+mode+'-helper.json'))
            self._r3_helper_path=path
            self.tp_worker_dispatch_single(-1,helper_trace,(True,path))
        return original_dispatch(self,func,args)
    def wait(self,devices,func,args,dev_args=None):
        result=original_wait(self,devices,func,args,dev_args)
        if func is rank_observer and args==(False,):
            self.tp_worker_dispatch_single(-1,helper_trace,(False,self._r3_helper_path))
        return result
    ModelTP.tp_dispatch_all=dispatch
    ModelTP.tp_worker_dispatch_wait_multi=wait
    # Mode changes use dispatch_all; retain a parent copy solely for output names.
    original=ModelTP.tp_dispatch_all
    def dispatch_mode(self,func,args):
        if func is driver.set_mode: self._r3_graph_mode=bool(args[1])
        return original(self,func,args)
    ModelTP.tp_dispatch_all=dispatch_mode
    driver.rank_timers=rank_observer


def main():
    output=sys.argv[sys.argv.index('--output')+1]
    extension=sys.argv[sys.argv.index('--extension-dir')+1]
    os.environ['EXL3_R3_OUTPUT']=str(Path(output).resolve())
    sys.path.insert(1,str(Path(extension).resolve()))
    assert '--host-timers' in sys.argv and '--trace-tokens' not in sys.argv
    assert os.environ.get('EXL3_R3_OBSERVER','torch') in ('torch','host','helper','nsight','events')
    os.environ.update(EXL3_NATIVE_PLAN='1',EXL3_NATIVE_GLM_BLOCK='1',EXL3_NATIVE_QWEN_BLOCK='1')
    install()
    source=Path(__file__).read_bytes()
    try:
        driver.main()
    finally:
        folder=Path(output)
        if folder.is_dir():
            (folder/'observer.py').write_bytes(source)
            flags=('EXL3_R3_OBSERVER','EXL3_GRAPHS','EXL3_NATIVE_TP_GRAPHS',
                   'EXL3_TP_NO_FWD_BARRIER','EXL3_TP_REDUCE_MAX_ISA','EXL3_TP_REDUCE_THREADS')
            (folder/'observer-env.json').write_text(json.dumps({k:os.environ.get(k) for k in flags},indent=2))


if __name__=='__main__': main()
