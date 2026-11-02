"""Temporary diagnostics over the unchanged installed wheel. All hooks opt in."""
import importlib.util, json, os, sys, threading, time, statistics
from pathlib import Path
spec=importlib.util.spec_from_file_location('ds41_base',Path(__file__).with_name('serve_deepseek_v41_ir.py'))
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
dxgi_spec=importlib.util.spec_from_file_location('ds41_dxgi',Path(__file__).with_name('deepseek_v41_dxgi.py'))
dxgi=importlib.util.module_from_spec(dxgi_spec);dxgi_spec.loader.exec_module(dxgi)
torch=b.torch
from exllamav3.modules.row_table import RowTable
from exllamav3.util import device_copy
from exllamav3.modules import native_deepseek_v41 as binding

hooks=[];rows={};local=threading.local();samples={};active=False;mode='off';first_ns=None
original_affinity=None
original_blocktime=None
def blocktime(value=None):
    import ctypes as C
    global original_blocktime
    try:
        runtime=C.CDLL('libiomp5md.dll');runtime.kmp_get_blocktime.restype=C.c_int
        runtime.kmp_set_blocktime.argtypes=[C.c_int];runtime.kmp_set_blocktime.restype=None
    except (OSError,AttributeError) as error:return {'unavailable':str(error)}
    if original_blocktime is None:original_blocktime=runtime.kmp_get_blocktime()
    if value is not None:
        value=original_blocktime if value=='restore' else value
        assert isinstance(value,int) and value>=0
        runtime.kmp_set_blocktime(value)
    return dict(milliseconds=runtime.kmp_get_blocktime(),original=original_blocktime)
def main_affinity(group=None):
    import ctypes as C
    global original_affinity
    class Affinity(C.Structure):
        _fields_=[('mask',C.c_size_t),('group',C.c_ushort),('reserved',C.c_ushort*3)]
    class Processor(C.Structure):
        _fields_=[('group',C.c_ushort),('number',C.c_ubyte),('reserved',C.c_ubyte)]
    k=C.WinDLL('kernel32',use_last_error=True);k.GetCurrentThread.restype=C.c_void_p
    k.GetThreadGroupAffinity.argtypes=[C.c_void_p,C.POINTER(Affinity)]
    k.SetThreadGroupAffinity.argtypes=[C.c_void_p,C.POINTER(Affinity),C.POINTER(Affinity)]
    k.GetActiveProcessorCount.argtypes=[C.c_ushort];k.GetActiveProcessorCount.restype=C.c_ulong
    h=k.GetCurrentThread();current=Affinity()
    assert k.GetThreadGroupAffinity(h,C.byref(current))
    if original_affinity is None:original_affinity=(current.group,current.mask)
    if group is not None:
        desired=Affinity()
        if group=='restore':desired.group,desired.mask=original_affinity
        else:
            assert group in (0,1)
            count=k.GetActiveProcessorCount(group);assert 0<count<=64
            desired.group=group;desired.mask=(1<<count)-1
        previous=Affinity();assert k.SetThreadGroupAffinity(h,C.byref(desired),C.byref(previous))
        assert k.GetThreadGroupAffinity(h,C.byref(current))
    processor=Processor();k.GetCurrentProcessorNumberEx(C.byref(processor))
    return dict(group=current.group,mask=hex(current.mask),original=original_affinity,thread_id=threading.get_native_id(),
        current_processor=dict(group=processor.group,number=processor.number))
def record(label,wall,cpu,exclusive,exclusive_cpu):
    r=rows.setdefault(label,dict(count=0,wall_ns=0,cpu_ns=0,exclusive_ns=0,exclusive_cpu_ns=0,max_ns=0))
    r['count']+=1;r['wall_ns']+=wall;r['cpu_ns']+=cpu;r['exclusive_ns']+=exclusive;r['exclusive_cpu_ns']+=exclusive_cpu;r['max_ns']=max(r['max_ns'],wall)

def measured(label,fn,*args,**kwargs):
    if not active:return fn(*args,**kwargs)
    stack=getattr(local,'stack',None)
    if stack is None:stack=local.stack=[]
    frame=[0,0];stack.append(frame);c=time.thread_time_ns();w=time.perf_counter_ns()
    try:return fn(*args,**kwargs)
    finally:
        wall=time.perf_counter_ns()-w;cpu=time.thread_time_ns()-c;stack.pop()
        if stack:stack[-1][0]+=wall;stack[-1][1]+=cpu
        record(label,wall,cpu,wall-frame[0],cpu-frame[1])

def hook(obj,name,fn):
    old=getattr(obj,name);hooks.append((obj,name,old));setattr(obj,name,fn(old))

def uninstall():
    global active
    active=False
    for obj,name,old in reversed(hooks):setattr(obj,name,old)
    hooks.clear()

def install(kind):
    global mode,rows,samples,first_ns
    uninstall();mode=kind;rows={};samples={};first_ns=None
    if kind=='off':return
    if kind=='prefill':
        from exllamav3.generator.job import Job
        hook(Job,'prefill',lambda old:lambda *a,**kw:measured('Job.prefill',old,*a,**kw))
        for name in ('_stage','_gather_rows','lookup'):
            hook(RowTable,name,lambda old,n=name:lambda *a,**kw:measured('RowTable.'+n,old,*a,**kw))
        for name in dir(b.Generator):
            if name.startswith('iterate_') and callable(getattr(b.Generator,name)):
                hook(b.Generator,name,lambda old,n=name:lambda *a,**kw:measured('Generator.'+n,old,*a,**kw))
        for role,model in (('target',b.serving.container.model),('draft',b.serving.container.draft_model)):
            if model:
                hook(model,'forward',lambda old,r=role:lambda *a,**kw:measured('Model.'+r+'|q='+str((a[0] if a else kw['input_ids']).shape[-1]),old,*a,**kw))
                hook(model,'prefill',lambda old,r=role:lambda *a,**kw:measured('Model.prefill.'+r+'|q='+str((a[0] if a else kw['input_ids']).shape[-1]),old,*a,**kw))
        return
    for name in ('to','cpu','copy_'):
        def factory(old,name=name):
            def wrapped(x,*args,**kwargs):
                if not active:return old(x,*args,**kwargs)
                caller=sys._getframe(1);site=Path(caller.f_code.co_filename).name+':'+str(caller.f_lineno)
                src=args[0] if name=='copy_' and args and isinstance(args[0],torch.Tensor) else x
                label=f'Tensor.{name}|{site}|{src.device}|{src.dtype}|{src.numel()*src.element_size()}B|args={str(args[0]) if args and not isinstance(args[0],torch.Tensor) else "tensor_or_none"}|nb={kwargs.get("non_blocking",False)}'
                result=measured(label,old,x,*args,**kwargs)
                key=label+f'|result={result.device}|identity={result is x}'
                r=rows.pop(label,None)
                if r:
                    dst=rows.setdefault(key,{k:0 for k in r})
                    for k,v in r.items():dst[k]=max(dst[k],v) if k=='max_ns' else dst[k]+v
                return result
            return wrapped
        hook(torch.Tensor,name,factory)
    for name in ('_stage','_gather_rows','lookup'):
        if hasattr(RowTable,name):hook(RowTable,name,lambda old,n=name:lambda *a,**kw:measured('RowTable.'+n,old,*a,**kw))
    for name in ('synchronize',):hook(torch.cuda,name,lambda old:lambda *a,**kw:measured('cuda.synchronize',old,*a,**kw))
    for cls in (torch.cuda.Event,torch.cuda.Stream):
        hook(cls,'synchronize',lambda old,cl=cls:lambda *a,**kw:measured(cl.__name__+'.synchronize',old,*a,**kw))
    for name in dir(b.Generator):
        if name.startswith('iterate_') and callable(getattr(b.Generator,name)):
            hook(b.Generator,name,lambda old,n=name:lambda *a,**kw:measured('Generator.'+n,old,*a,**kw))
    c=b.serving.container
    for role,model in (('target',c.model),('draft',c.draft_model)):
        if model is None:continue
        hook(model,'forward',lambda old,r=role:lambda *a,**kw:measured('Model.'+r,old,*a,**kw))
        for block in model.modules:
            if getattr(block,'_native_carry_pre',False):
                def factory(old):
                    def guard(block,x,params):
                        if active and len(samples)<4:
                            key=(str(x.device),x.shape[1],type(block.attn).__name__)
                            if key not in samples:samples[key]=(block,x,params.copy(),old)
                        return measured('native.guard',old,block,x,params)
                    return guard
                hook(block,'_native_plan_get',factory)

old_iterate=b.Generator.iterate
def iterate(self,*args,**kwargs):
    global active,first_ns
    generated=max((j.new_tokens for j in self.active_jobs),default=0)
    active=(generated<1 if mode=='prefill' else mode!='off' and generated>=64)
    if active and first_ns is None:first_ns=time.time_ns()
    results=measured('Generator.iterate',old_iterate,self,*args,**kwargs)
    if any(r.get('eos') for r in results):
        active=False
        if mode!='off':
            (b.OUT/(b.state['name']+'-host.json')).write_text(json.dumps(dict(first_ns=first_ns,end_ns=time.time_ns(),mode=mode,
                rows=rows,status=extra_status()),indent=2))
    return results
b.Generator.iterate=iterate

def extra_status():
    g=b.serving.container.generator.generator
    try:
        from exllamav3.util.openmp import openmp_status
        implemented_policy=openmp_status()
    except ImportError:
        implemented_policy=None
    return dict(pid=os.getpid(),torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),blocktime=blocktime(),
        implemented_policy=implemented_policy,
        generator={k:v for k,v in vars(g).items() if ('chunk' in k or 'prefetch' in k) and isinstance(v,(str,int,bool,float,type(None)))},
        copy_stats=device_copy.stats,copy_verdicts={str(k):v for k,v in device_copy._verdicts.items()},
        copy_forced=device_copy.FORCED,thread_id=threading.get_native_id(),
        environment={k:v for k,v in os.environ.items() if k.startswith(('CUDA','EXL','OMP','KMP','MKL','OPENBLAS'))},
        main_affinity=main_affinity(),dxgi=dxgi.query(),memory=b.memory(),allocator=[{k:v for k,v in torch.cuda.memory_stats(i).items()
            if k.startswith(('num_','active_bytes.all.','inactive_split_bytes.all.'))} for i in (0,1)])

old_setup=b.server.setup_app
def setup(*args,**kwargs):
    app=old_setup(*args,**kwargs)
    @app.post('/__followup/control',dependencies=[b.Depends(b.check_admin_key)])
    async def control(body:dict):
        c=b.serving.container
        if c.active_job_ids or c.generator.jobs:raise b.HTTPException(409,'requests active')
        if 'mode' in body:install(body['mode'])
        if 'threads' in body:
            assert body['threads'] in (1,4,8,56)
            torch.set_num_threads(body['threads'])
        if 'main_group' in body:main_affinity(body['main_group'])
        if 'blocktime' in body:
            result=blocktime(body['blocktime'])
            if 'unavailable' in result:raise b.HTTPException(409,str(result))
        if 'chunk_size' in body:
            size=body['chunk_size'];assert size in (1024,2048,4096)
            c.generator.generator.max_chunk_size=size
            c.generator.generator.shared_chunk_size=min(c.generator.generator.shared_chunk_size,size)
        if 'serial_readback' in body:
            import importlib
            importlib.import_module('exllamav3.generator.generator')._NO_BATCH_READBACK=bool(body['serial_readback'])
        if body.get('empty_cache'):
            for i in (0,1):
                with torch.cuda.device(i):torch.cuda.synchronize();torch.cuda.empty_cache()
        if body.get('guard_microbench'):
            result=[];was=b.native_plan.enabled;b.native_plan.enabled=True
            try:
                import inspect,textwrap
                from exllamav3.modules.dsv4 import DSV4Attention
                from exllamav3.modules.arch_specific.dspark import DSparkAttention
                from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
                source=textwrap.dedent(inspect.getsource(binding.get_native_block))
                source='\n'.join(line for line in source.splitlines() if not line.startswith('    from .'))
                namespace=dict(binding.__dict__,DSV4Attention=DSV4Attention,DSparkAttention=DSparkAttention,BlockSparseMLP=BlockSparseMLP)
                exec(compile(source,'<guard_imports_hoisted>','exec'),namespace)
                hoisted=namespace['get_native_block']
                for key,(block,x,params,fn) in samples.items():
                    for _ in range(100):assert fn(block,x,params) is not None
                    assert hoisted(block,x,params) is fn(block,x,params)
                    for variant,f in [('original',fn),('imports_hoisted',hoisted)]:
                        timings=[]
                        for _ in range(7):
                            t=time.perf_counter_ns()
                            for j in range(2000):f(block,x,params)
                            timings.append((time.perf_counter_ns()-t)/2000)
                        result.append(dict(key=key,variant=variant,ns_per_call=timings,median_ns=statistics.median(timings)))
            finally:b.native_plan.enabled=was
            (b.OUT/'guard-microbench.json').write_text(json.dumps(result,indent=2))
            return result
        return extra_status()
    @app.get('/__followup/status',dependencies=[b.Depends(b.check_admin_key)])
    async def get_status():return extra_status()
    return app
b.server.setup_app=setup
if __name__=='__main__':
    import main
    main.entrypoint()
