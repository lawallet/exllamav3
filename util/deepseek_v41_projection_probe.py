"""Reversible process-local projection capture; NOT a release implementation.

Capture the unchanged projection calls with private graph storage. Positional
rotation and every cache/index operation remain in the original Python caller.
"""
import torch
from exllamav3.modules import dsv4

original=dsv4.DSV4Attention._project_qkv
enabled=False
plans={}
checks=[]

def raw(a,x,params):
    qr=a.q_norm.forward(a.q_a.forward(x,params),params,out_dtype=torch.half)
    return qr,a.q_b.forward(qr,params),a.wkv.forward(x,params)

class Plan:
    def __init__(self,a,x,params):
        self.input=torch.empty_like(x)
        self.input.copy_(x)
        with torch.cuda.device(x.device):
            stream=torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):raw(a,self.input,params)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize(x.device)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=stream):self.outputs=raw(a,self.input,params)
        self.replays=0
        # Exact operation checks are outside primary timing, once per specialization.
        for scale in (1.,0.73):
            test=x*scale
            expected=raw(a,test,params)
            actual=self.run(test)
            assert all(torch.equal(u,v) for u,v in zip(expected,actual)),(a.key,x.shape,'projection capture mismatch')
        checks.append(dict(key=a.key,shape=list(x.shape),device=str(x.device),exact=True))

    def run(self,x):
        self.input.copy_(x)
        self.graph.replay();self.replays+=1
        return self.outputs

def projected(a,x,params,position,positions=None):
    if (not enabled or type(a) is not dsv4.DSV4Attention or a.q_head_norm
        or a.config.arch_string!='DeepseekV41ForCausalLM'
        or x.shape[0]!=1 or not 1<=x.shape[1]<=5 or x.dtype!=torch.half or not x.is_contiguous()
        or any(k in params for k in ('capture','quant_preserve','ovr','reconstruct','loras','backend'))
        or params.get('prefill') or params.get('autosplit_measure')):
        return original(a,x,params,position,positions)
    key=(id(a),tuple(x.shape),x.device,torch.cuda.current_stream(x.device).cuda_stream)
    plan=plans.get(key)
    if plan is None:plan=plans[key]=Plan(a,x,params)
    qr,q,kv=plan.run(x)
    b,s,_=x.shape;rd=a.rope_head_dim
    q=q.view(b,s,a.num_q_heads,a.head_dim);kv=kv.view(b,s,1,a.head_dim)
    dsv4._ext_rope(q[...,-rd:],a._rope_type(),position,positions=positions)
    dsv4.ext.rope(kv,kv,None,None,a._rope_type(),position,positions,None,
        int(dsv4.RopeStyle.GPTJ),1.0,a.kv_norm_w,None,a.rms_norm_eps,0.0,0.0,0,1,a.head_dim-rd)
    return qr,q,kv.view(b,s,a.head_dim)

def set_enabled(value):
    global enabled
    enabled=bool(value)
    dsv4.DSV4Attention._project_qkv=projected if enabled else original

def status():
    return dict(enabled=enabled,plans=len(plans),replays=sum(p.replays for p in plans.values()),checks=checks)
