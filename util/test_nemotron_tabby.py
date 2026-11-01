"""Repeatable real HTTP cohorts; run with Tabby Python -I (installed wheel)."""
import argparse, asyncio, hashlib, json, time
from pathlib import Path
import httpx, yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'native-results/nemotron-wheel-test'
PROMPTS=[
 'Write a Python implementation of a bounded LRU cache with get and put methods. Explain how eviction works and include unit tests for updating a key and exceeding capacity.',
 'Implement a Python topological sort for a directed graph. Detect cycles and include tests for disconnected vertices and a cycle.',
 'Write a Python function that merges overlapping half-open intervals. Explain the edge cases and provide unit tests.',
 'Implement a small Python retry decorator with exponential backoff, injectable sleep, and tests for success and exhausted retries.',
]

def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')

async def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',default='smoke');p.add_argument('--contexts',default='0')
 p.add_argument('--batches',default='1,2,3,4');p.add_argument('--repeats',type=int,default=1)
 p.add_argument('--tokens',type=int,default=192);p.add_argument('--mode',choices=['audit','measure','off'],default='measure')
 p.add_argument('--identical',action='store_true');p.add_argument('--reset-cache',action='store_true')
 p.add_argument('--draft-cap',type=int,default=0);p.add_argument('--ir-only',action='store_true')
 p.add_argument('--admin');p.add_argument('--body',default='{}');p.add_argument('--body-file',type=Path)
 a=p.parse_args();dest=OUT/a.output;dest.mkdir(parents=True,exist_ok=True)
 # Read existing credentials locally. Never serialize credentials or headers.
 key=yaml.safe_load(Path('D:/tabbyAPI/api_tokens.yml').read_text())['admin_key']
 async with httpx.AsyncClient(base_url='http://127.0.0.1:5007',headers={'Authorization':'Bearer '+key},timeout=7200,trust_env=False) as client:
  async def post(path,body):
   r=await client.post(path,json=body);r.raise_for_status();return r.json()
  if a.admin:
   body=json.loads(a.body_file.read_text() if a.body_file else a.body)
   r=await (client.get(a.admin) if a.body=='{}' and a.admin.endswith(('status','results')) else client.post(a.admin,json=body))
   r.raise_for_status()
   try:data=r.json()
   except ValueError:data={'sse':r.text}
   save(dest/'admin.json',data);print(str(data)[:2500],flush=True);return
  import torch,exllamav3
  from exllamav3 import Config,Tokenizer
  assert Path(exllamav3.__file__).is_relative_to(Path('D:/tabbyAPI/venv'))
  tok=Tokenizer.from_config(Config.from_directory('D:/tabbyAPI/models/Nemotron-3.5-Super-VL-4.23bpw'))
  def prompt(context,i):
   tail='<|im_start|>user\n'+PROMPTS[i]+'\n<|im_end|>\n<|im_start|>assistant\n'
   if not context:return tail
   head=tok.encode('<|im_start|>user\nRepository context for agent '+str(i)+':\n```python\n',encode_special_tokens=True)
   body=tok.encode('\n'.join(f'def helper_{i}_{j}(items):\n    """Return stable unique values for task {i}."""\n    return list(dict.fromkeys(items))\n' for j in range(500)))
   end=tok.encode('\n```\nUsing this context, '+tail.split('user\n')[1],encode_special_tokens=True)
   n=context-head.shape[1]-end.shape[1]
   ids=torch.cat((head,body.repeat(1,(n+body.shape[1]-1)//body.shape[1])[:,:n],end),dim=1)
   text=tok.decode(ids,decode_special_tokens=True)[0]
   actual=tok.encode(text,encode_special_tokens=True)
   # Text API roundtrip may fuse the artificial cut. Record actual length/hash.
   return text
  async def request(text,i):
   start=time.perf_counter();events=[];pieces=[];usage=None;finish=None
   body=dict(model='Nemotron-3.5-Super-VL-4.23bpw',prompt=text,stream=True,stream_options={'include_usage':True},
     max_tokens=a.tokens,min_tokens=a.tokens,temperature=0,top_k=1,top_p=1,min_p=0,
     repetition_penalty=1,frequency_penalty=0,presence_penalty=0,add_bos_token=False,token_healing=False)
   async with client.stream('POST','/v1/completions',json=body) as r:
    if r.status_code!=200:raise RuntimeError((r.status_code,(await r.aread()).decode()))
    async for line in r.aiter_lines():
     if not line.startswith('data: '):continue
     payload=line[6:]
     if payload=='[DONE]':continue
     event=json.loads(payload)
     if 'error' in event:raise RuntimeError(event)
     if event.get('usage'):usage=event['usage']
     for choice in event.get('choices',[]):
      if choice.get('text'):events.append(time.perf_counter()-start);pieces.append(choice['text'])
      if choice.get('finish_reason'):finish=choice['finish_reason']
   text=''.join(pieces)
   assert usage and usage['completion_tokens']==a.tokens,(usage,finish,text[-100:])
   return dict(index=i,start=start,wall=time.perf_counter()-start,arrivals=events,text=text,
     sha256=hashlib.sha256(text.encode()).hexdigest(),usage=usage,finish=finish)
  for context in map(int,a.contexts.split(',')):
   prompts=[prompt(context,i if not a.identical else 0) for i in range(4)]
   facts=[dict(length=tok.encode(t,encode_special_tokens=True).shape[-1],sha256=hashlib.sha256(t.encode()).hexdigest()) for t in prompts]
   save(dest/f'prompts-c{context}.json',dict(texts=prompts,facts=facts,sampling='greedy; no seed exposed by this Tabby version'))
   for batch in map(int,a.batches.split(',')):
    for repeat in range(a.repeats):
     for ir in ([True] if a.ir_only else ([False,True] if repeat%2==0 else [True,False])):
      name=f'c{context}-b{batch}-r{repeat}-'+('ir' if ir else 'stock')
      assert not (dest/(name+'.json')).exists(),name
      control=dict(name=name,mode=a.mode,ir=ir,reset_cache=a.reset_cache,draft_cap=a.draft_cap)
      status=await post('/__validation/control',control)
      tune=OUT/'runtime/tune.bin';tune_before=hashlib.sha256(tune.read_bytes()).hexdigest()
      print('START',name,'contexts',[f['length'] for f in facts[:batch]],flush=True)
      begin=time.perf_counter();requests=await asyncio.gather(*(request(t,i) for i,t in enumerate(prompts[:batch])))
      wall=time.perf_counter()-begin
      r=await client.get('/__validation/results');r.raise_for_status();diagnostics=r.json()
      save(dest/(name+'.json'),dict(name=name,context=context,batch=batch,repeat=repeat,ir=ir,identical=a.identical,
        control=control,status_before=status,prompts=facts[:batch],requests=requests,wall=wall,diagnostics=diagnostics,
        tune_before=tune_before,tune_after=hashlib.sha256(tune.read_bytes()).hexdigest()))
      shapes={str(f['shape']) for f in diagnostics['forwards']}
      print('DONE',name,'wall',round(wall,2),'shapes',sorted(shapes),'dispatch',diagnostics['dispatch'],flush=True)

if __name__=='__main__':asyncio.run(main())
