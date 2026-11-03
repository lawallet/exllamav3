"""Offline full-token/MTP, placement, memory and workload gates for completed cohorts."""
import argparse,json,re,statistics
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/generation-pipeline'
def stats(xs):
    return dict(n=len(xs),median=statistics.median(xs),minimum=min(xs),maximum=max(xs),
        stdev=statistics.stdev(xs) if len(xs)>1 else None,variance=statistics.variance(xs) if len(xs)>1 else None)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    refs={};files=[];groups=defaultdict(lambda:dict(decode=[],prefill=[]));cohorts=[]
    for folder in sorted(BASE.iterdir()):
        if not folder.is_dir() or not (folder/'final-status.json').exists():continue
        if not (folder/'server.json').exists():continue
        command=json.loads((folder/'server.json').read_text())['command']
        for option,values in (('--gpu-split',[76,46,0]),('--draft-gpu-split',[90,0,0])):
            start=command.index(option)+1;assert [float(v) for v in command[start:start+3]]==values
        cohorts.append(folder.name)
        for path in sorted(folder.glob('*.json')):
            if any(x in path.name for x in ('-trace','-host','summary')):continue
            d=json.loads(path.read_text())
            if 'result' not in d or 'response' not in d:continue
            j,=d['result']['state']['jobs'];s=d['result']['status'];extra=d['extra_before']
            assert extra['blocktime']['milliseconds']==0 and extra['torch_threads']==56,path
            assert j['cached_tokens']==0 and j['new_tokens']==d['tokens'],path
            assert j['prompt_tokens']=={'short':177,'8k':8192,'65k':64511}[d['context']],path
            assert len(j['ids'][0])==j['prompt_tokens']+j['new_tokens'],path
            blocks=Counter(row['device'] for row in s['placement'] if re.fullmatch(r'layers\.\d+',row['key']))
            assert blocks=={'cuda:0':26,'cuda:1':14},(path,blocks)
            assert s['generator']['cache_tokens']==65536
            assert s['generator']['drafts']==(3 if d['mtp'] else 0)
            assert len(s['ngrams'])==2 and all(n['mode']=='fp16_disk' for n in s['ngrams'])
            for before,after in zip(extra['allocator'],d['extra_after']['allocator']):
                assert before['num_ooms']==after['num_ooms'],(path,'runtime OOM')
            key=(d['context'],d['tokens'],d['mtp'],d.get('ir',True),min(d['minimum'],d['tokens']))
            signature=(j['ids'],j['accepted_draft_tokens'],j['rejected_draft_tokens'])
            assert key not in refs or refs[key]==signature,(path,'token/MTP parity')
            refs.setdefault(key,signature);files.append(str(path.relative_to(ROOT)))
            if not d['name'].startswith('steady'):continue
            label=str((folder.name,*key,d.get('serial_readback',False),d.get('candidate',False),d.get('async_seed',False)))
            groups[label]['decode'].append(j['new_tokens']/j['time_generate'])
            groups[label]['prefill'].append(j['time_prefill'])
    # Match the already validated .omp1 long-context release evidence as well.
    old_checks=[]
    for context in ('8k','65k'):
        old=ROOT/f'native-results/deepseek-v41/omp-policy/installed-deepseek/{context}-steady.json'
        if old.exists() and (context,512,True,True,512) in refs:
            j=json.loads(old.read_text())['result']['state']['jobs'][0]
            assert refs[(context,512,True,True,512)]==(j['ids'],j['accepted_draft_tokens'],j['rejected_draft_tokens'])
            old_checks.append(str(old.relative_to(ROOT)))
    result=dict(completed_cohorts=cohorts,parity_requests=len(files),workload_cohorts=len(refs),
        all_tokens_and_mtp_exact=True,placement_26_14=True,torch_threads=56,blocktime_ms=0,
        context_capacity=65536,ssd_engram_unchanged=True,no_request_ooms=True,old_release_checks=old_checks,
        groups={k:{name:stats(xs) for name,xs in v.items()} for k,v in groups.items()},files=files)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','files')},indent=2))
if __name__=='__main__':main()
