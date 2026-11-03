"""Resolve noisy secondary GLM observations with warmed alternating controls."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/generation-pipeline'

def main():
    out=BASE/'mode-controls';out.mkdir(exist_ok=False)
    deadline=time.monotonic()+7200
    while not (BASE/'tp-controls/complete.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('TP controls incomplete')
        time.sleep(3)
    results=[]
    for case in ('glm','glm-tp'):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0' if case=='glm-tp' else '2,0,1',
            EXL3_NATIVE_PLAN='1',EXL3_NATIVE_GLM_BLOCK='1',EXL3_NATIVE_QWEN_BLOCK='1',
            EXL3_NATIVE_MOE_ASSIGN='1',EXL3_OMP_BLOCKTIME_MS='0',EXL3_NGRAM_STREAM='1',
            PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1')
        for k in ('EXL3_STOP_MASK_CACHE','EXL3_ASYNC_DRAFT_SEED'):env.pop(k,None)
        command=[sys.executable,'-I',str(ROOT/'util/benchmark_pipeline_models.py'),'--case',case,
            '--output',str(out/case),'--rounds','5','--mode-controls',
            '--package-dir',str(BASE/'stage-pipeline-v3'),'--settings','0','1']
        print('START '+case,flush=True)
        with (out/(case+'.log')).open('w') as log:
            r=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
        results.append(dict(case=case,command=command,exit_code=r.returncode))
        (out/'progress.json').write_text(json.dumps(results,indent=2))
        print('END '+case+' '+str(r.returncode),flush=True);r.check_returncode()
    (out/'complete.json').write_text(json.dumps(dict(success=True),indent=2))
if __name__=='__main__':main()
