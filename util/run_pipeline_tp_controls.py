"""Limited matched old-wheel/product TP controls after the queued GPU campaign."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'native-results/deepseek-v41/generation-pipeline'

def main():
    out=BASE/'tp-controls';out.mkdir(exist_ok=False)
    deadline=time.monotonic()+5400
    while not (BASE/'tail-validation/complete.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('tail validation incomplete')
        time.sleep(3)
    results=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0',EXL3_NATIVE_PLAN='1',EXL3_NATIVE_GLM_BLOCK='1',
        EXL3_NATIVE_QWEN_BLOCK='1',EXL3_NATIVE_MOE_ASSIGN='1',EXL3_OMP_BLOCKTIME_MS='0',
        EXL3_NGRAM_STREAM='1',PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1')
    for k in ('EXL3_STOP_MASK_CACHE','EXL3_ASYNC_DRAFT_SEED'):env.pop(k,None)
    for name,extra in [('old-wheel',['--baseline-wheel','--settings','0']),
                       ('product',['--package-dir',str(BASE/'stage-pipeline-v3'),'--settings','0','1'])]:
        command=[sys.executable,'-I',str(ROOT/'util/benchmark_pipeline_models.py'),'--case','glm-tp',
            '--output',str(out/name),'--rounds','5','--tokens','256','--primary-only',*extra]
        print('START '+name,flush=True)
        with (out/(name+'.log')).open('w') as log:
            r=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
        results.append(dict(name=name,command=command,exit_code=r.returncode))
        (out/'progress.json').write_text(json.dumps(results,indent=2))
        print('END '+name+' '+str(r.returncode),flush=True);r.check_returncode()
    (out/'complete.json').write_text(json.dumps(dict(success=True),indent=2))
if __name__=='__main__':main()
