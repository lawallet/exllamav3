"""Sequential cross-model validation using the established placements."""
import argparse,json,os,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--package-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',nargs='+',default=['qwen','glm','glm-tp','nemotron','deepseek4','gemma','devstral'])
    p.add_argument('--after',type=Path,help='Wait for this isolated server cohort to finish successfully')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if a.after:
        import psutil
        pid=json.loads((a.after/'server.json').read_text())['pid']
        print('Waiting for server cohort '+str(a.after),flush=True)
        try:
            process=psutil.Process(pid)
            while process.is_running():
                try:process.wait(timeout=5)
                except psutil.TimeoutExpired:continue
                break
        except psutil.NoSuchProcess:pass
        assert (a.after/'final-status.json').exists(),'preceding cohort did not finish successfully'
    outcomes=[]
    for case in a.cases:
        out=a.output/case
        cmd=[sys.executable,'-I',str(ROOT/'util/benchmark_pipeline_models.py'),'--case',case,
             '--output',str(out),'--package-dir',str(a.package_dir.resolve()),'--settings','0','1',
             '--rounds','3' if case in ('gemma','devstral') else '5',
             '--tokens','128' if case in ('gemma','devstral') else '256']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,0' if case=='glm-tp' else '2,0,1',EXL3_NATIVE_PLAN='1',
                 EXL3_OMP_BLOCKTIME_MS='0',EXL3_NATIVE_MOE_ASSIGN='1',EXL3_NGRAM_STREAM='1',PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1')
        print('START '+case,flush=True);t=time.time()
        with (a.output/(case+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        entry=dict(case=case,command=cmd,exit_code=result.returncode,seconds=time.time()-t)
        if result.returncode==0:
            groups={}
            for enabled in (False,True):
                rows=[json.loads(p.read_text()) for p in out.glob(f'steady-*-cache{int(enabled)}.json')]
                values=[r['jobs'][0]['new_tokens']/r['jobs'][0]['time_generate'] for r in rows]
                groups[str(enabled)]=dict(n=len(values),median=statistics.median(values),minimum=min(values),maximum=max(values),
                    stdev=statistics.stdev(values),values=values)
            entry['primary']=groups
            entry['gain_pct']=100*(groups['True']['median']/groups['False']['median']-1)
            with (ROOT/'doc/deepseek_v41_ir_investigation.md').open('a',encoding='utf-8') as report:
                report.write(f"\nCross-model `{case}` completed: disabled/enabled masked decode medians "
                    f"{groups['False']['median']:.3f}/{groups['True']['median']:.3f} tok/s "
                    f"({entry['gain_pct']:+.2f}%). Full token/MTP, EOS-boundary, seeded sampling, "
                    f"ordinary/stock, concurrent and cancellation checks passed. Evidence: `{out.relative_to(ROOT).as_posix()}`. "
                    "This is a per-process result; uncertainty and final recommendations are assessed separately.\n")
        outcomes.append(entry);(a.output/'progress.json').write_text(json.dumps(outcomes,indent=2))
        print('END '+json.dumps(entry),flush=True)
        if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
