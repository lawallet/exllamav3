"""Fresh import checks for independent opt-in flags and compatible defaults."""
import argparse,json,os,subprocess,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument('--package-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    code='''import json,sys
sys.path.insert(0,sys.argv[1])
from exllamav3.generator import stop_mask_cache
from exllamav3.util import draft_seed
print(json.dumps(dict(mask=stop_mask_cache.enabled,seed=draft_seed.enabled,mask_file=stop_mask_cache.__file__,seed_file=draft_seed.__file__)))
'''
    records=[]
    for mask,seed in ((None,None),('0','0'),('1','0'),('0','1'),('1','1')):
        env=dict(os.environ)
        for key,value in (('EXL3_STOP_MASK_CACHE',mask),('EXL3_ASYNC_DRAFT_SEED',seed)):
            env.pop(key,None)
            if value is not None:env[key]=value
        result=subprocess.run([sys.executable,'-I','-c',code,str(a.package_dir.resolve())],env=env,capture_output=True,text=True,check=True)
        actual=json.loads(result.stdout)
        assert actual['mask']==(mask=='1') and actual['seed']==(seed=='1'),actual
        assert Path(actual['mask_file']).is_relative_to(a.package_dir.resolve())
        records.append(dict(mask_env=mask,seed_env=seed,actual=actual))
    a.output.write_text(json.dumps(records,indent=2));print('Five fresh-process activation/default checks passed')
if __name__=='__main__':main()
