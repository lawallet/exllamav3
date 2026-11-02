"""Sequential, same-placement stock/native MTP trials through isolated TabbyAPI."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import httpx
import yaml

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budgets',type=float,nargs='+',default=[90,86,82])
    p.add_argument('--output',type=Path,default=ROOT/'native-results/deepseek-v41/splits')
    a=p.parse_args()
    key=yaml.safe_load(Path('D:/tabbyAPI/api_tokens.yml').read_text())['admin_key']
    client=httpx.Client(base_url='http://127.0.0.1:5008',headers={'Authorization':'Bearer '+key},timeout=10,trust_env=False)
    def run(script,*args):
        subprocess.run([sys.executable,'-I',str(ROOT/'util'/script),*map(str,args)],cwd=ROOT,check=True)
    def campaign(*args):run('deepseek_v41_campaign.py',*args)
    def summarize(folder):
        run('verify_deepseek_v41_cohorts.py',folder)
        run('summarize_deepseek_v41.py',folder)
    def stop():
        r=client.post('/__validation/shutdown',json={});r.raise_for_status()
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            with socket.socket() as probe:
                try:
                    probe.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
                    probe.bind(('127.0.0.1',5008));return
                except OSError:time.sleep(2)
        raise RuntimeError('Validation server did not stop; no new server launched')
    # Require the caller to stop the previous cohort server first.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        probe.bind(('127.0.0.1',5008))
    for budget in a.budgets:
        out=a.output/f'budget{budget:g}'
        assert not out.exists(),out
        campaign('start','--output',out,'--split',budget,46,0,'--package-root',ROOT,
                 '--extension-dir',ROOT/'native-results/deepseek-v41/package/build/lib.win-amd64-cpython-313')
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            try:
                r=client.get('/__validation/status')
                if r.status_code==200 and r.json()['model_loaded']:
                    (out/'loaded.json').write_text(json.dumps(r.json(),indent=2));break
            except httpx.HTTPError:pass
            log=out/'server.log'
            if log.exists() and 'RuntimeError: Insufficient VRAM' in log.read_text(errors='replace'):
                (out/'load-failure.json').write_text(json.dumps(dict(budget=budget,error='Insufficient VRAM in split'),indent=2))
                raise RuntimeError(f'{out}: split cannot fit model/cache; inspect server.log')
            time.sleep(5)
        else:
            raise RuntimeError(f'{out}: model not ready within 15 minutes; inspect server.log')
        print('READY',budget,flush=True)
        campaign('run','--output',out/'audit','--compare','--graphs','--audit','--mtp','on','--tokens',128,'--repeats',1)
        summarize(out/'audit')
        campaign('run','--output',out/'timing','--compare','--graphs','--mtp','on','--tokens',512,'--repeats',3,'--telemetry')
        summarize(out/'timing')
        stop()

if __name__=='__main__':main()
