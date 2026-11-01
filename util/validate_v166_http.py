"""Run isolated real TabbyAPI HTTP smoke cohorts against the installed wheel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
TABBY = Path('D:/tabbyAPI')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stage = out/'harness'
    (stage/'util').mkdir(parents=True)
    for name in ('serve_nemotron_wheel_test.py', 'test_nemotron_tabby.py'):
        shutil.copy2(ROOT/'util'/name, stage/'util'/name)
    evidence = stage/'native-results/nemotron-wheel-test'
    runtime = evidence/'runtime'
    runtime.mkdir(parents=True)
    shutil.copy2(ROOT/'native-results/nemotron-moe-performance/long-cuda/final-tune.bin', runtime/'tune.bin')
    configs = [TABBY/'config.yml', TABBY/'api_tokens.yml']
    def config_hashes():
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in configs if p.exists()}
    before = config_hashes()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='2', EXL3_NATIVE_MOE_ASSIGN='1',
        PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1', EXLLAMAV3_TUNE_CACHE=str(runtime/'tune.bin'),
        TRITON_CACHE_DIR=str(runtime/'triton'))
    command = [sys.executable, '-I', str(stage/'util/serve_nemotron_wheel_test.py'),
        '--host','127.0.0.1','--port','5007','--model-name','Nemotron-3.5-Super-VL-4.23bpw',
        '--max-seq-len','131072','--cache-size','524288','--max-batch-size','4',
        '--gpu-split','92','--draft-gpu-split','92','--draft-mode','mtp','--draft-num-tokens','3',
        '--dynamic-draft','false','--vision','false','--log-live-status','false','--log-generation-params','false',
        '--use-as-default','draft_mode','draft_num_tokens','dynamic_draft','max_batch_size','chunk_size']
    # Credentials remain local and are never included in commands or evidence.
    key = yaml.safe_load((TABBY/'api_tokens.yml').read_text())['admin_key']
    with httpx.Client(base_url='http://127.0.0.1:5007', headers={'Authorization':'Bearer '+key},
                      timeout=5, trust_env=False) as client:
        try:
            client.get('/__validation/status')
        except httpx.ConnectError:
            pass
        else:
            raise RuntimeError('Port 5007 already has a server; leave it untouched')
        with (out/'server.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=TABBY, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            report = dict(pid=process.pid, command=command, config_hashes_before=before)
            try:
                deadline = time.monotonic()+600
                while True:
                    assert process.poll() is None, 'Server exited before readiness; see server.log'
                    try:
                        response = client.get('/__validation/status')
                        response.raise_for_status()
                        status = response.json()
                        if status['model_loaded']:
                            break
                    except (httpx.ConnectError, httpx.ReadTimeout):
                        pass
                    assert time.monotonic() < deadline, 'Server readiness timeout'
                    time.sleep(2)
                assert status['version'] == '1.6.6+nativeplan.v2.tp.moe1', status
                report['status_before'] = status
                print('SERVER READY', status['version'], flush=True)
                with (out/'client.log').open('w', encoding='utf-8') as client_log:
                    subprocess.run([sys.executable, '-I', str(stage/'util/test_nemotron_tabby.py'),
                        '--output','smoke','--contexts','0','--batches','1,2,3,4',
                        '--repeats','1','--tokens','128','--mode','audit'],
                        cwd=TABBY, env=env, stdout=client_log, stderr=subprocess.STDOUT, check=True)
                rows = [json.loads(p.read_text()) for p in (evidence/'smoke').glob('c*-b*-r*.json')]
                assert len(rows) == 8, len(rows)
                shadow = [s for row in rows for s in row['diagnostics']['shadow']]
                assert shadow and all(s['exact'] and s['default_stream'] for s in shadow)
                dispatched = {int(k) for row in rows for k in row['diagnostics']['dispatch']}
                assert {12,16} <= dispatched, dispatched
                assert all(r['tune_before'] == r['tune_after'] for r in rows)
                report.update(cohorts=len(rows), requests=sum(len(r['requests']) for r in rows),
                    tokens=sum(q['usage']['completion_tokens'] for r in rows for q in r['requests']),
                    same_input_oracles=len(shadow), observed_native_rows=sorted(dispatched), passed=True)
            finally:
                if process.poll() is None:
                    try:
                        client.post('/__validation/shutdown', json={}).raise_for_status()
                    except httpx.HTTPError as error:
                        report['shutdown_error'] = type(error).__name__
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait(timeout=30)
                        report['forced_shutdown'] = True
                report['server_exitcode'] = process.returncode
                report['config_hashes_after'] = config_hashes()
                (out/'result.json').write_text(json.dumps(report, indent=2))
                assert report['config_hashes_after'] == before, 'Persistent Tabby configuration changed'
    assert process.returncode == 0 and not report.get('forced_shutdown'), report
    print(json.dumps({k:v for k,v in report.items() if k not in ('command','status_before')}, indent=2))


if __name__ == '__main__':
    main()
