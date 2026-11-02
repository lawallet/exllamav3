"""Consolidate installed release evidence and verify temporary processes exited."""
from collections import Counter
import json
from pathlib import Path
import re
import socket
import psutil

BASE=Path(__file__).resolve().parents[1]/'native-results/deepseek-v41/omp-policy'


def read(name):return json.loads((BASE/name).read_text(encoding='utf-8-sig'))


def main():
    proof=read('package/installed-proof.json');package=read('package/proof.json')
    assert proof['payload_exact'] and proof['rollback_preserved']
    parity=read('parity-installed-final.json');assert parity['exact']
    runs=read('installed-validation-runs.json');assert len(runs)==2 and all(r['code']==0 for r in runs)
    for folder,expected in (('installed-deepseek',0),('installed-deepseek-unset',200)):
        state=read(folder+'/final-status.json');s=state['status'];extra=state['extra']
        assert s['version']==package['version'] and 'site-packages' in s['package'] and 'site-packages' in s['extension']
        assert extra['torch_threads']==56 and extra['blocktime']['milliseconds']==expected
        policy=extra['implemented_policy']
        assert policy['actual_ms']==expected
        assert policy['requested_ms']==(0 if expected==0 else None)
        assert policy['applied_on_thread']==(expected==0)
        assert Counter(m['device'] for m in s['placement'] if re.fullmatch(r'layers\.\d+',m['key']))=={'cuda:0':26,'cuda:1':14}
        assert s['generator']==dict(batch=1,drafts=3,dynamic=False,cache_tokens=65536)
        assert s['ngrams'] and all(n['mode']=='fp16_disk' for n in s['ngrams'])
        assert s['graph_replays']>0
    summary=read('summary.json');rates={}
    for folder,ms in (('old-release-final-control',200),('installed-deepseek',0),('installed-deepseek-unset',200)):
        rates[folder]={mode:summary[folder]['groups'][f'steady/short/b1/native{flag}/mtp1/ms{ms}']['tps']
                       for mode,flag in (('stock',0),('native',1))}
    names=('benchmark_omp_models.py','serve_deepseek_v41_followup.py','deepseek_v41_telemetry.py')
    active=[p.pid for p in psutil.process_iter() if p.name().lower().startswith('python') and any(arg.endswith(names) for arg in p.cmdline())]
    assert not active,active
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        probe.bind(('127.0.0.1',5008))
    result=dict(version=package['version'],implementation_commit=package['source_commit'],wheel=package['wheel'],
        wheel_sha256=package['wheel_sha256'],extension_sha256=package['extension_sha256'],rates=rates,
        native_gain_vs_old_default_percent=100*(rates['installed-deepseek']['native']['median']/rates['old-release-final-control']['native']['median']-1),
        native_gain_vs_new_unset_percent=100*(rates['installed-deepseek']['native']['median']/rates['installed-deepseek-unset']['native']['median']-1),
        exact_requests=sum(parity['requests_by_folder'].values()),exact_cohorts=parity['cohorts'],
        installed_native_cpu_core_equivalents=read('installed-deepseek/telemetry-summary.json')['monitored-r0-ir1.json']['phases']['decode']['server_cpu_core_equivalents'],
        temporary_servers_and_monitors_exited=True,validation_port_free=True,preservation=proof)
    (BASE/'final-release-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
