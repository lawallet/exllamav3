"""Freeze source, gate packaging, and archive a fully validated Laguna wheel."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
V = ROOT/'native-results/laguna'
EXT = ROOT/'native-results/devstral2/extension'
BASE = '3e83530ff2cecea8e920fd352360715c38a33529'
p = argparse.ArgumentParser()
p.add_argument('phase',choices=['freeze','gates','finalize'])
p.add_argument('--campaign',default='campaign-r2')
p.add_argument('--package',default='package-v1')
p.add_argument('--wheel-validation',default='wheel-validation')
p.add_argument('--output',default='release-v1')
p.add_argument('--tag',default='native-laguna-ir-v2-v1')
a = p.parse_args()
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,data): path.write_text(json.dumps(data,indent=2))
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT).decode().strip()
def sources(): return {f.relative_to(ROOT).as_posix():sha(f) for f in (ROOT/'exllamav3').rglob('*')
                      if f.is_file() and f.suffix in ('.py','.cpp','.h','.cu','.cuh')}
extension, = EXT.glob('exllamav3_ext*.pyd')
assert sha(extension)=='b89a859f85765405f77e517f6faed02456cd23a46e48226258ba27c5dd1d6fd3'
assert not git('diff',BASE,'--','exllamav3/exllamav3_ext','exllamav3/modules/native_plan.py')
if a.phase=='freeze':
    assert not (V/'validated-source.json').exists()
    write(V/'validated-source.json',sources())
    print('Frozen product source and unchanged native binary')
elif a.phase=='gates':
    assert read(V/'validated-source.json')==sources(), 'Product changed during validation'
    runs=read(V/a.campaign/'runs.json')
    required={'guards','sites','short-201','generator','reuse','short-20','remapped','long','performance','gemma-smoke','deepseek-smoke'}
    assert {r['name'] for r in runs}==required and all(r['returncode']==0 for r in runs)
    assert 'Ran 135 tests' in (V/a.campaign/'guards.log').read_text()
    sites=read(V/a.campaign/'sites/sites.json')['records']
    assert len(sites)==288 and all(r['exact'] and r['rejected_safely'] for r in sites)
    for name,steps in [('short-201',32),('short-20',32),('remapped',16),('long',16)]:
        for b,q in ([(1,1),(1,5)] if name=='long' else [(1,1),(2,1),(3,1),(4,1),(1,5),(2,6),(1,7)]):
            stock=read(V/a.campaign/name/f'b{b}-q{q}-0.json')
            plan=read(V/a.campaign/name/f'b{b}-q{q}-1.json')
            assert stock['records']==plan['records'] and stock['boundaries']==plan['boundaries']
            assert plan['counts'].get('begin',0)==(48*steps if q<=6 else 0)
            assert plan['counts']['python_route']==47*steps
            assert plan['counts'].get('python_moe_native_calls',0)==(47*steps if b*q<=8 else 0)
    for name in ('short-201','short-20'): assert read(V/a.campaign/name/'reload.json')['exact']
    assert read(V/a.campaign/'short-20/topology.json')['exact']
    for i in (4,5):
        r=read(V/a.campaign/f'reuse/case{i}-1.json')
        assert r['reuse']['pages']>0 and r['reuse']['slots']>0
    assert all(r['median_change']>=-.03 for r in read(V/a.campaign/'performance/performance.json'))
    # The diagnostic deliberately trips the existing -3% investigation alarm.
    # Preserve that failure; promotion requires its completed, exact-token
    # measurements plus a successful independent placement control and disclosure.
    repeat=read(V/'performance-repeat/run.json')
    assert repeat['returncode']==1
    assert 'AssertionError: Repeatable regression requires investigation' in (V/'performance-repeat/run.log').read_text()
    assert len(read(V/'performance-repeat/results/samples.json'))==40
    assert len(read(V/'performance-repeat/results/performance.json'))==2
    control=read(V/'performance-control/runs.json')
    assert len(control)==2 and all(r['returncode']==0 for r in control)
    assert all(r['median_change']>=-.03 for r in read(V/'performance-control/performance/performance.json'))
    assert 'two-GPU B4 regression remains a measured limitation' in (ROOT/'doc/laguna_ir_report.md').read_text()
    assert not git('diff','HEAD','--','exllamav3','doc','tests','util')
    write(V/'checkout-gates.json',dict(all_required_checkout_gates_passed=True,
        commit=git('rev-parse','HEAD'),extension_sha256=sha(extension),source_hashes=sources(),
        evidence=str(V/a.campaign),scope='Full Laguna; guard suite; Gemma v2 and DeepSeek v1 short smokes; no shared IR change',
        performance_investigation='Two-GPU B4 -3.12% alarm preserved; single-GPU control passes; no speedup claim; measured limitation disclosed'))
    print('PASS all checkout gates and source/binary identity')
else:
    package=V/a.package; validation=V/a.wheel_validation
    proof=read(package/'package-proof.json')
    assert read(validation/'passed.json')['all_selected_gates_passed']
    runs=read(validation/'runs.json')
    assert {r['name'] for r in runs}=={'sites','short-201','short-20'} and all(r['returncode']==0 for r in runs)
    assert git('rev-parse',a.tag+'^{commit}')==proof['commit']==git('rev-parse','HEAD')
    assert read(V/'validated-source.json')==sources()
    out=V/a.output; out.mkdir(exist_ok=False)
    wheel=Path(proof['wheel']); assert sha(wheel)==proof['wheel_sha256']
    shutil.copy2(wheel,out/wheel.name)
    for name in ('laguna_ir_report.md','laguna_ir_mapping.md'):
        shutil.copy2(ROOT/'doc'/name,out/name)
    for source,name in [(package/'checkout-gates.json','checkout-gates.json'),
                        (validation/'runs.json','extracted-wheel-runs.json'),
                        (V/a.campaign/'runs.json','validation-summary.json')]: shutil.copy2(source,out/name)
    proof['status']='validated'; write(out/'package-proof.json',proof)
    subprocess.run(['git','archive','--format=zip','--output='+str(out/'source.zip'),proof['commit']],cwd=ROOT,check=True)
    subprocess.run(['git','bundle','create',str(out/'repository.bundle'),'--all'],cwd=ROOT,check=True)
    with (out/'bundle-verify.txt').open('w') as log:
        subprocess.run(['git','bundle','verify',str(out/'repository.bundle')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    metadata=dict(version='1.5.1+nativeplan.v2.laguna1',branch=proof['branch'],tag=a.tag,commit=proof['commit'],
        wheel=wheel.name,wheel_sha256=proof['wheel_sha256'],extension_sha256=proof['extension_sha256'],
        installed=False,evidence=str(V),extracted_validation=str(validation),
        elapsed_minutes=(time.time()-(V/'inspection.log').stat().st_ctime)/60)
    metadata['performance']=read(V/'performance-repeat/results/performance.json')
    metadata['initial_performance']=read(V/a.campaign/'performance/performance.json')
    metadata['single_gpu_control']=read(V/'performance-control/performance/performance.json')
    performance_text='\n'.join(f"B{r['batch']}: stock {r['median_tps']['False']:.4f} t/s; plan {r['median_tps']['True']:.4f} t/s; paired change {100*r['median_change']:+.3f}%." for r in metadata['performance'])
    write(out/'checkpoint.json',metadata)
    (out/'README.md').write_text(f'''# Validated Laguna-S-2.1 IR v2 checkpoint

Version: {metadata['version']}
Branch: {metadata['branch']}
Tag: {a.tag}
Commit: {proof['commit']}
Wheel: {wheel.name}
Wheel SHA256: {proof['wheel_sha256']}
Native extension SHA256: {proof['extension_sha256']}

Runtime: Windows amd64, CPython3.13, Torch2.9.0+cu128, CUDA12.8.
Select with EXL3_NATIVE_PLAN=1. Checkpoint: turboderp_Laguna-S-2.1-exl3_4.00bpw.
Uses unchanged IR v2 RMS_NORM, RES_ADD_NORM, RES_ADD; no new capabilities.
Routing/MoE remains the stock Python boundary around native routing/fused experts.
Tested B1/B2/B3/B4 Q1, B1 Q5, B2 Q6; Q7 falls back. Visibility 2,0,1 and 2,0,
plus different split under 0,2. Long context8192. SWA/paged cache reuse and reload
passed. See laguna_ir_report.md for all performance samples and exact coverage.
{performance_text}
These final rates use ten longer pairs and fixed benchmark-only CPU affinity0xff.
Unrestricted initial samples were unstable and are preserved in the report/evidence.
The extended two-GPU B4 result (-3.12%) tripped the investigation alarm. It remains
a measured limitation; separate single-GPU parity/performance controls passed.
No material speedup is claimed. See the report for every sample and both placements.

Compatibility: guard suite, Gemma v2 and DeepSeek v1 short smokes. No historical
MTP campaigns: generic execution, common dispatch and kernels are unchanged.
Extracted wheel: Laguna sites, short decode, reload and equivalent topology;
all product/native imports verified from extraction. Source matches checkpoint.
No production installation. The exact previously tested native binary is reused.
No claim of bit-reproducible recompilation; source, binary and packaging inputs
are recorded. SHA256SUMS.txt covers artifacts. Restore using
git clone repository.bundle restored, then check out {a.tag}, or extract source.zip.

Unsupported/unverified: prefill remains stock; TP, LoRA, capture, CPU offload,
external DFlash, maximum advertised context. Target has no native MTP/default drafter.
Evidence: {V}; wheel proof: {validation}.
Elapsed approximately {metadata['elapsed_minutes']:.1f} minutes; credit usage unavailable.
''')
    (out/'SHA256SUMS.txt').write_text(''.join(f'{sha(f)}  {f.name}\n' for f in sorted(out.iterdir()) if f.is_file()))
    print(json.dumps(metadata,indent=2))
