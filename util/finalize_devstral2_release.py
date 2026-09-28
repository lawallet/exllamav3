"""Archive a validated candidate after all extracted-wheel gates passed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
ROOT = Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
p = argparse.ArgumentParser()
p.add_argument('--package',type=Path,required=True)
p.add_argument('--validation',type=Path,required=True)
p.add_argument('--tag',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
proof=json.loads((a.package/'package-proof.json').read_text())
assert json.loads((a.validation/'passed.json').read_text())['all_package_gates_passed']
runs=json.loads((a.validation/'runs.json').read_text())
assert len(runs)==8 and all(r['returncode']==0 for r in runs)
commit=subprocess.check_output(['git','rev-parse',a.tag+'^{commit}'],cwd=ROOT).decode().strip()
assert commit==proof['commit']
wheel=Path(proof['wheel'])
assert sha(wheel)==proof['wheel_sha256']
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
shutil.copy2(wheel,out/wheel.name)
for name in ('devstral2_ir_mapping.md','devstral2_ir_report.md','native_ir_v2_design.md','native_ir_v2_report.md','gemma4_ir_assessment.md'):
    shutil.copy2(ROOT/'doc'/name,out/name)
proof['status']='validated'
proof['extracted_validation']=str(a.validation.resolve())
(out/'package-proof.json').write_text(json.dumps(proof,indent=2))
shutil.copy2(a.package/'checkout-gates.json',out/'checkout-gates.json')
shutil.copy2(a.validation/'runs.json',out/'extracted-wheel-runs.json')
subprocess.run(['git','archive','--format=zip','--output='+str(out/'source.zip'),commit],cwd=ROOT,check=True)
subprocess.run(['git','bundle','create',str(out/'repository.bundle'),'--all'],cwd=ROOT,check=True)
subprocess.run(['git','bundle','verify',str(out/'repository.bundle')],cwd=ROOT,check=True)
metadata=dict(version='1.5.1+nativeplan.v2.devstral1',branch=proof['branch'],tag=a.tag,commit=commit,
    wheel=wheel.name,wheel_sha256=proof['wheel_sha256'],extension_sha256=proof['extension_sha256'],
    extracted_validation=str(a.validation.resolve()),evidence=str((ROOT/'native-results/devstral2').resolve()),
    installed=False,all_package_gates_passed=True)
performance=json.loads((ROOT/'native-results/devstral2/campaign/performance/performance.json').read_text())
performance_text='\n'.join(f"- B{r['batch']}: stock {r['median_tps']['False']:.4f} t/s, plan {r['median_tps']['True']:.4f} t/s; median paired change {100*r['median_change']:+.3f}%." for r in performance)
metadata['performance']=performance
started=(ROOT/'util/inspect_devstral2_ir.py').stat()
metadata['elapsed_minutes_since_first_inspection_helper']=(time.time()-getattr(started,'st_birthtime',started.st_ctime))/60
(out/'checkpoint.json').write_text(json.dumps(metadata,indent=2))
(out/'validation-summary.json').write_text(json.dumps(dict(
    devstral=dict(sites=528,short_shapes=['B1Q1','B2Q1','B3Q1','B4Q1','B1Q5','B2Q6'],
        short_steps=32,native_calls_per_stage_per_shape=2816,unexpected_decode_fallback=0,
        generator_cases=7,physical_page_reuse=True,reload_exact=True,
        visibility=['2,0,1','2,0','0,2'],context=8192,native_mtp=False,performance=performance),
    native_guard_tests=127,v1_ordinary=['Qwen','GLM','DeepSeek'],v1_mtp=['Qwen','GLM'],
    deepseek_mtp='Not rerun: unaffected v1 executor; prior protected validation retained',
    gemma_scope='actual-weight sites, short decode, cancellation/page reuse, topology',
    extracted_wheel=[dict(name=r['name'],passed=r['returncode']==0) for r in runs]),indent=2))
(out/'README.md').write_text(f'''# Validated Devstral 2 / native-plan v2 checkpoint

Version: `{metadata['version']}`. Branch: `{metadata['branch']}`.
Tag: `{a.tag}`. Commit: `{commit}`.

Wheel: `{wheel.name}`

Wheel SHA256: `{proof['wheel_sha256']}`

Native extension SHA256: `{proof['extension_sha256']}`

Runtime: Windows amd64, CPython 3.13, Torch 2.9.0+cu128, CUDA 12.8.
Uses the exact tested extension; packaging did not recompile or install it.
Select registered plans with `EXL3_NATIVE_PLAN=1`.

Supported checkpoint: `turboderp_Devstral-2-123B-Instruct-2512-exl3_4.00bpw`.
IR v2 adds generic RES_ADD_NORM and RES_ADD operations with separate capability
detection. V1 and existing Gemma instructions retain their contracts.
Tested Devstral shapes: B1/B2/B3/B4 Q1, B1 Q5, B2 Q6; Q7 falls back.
Tested visibility: 2,0,1 and 2,0 with identical placement/results, plus 0,2
with different placement and exact stock/plan parity. Long context: 8192.
No maximum-context, TP, LoRA, capture or external speculative certification.
Prefill remains stock; this checkpoint has no native MTP component.

Performance (B4 rates are aggregate):

{performance_text}

See devstral2_ir_report.md for all individual performance pairs, medians,
invocation counts, concurrency/page-reuse, reload and remaining limitations.
Qwen/GLM/DeepSeek ordinary regressions passed. Qwen/GLM natural/forced MTP
checks completed before the compatibility scope was reduced at user request.
DeepSeek MTP was not rerun because v1 execution is unchanged. Gemma's v2 sites,
short decode, cancellation/page reuse and topology checks passed; its broad
historical campaign was not repeated. Extracted-wheel gates cover Devstral sites/short/reload/
topology, Gemma sites/short, and ordinary DeepSeek/Qwen/GLM compatibility.
Each wrapper verified product and native imports came from the extracted wheel.

Restore source with `git clone repository.bundle restored` and check out
`{a.tag}`, or extract source.zip. SHA256SUMS.txt covers the release artifacts.
Large diagnostic evidence remains at `{metadata['evidence']}`; extracted-wheel
proof remains at `{metadata['extracted_validation']}`. No production package
was installed. The wheel is a validated local release artifact.

Approximate elapsed implementation/testing/packaging effort since the first
inspection helper: {metadata['elapsed_minutes_since_first_inspection_helper']:.1f} minutes.
Token/credit usage is not exposed by this session.
''')
(out/'SHA256SUMS.txt').write_text(''.join(f'{sha(f)}  {f.name}\n' for f in sorted(out.iterdir()) if f.is_file()))
print(json.dumps(metadata,indent=2))
