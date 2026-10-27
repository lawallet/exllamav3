"""Run the existing integration gates in isolated processes; never install a package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extension-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--only', nargs='+', required=True)
    p.add_argument('--model-root', type=Path, default=Path(r'D:\tabbyAPI\models'))
    p.add_argument('--test-deps', type=Path, default=ROOT/'native-results/v1.6.5-integration/test-deps')
    p.add_argument('--baseline-dir', type=Path, default=ROOT/'native-results/v1.6.5-integration/baseline')
    p.add_argument('--suite-paths', nargs='+', default=['tests'])
    p.add_argument('--native-plan', action='store_true', help='enable plan selection for legacy layer-split gates')
    p.add_argument('--devices', help='CUDA visibility order; defaults to CUDA_VISIBLE_DEVICES or largest GPUs first')
    a = p.parse_args()
    visible = a.devices or os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        devices = [value.strip() for value in visible.split(',')]
    else:
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total',
            '--format=csv,noheader,nounits'], text=True)
        entries = [tuple(int(value.strip()) for value in line.split(','))
                   for line in inventory.splitlines() if line.strip()]
        devices = [str(index) for index, memory in sorted(entries, key=lambda row: (-row[1], row[0]))]
    if len(devices) < 2:
        p.error('This integration campaign requires at least two visible GPUs')
    if any(name.endswith('-three') for name in a.only) and len(devices) < 3:
        p.error('Three-GPU model gates require three visible GPUs')
    all_devices, two_devices, single_device = ','.join(devices), ','.join(devices[:2]), devices[0]
    three_devices = ','.join(devices[:3])
    ext = a.extension_dir.resolve()
    binary, = ext.glob('exllamav3_ext*.pyd')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    def sha(f):
        with f.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    def sources():
        return {f.relative_to(ROOT).as_posix(): sha(f) for f in (ROOT/'exllamav3').rglob('*')
                if f.is_file() and f.suffix in ('.py', '.cpp', '.h', '.cu', '.cuh')}
    before = sources()
    proof = dict(source_hashes=before, extension=str(binary), extension_sha256=sha(binary),
                 python=sys.executable)
    (out/'inputs.json').write_text(json.dumps(proof, indent=2))
    jobs = {}
    def job(name, cmd, mapping=None):
        jobs[name] = (cmd, mapping or all_devices)
    job('guards', ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_native*.py', '-v'])
    job('dispatch', ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_decode_dispatch_profile.py', '-v'])
    job('suite', ['-m', 'pytest', *a.suite_paths, '-q', '--disable-warnings', '--maxfail=20', '--tb=short',
                  '--junitxml='+str(out/'suite.xml')])
    job('slow-suite', ['-m', 'pytest', *a.suite_paths, '--slow', '-m', 'slow', '-q',
                      '--disable-warnings', '--maxfail=20', '--tb=short',
                      '--junitxml='+str(out/'suite.xml')])
    for device in (0, 1):
        job(f'mgemm-{device}', ['tests/test_mgemm_variable_width.py', '--extension-dir', str(ext),
            '--device', f'cuda:{device}', '--output', str(out/f'mgemm-{device}.json')])
    for name, script in [('glm-sites','check_native_plan_primitives'),
                         ('deepseek-sites','check_deepseek_plan_sites'),
                         ('gemma-sites','check_gemma4_v2_sites')]:
        job(name, [f'util/{script}.py', '--extension-dir', str(ext), '--output', str(out/name)])
    models = dict(qwen='Qwen3.8-Flash-Next-Uncensored-exl3-4bpw',
                  glm='GLM-5.3-Flash-Uncensored-EXL3-3.05bpw',
                  deepseek='turboderp_DeepSeek-V4-Flash-0731-exl3_3.04bpw',
                  gemma='turboderp_gemma-4-31b-it-exl3_6.00bpw',
                  devstral='turboderp_Devstral-2-123B-Instruct-2512-exl3_4.00bpw')
    job('qwen-layer', ['util/profile_qwen_decode.py', '--model', str(a.model_root/models['qwen']),
        '--extension-dir', str(ext), '--output', str(out/'qwen-layer'),
        '--prompt-file', 'doc/native_qwen_prompt.txt', '--device', 'cuda:0',
        '--ab', '--compare-plan', '--tokens', '128', '--runs', '1'])
    job('glm-layer', ['util/profile_glm_decode.py', '--model', str(a.model_root/models['glm']),
        '--extension-dir', str(ext), '--output', str(out/'glm-layer'), '--split', '92.5', '44',
        '--smoke', '--ab', '--compare-plan', '--tokens', '32', '--runs', '1', '--profile-tokens', '4'], two_devices)
    job('deepseek-layer', ['util/validate_deepseek_plan.py', '--extension-dir', str(ext),
        '--output', str(out/'deepseek-layer'), '--steps', '8', '--reload'])
    for model, folder in models.items():
        for topology, mapping in [('two',two_devices), ('three',three_devices), ('single',single_device)]:
            name = f'{model}-{topology}'
            cmd = ['util/validate_native_tp.py', '--model', str(a.model_root/folder),
                   '--extension-dir', str(ext), '--output', str(out/name),
                   '--batches', '1', '2', '3', '4', '--steps', '8', '--runs', '3', '--generator']
            if topology == 'single':
                cmd += ['--single', '--single-parity']
                if model == 'qwen':
                    cmd += ['--allow-host-ngram-baseline']
            if model in ('qwen', 'glm', 'deepseek'):
                cmd += ['--mtp', '--generator-bench']
            if model == 'gemma':
                cmd += ['--reload']
            job(name, cmd, mapping)
        baseline = a.baseline_dir.resolve()
        name = f'{model}-baseline'
        cmd = [str(baseline/'util/validate_native_tp.py'), '--model', str(a.model_root/folder),
               '--extension-dir', str(baseline), '--output', str(out/name),
               '--batches', '1', '2', '3', '4', '--steps', '8', '--runs', '3']
        if model in ('qwen','glm','deepseek'):
            cmd += ['--mtp', '--generator-bench']
        job(name, cmd, two_devices)
    job('cold', ['util/diagnose_tp_cold_start.py', '--package', str(ext),
                 '--output', str(out/'cold'), '--cold'], two_devices)
    unknown = set(a.only)-jobs.keys()
    if unknown:
        p.error(f'Unknown gates: {unknown}')
    runs = []
    for name in a.only:
        cmd, mapping = jobs[name]
        print('START', name, flush=True)
        start = time.monotonic()
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT),str(ext),str(ROOT/'tests'),str(a.test_deps.resolve())]),
                   CUDA_VISIBLE_DEVICES=mapping, EXL3_NATIVE_PLAN='1' if a.native_plan else '0',
                   EXL3_NATIVE_QWEN_BLOCK='0', EXL3_NATIVE_GLM_BLOCK='0')
        with (out/(name+'.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        runs.append(dict(name=name, args=cmd, mapping=mapping, returncode=result.returncode,
                         seconds=time.monotonic()-start))
        (out/'runs.json').write_text(json.dumps(runs,indent=2))
        print('END', name, result.returncode, flush=True)
    assert sources() == before, 'Product sources changed during validation'
    assert sha(binary) == proof['extension_sha256'], 'Extension changed during validation'
    passed = all(r['returncode'] == 0 for r in runs)
    (out/('passed.json' if passed else 'failed.json')).write_text(json.dumps(dict(runs=runs,**proof),indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
