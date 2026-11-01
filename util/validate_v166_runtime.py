"""Run established model oracles against an extracted or installed package.

Only harnesses are staged: no source package can shadow the selected runtime.
All commands, exit codes, package origins and payload hashes are recorded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', nargs='+', required=True)
    parser.add_argument('--version', default='1.6.6+nativeplan.v2.tp.moe1')
    parser.add_argument('--glm-single-reason',
        help='specific shared-IR check or regression requiring GLM on one GPU; never a performance target')
    a = parser.parse_args()
    requested = list(a.only)
    skipped = {}
    if 'glm-single' in a.only and not a.glm_single_reason:
        skipped['glm-single'] = 'GLM deployment is TP2+MTP; no shared-IR regression requires a single-GPU run'
        a.only = [name for name in a.only if name != 'glm-single']
        print('SKIP glm-single:', skipped['glm-single'], flush=True)
    if not a.only:
        parser.error('No applicable gates remain')
    package = a.package.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'selection.json').write_text(json.dumps(dict(requested=requested, selected=a.only,
        skipped=skipped, glm_single_reason=a.glm_single_reason), indent=2))
    stage = out/'harness'
    for folder in ('util', 'tests'):
        shutil.copytree(ROOT/folder, stage/folder,
                        ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
    (stage/'doc').mkdir()
    for p in (ROOT/'doc').glob('*prompt.txt'):
        shutil.copy2(p, stage/'doc'/p.name)
    shutil.copy2(ROOT/'pyproject.toml', stage/'pyproject.toml')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='2,0,1', EXL3_NATIVE_PLAN='0',
               EXL3_NATIVE_QWEN_BLOCK='0', EXL3_NATIVE_GLM_BLOCK='0', EXL3_NATIVE_MOE_ASSIGN='0')
    env['PYTHONPATH'] = os.pathsep.join(map(str, [package, stage/'util', stage/'tests',
        ROOT/'native-results/v1.6.5-integration/test-deps']))
    def payload():
        files = list((package/'exllamav3').rglob('*.py')) + list(package.glob('exllamav3_ext*.pyd'))
        return {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    before = payload()
    probe = subprocess.run([sys.executable, '-c',
        "import json,exllamav3,torch; from exllamav3.ext import exllamav3_ext as e; "
        "from exllamav3.version import __version__ as v; "
        "print(json.dumps(dict(package=exllamav3.__file__,extension=e.__file__,version=v,torch=torch.__version__)))"],
        cwd=stage, env=env, capture_output=True, text=True, check=True)
    origins = json.loads(probe.stdout)
    assert Path(origins['package']).is_relative_to(package), origins
    assert Path(origins['extension']).parent == package, origins
    assert origins['version'] == a.version, origins
    (out/'inputs.json').write_text(json.dumps(dict(origins=origins, hashes=before), indent=2))
    models = dict(glm='GLM-5.3-Flash-Uncensored-EXL3-3.05bpw',
        qwen='Qwen3.8-Flash-Next-Uncensored-exl3-4bpw',
        deepseek='turboderp_DeepSeek-V4-Flash-0731-exl3_3.04bpw',
        gemma='turboderp_gemma-4-31b-it-exl3_6.00bpw',
        devstral='turboderp_Devstral-2-123B-Instruct-2512-exl3_4.00bpw')
    jobs = {}
    for model, folder in models.items():
        for topology, devices in [('single','2'), ('two','2,0'), ('three','2,0,1')]:
            name = f'{model}-{topology}'
            script = 'validate_native_graphs' if model == 'glm' and topology != 'single' else 'validate_native_tp'
            cmd = [f'util/{script}.py', '--model', str(Path('D:/tabbyAPI/models')/folder),
                   '--extension-dir', str(package), '--output', str(out/name),
                   '--batches', '1', '2', '3', '4', '--steps', '8', '--runs', '3', '--generator']
            if topology == 'single':
                cmd += ['--single', '--single-parity']
                if model == 'glm': cmd[cmd.index('--runs')+1] = '0'
                if model == 'qwen': cmd += ['--allow-host-ngram-baseline']
            if model in ('qwen','glm','deepseek'): cmd += ['--mtp']
            if model == 'gemma' or (model == 'glm' and topology == 'two'): cmd += ['--reload']
            if model == 'qwen' and topology == 'two': cmd += ['--generator-bench']
            if model == 'deepseek' and topology == 'three':
                cmd += ['--triton-cache-root', str(out/name/'triton')]
            jobs[name] = (cmd, devices)
    qwen_repeat = list(jobs['qwen-two'][0])
    qwen_repeat[qwen_repeat.index('--output')+1] = str(out/'qwen-repeat')
    qwen_repeat[qwen_repeat.index('--runs')+1] = '0'
    qwen_repeat += ['--generator-bench-runs', '5']
    jobs['qwen-repeat'] = (qwen_repeat, '2,0')
    jobs['glm-bench'] = (['util/profile_final_ir.py', '--model', str(Path('D:/tabbyAPI/models')/models['glm']),
        '--extension-dir', str(package), '--package-root', str(package),
        '--output', str(out/'glm-bench'), '--modes', 'native', 'graphs',
        '--mtp', '--lengths', '128', '1024', '--runs', '3', '--warm-longest'], '2,0')
    for phase in ('validate','bench'):
        name = 'nemotron-'+phase
        cmd = ['util/validate_native_moe.py', '--model', 'D:/tabbyAPI/models/Nemotron-3.5-Super-VL-4.23bpw',
            '--output', str(out/name), '--phase', phase,
            '--tune-from', str(ROOT/'native-results/nemotron-moe-performance/long-cuda/final-tune.bin'),
            '--repeats', '1' if phase == 'validate' else '3', '--tokens', '128']
        if phase == 'validate': cmd += ['--reload']
        else: cmd += ['--contexts', '0,4096,16384', '--batches', '1,3,4', '--rounds', '48']
        jobs[name] = (cmd, '2')
    # Repeat the ineligible B1 control with a longer steady-state window when
    # short samples vary. Keep this separate from the original benchmark data.
    jobs['nemotron-control'] = (['util/validate_native_moe.py', '--model',
        'D:/tabbyAPI/models/Nemotron-3.5-Super-VL-4.23bpw',
        '--output', str(out/'nemotron-control'), '--phase', 'bench',
        '--tune-from', str(ROOT/'native-results/nemotron-moe-performance/long-cuda/final-tune.bin'),
        '--contexts', '16384', '--batches', '1', '--repeats', '3',
        '--warmup', '128', '--rounds', '256'], '2')
    batch_control = list(jobs['nemotron-control'][0])
    for flag, value in [('--output', str(out/'nemotron-batch-control')), ('--contexts', '0'),
                        ('--batches', '4'), ('--warmup', '64'), ('--rounds', '128')]:
        batch_control[batch_control.index(flag)+1] = value
    jobs['nemotron-batch-control'] = (batch_control, '2')
    jobs['installed-regression'] = (['-m','pytest',
        *[str(p.relative_to(stage)) for p in (stage/'tests').glob('test_native*.py')],
        'tests/kernels/moe', 'tests/modules/moe', 'tests/kernels/hyperconnections',
        'tests/kernels/recurrent', 'tests/generator/test_failure_containment.py',
        '-q', '--tb=short', '--junitxml='+str(out/'installed-regression.xml')], '2,0,1')
    unknown = set(a.only) - jobs.keys()
    if unknown: parser.error(f'Unknown gates: {unknown}')
    runs = []
    for name in a.only:
        cmd, devices = jobs[name]
        print('START', name, flush=True)
        start = time.monotonic()
        with (out/(name+'.log')).open('w', encoding='utf-8') as log:
            r = subprocess.run([sys.executable, *cmd], cwd=stage,
                env=dict(env, CUDA_VISIBLE_DEVICES=devices), stdout=log, stderr=subprocess.STDOUT)
        runs.append(dict(name=name, args=cmd, devices=devices, returncode=r.returncode,
                         seconds=time.monotonic()-start))
        (out/'runs.json').write_text(json.dumps(runs, indent=2))
        print('END', name, r.returncode, flush=True)
        if r.returncode: break
    assert payload() == before, 'Runtime payload changed during validation'
    passed = len(runs) == len(a.only) and all(r['returncode'] == 0 for r in runs)
    (out/('passed.json' if passed else 'failed.json')).write_text(json.dumps(runs, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
