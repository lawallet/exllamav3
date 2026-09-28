"""Validation-only backend switching; never used in timed block execution."""
from contextlib import contextmanager


@contextmanager
def backend(model, plan):
    from exllamav3.modules import native_plan
    previous = native_plan.enabled
    def clear():
        models = model if isinstance(model, (list, tuple)) else [model]
        for block in (b for m in models for b in m.modules):
            for name in ('_native_qwen', '_native_glm'):
                if hasattr(block, name): setattr(block, name, None)
            for name in ('_native_qwen_mtp', '_native_glm_mtp', '_native_glm_batch', '_native_deepseek', '_native_gemma', '_native_devstral', '_native_laguna'):
                if hasattr(block, name): setattr(block, name, {})
    clear()
    native_plan.enabled = plan
    try:
        yield
    finally:
        clear()
        native_plan.enabled = previous


def three_way_reference(model, run):
    from exllamav3.modules import native_plan
    if not native_plan.enabled:
        return None
    with backend(model, False):
        return run(True)


def benchmark(model, run, tokens, runs, output, profile_cls, profile_tokens):
    """Alternate backends on one loaded model; specialize/warm outside timing."""
    import json
    records = []
    expected = None
    for repeat in range(runs):
        for plan in ((False, True) if repeat % 2 == 0 else (True, False)):
            with backend(model, plan):
                run(64)
                result = dict(backend='plan' if plan else 'bespoke', repeat=repeat, **run(tokens))
                records.append(result)
                (output / 'bespoke-plan-benchmark.json').write_text(json.dumps(records, indent=2))
                exact = {k: result[k] for k in ('completion_sha256', 'accepted', 'rejected', 'proposed',
                         'accepted_draft_tokens', 'rejected_draft_tokens', 'proposed_draft_tokens') if k in result}
                if expected is None: expected = exact
                assert exact == expected, 'backend benchmark completion/acceptance mismatch'
                print(json.dumps(result), flush=True)
                if repeat == 0 and profile_cls is not None:
                    with profile_cls(model) as profile:
                        run(profile_tokens)
                    (output / f'dispatch-{result["backend"]}.json').write_text(json.dumps(profile.report(), indent=2))
    from statistics import median
    rates = {name: median(r['decode_tokens_per_second'] for r in records if r['backend'] == name)
             for name in ('bespoke', 'plan')}
    comparison = dict(median_tps=rates, relative_change=rates['plan']/rates['bespoke']-1)
    (output/'bespoke-plan-performance.json').write_text(json.dumps(comparison, indent=2))
    if runs >= 3 and comparison['relative_change'] < -.03:
        raise AssertionError(f'Repeatable plan regression requires investigation: {comparison}')
    return records


def memory_report(model, output):
    import json
    import torch
    participating = {torch.device(m.device).index for m in model.modules
                     if m.device is not None and torch.device(m.device).type == 'cuda'}
    report = [dict(device=i, participating=i in participating,
                   allocated=torch.cuda.memory_allocated(i),
                   reserved=torch.cuda.memory_reserved(i), peak_allocated=torch.cuda.max_memory_allocated(i))
              for i in range(torch.cuda.device_count())]
    (output / 'device-memory.json').write_text(json.dumps(report, indent=2))
    assert all(r['peak_allocated'] == 0 for r in report if not r['participating']), report


def benchmark_jobs(common, run_jobs, suite, output, runs, diagnostic=False):
    import json
    from statistics import median
    records = []
    comparisons = []
    cohorts = ([(False, n) for n in (2, 3, 4)] if suite in ('ordinary', 'all') else [])
    cohorts += ([(True, n) for n in (2, 3)] if suite in ('mtp', 'all') else [])
    models = [m for m in common[:2] if m is not None]
    for mtp, count in cohorts:
        start = len(records)
        expected = None
        kw = dict(native=True, mtp=mtp, count=count, lengths=[384, 512, 640, 768][:count])
        for repeat in range(runs):
            for plan in ((False, True) if repeat % 2 == 0 else (True, False)):
                with backend(models, plan):
                    _, warm = run_jobs(*common, **kw, **({'diagnostic': True} if diagnostic else {}))
                    _, result = run_jobs(*common, **kw)
                    exact = {j: {k: v[k] for k in ('ids', 'rng', 'proposed', 'accepted', 'rejected', 'completion_sha256')}
                             for j, v in result['per_job'].items()}
                    warm_exact = {j: {k: v[k] for k in ('ids', 'rng', 'proposed', 'accepted', 'rejected', 'completion_sha256')}
                                  for j, v in warm['per_job'].items()}
                    assert result['shapes'] == warm['shapes'] and exact == warm_exact, 'Instrumented warmup differs from timed cohort'
                    result.update(backend='plan' if plan else 'bespoke', repeat=repeat,
                                  calls=warm['calls'], call_count_source='separate warmed cohort')
                    records.append(result)
                    (output/'bespoke-plan-benchmark.json').write_text(json.dumps(records, indent=2))
                    if expected is None: expected = exact
                    assert exact == expected, 'Concurrent backend output/state mismatch'
                    print('PLAN BENCH', mtp, count, result['backend'], result['aggregate_tps'], flush=True)
        rates = {name: median(r['aggregate_tps'] for r in records[start:] if r['backend'] == name)
                 for name in ('bespoke', 'plan')}
        comparison = dict(mtp=mtp, batch=count, median_tps=rates, relative_change=rates['plan']/rates['bespoke']-1)
        comparisons.append(comparison)
        (output/'bespoke-plan-performance.json').write_text(json.dumps(comparisons, indent=2))
        if runs >= 3 and comparison['relative_change'] < -.03:
            raise AssertionError(f'Concurrent plan regression requires investigation: {comparison}')
