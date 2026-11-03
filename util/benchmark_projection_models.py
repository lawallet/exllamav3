"""Reuse the established cross-model workloads, changing only the A/B policy.

This diagnostic adapter deliberately leaves the validated stop-mask and draft
seed flags enabled. The source substitutions are checked so harness drift fails
explicitly instead of silently benchmarking the wrong feature.
"""
from pathlib import Path

source_path=Path(__file__).with_name('benchmark_pipeline_models.py')
source=source_path.read_text(encoding='utf-8')
old='''                stop_mask_cache.enabled=bool(value)
                draft_seed.enabled=bool(value)'''
new='''                native_projection.enabled=bool(value)
                stop_mask_cache.enabled=True
                draft_seed.enabled=True'''
assert source.count(old)==1
source=source.replace(old,new)
old='from exllamav3.modules import native_plan, native_moe, transformer'
assert source.count(old)==2
source=source.replace(old,old+', native_projection')
exec(compile(source,str(source_path),'exec'),globals())
