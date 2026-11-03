"""Isolated pipeline A/B control; package source is selected before importing."""
import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('pipeline_followup',Path(__file__).with_name('serve_deepseek_v41_followup.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
b=f.b
from exllamav3.generator import stop_mask_cache
import exllamav3.architecture.deepseek_v4_mtp as draft_arch

# Experimental switch, confined to this diagnostic process. The generator owns
# this pinned seed buffer until the final draft readback completes. Device-to-
# device copies, pageable sources and non-token tensors retain their old path.
async_seed=False
old_to_dev=draft_arch.to_dev
def experimental_to_dev(t,device):
    if async_seed and t.device.type=='cpu' and t.dtype==b.torch.long and t.is_pinned():
        return f.device_copy.to_device(t,device,non_blocking=True)
    return old_to_dev(t,device)
draft_arch.to_dev=experimental_to_dev
try:
    from exllamav3.util import draft_seed as implemented_seed
except ImportError:
    implemented_seed=None
if implemented_seed is not None:
    draft_arch.to_dev=old_to_dev

old_setup=b.server.setup_app
def policy_status():
    return dict(enabled=stop_mask_cache.enabled,
        async_seed=implemented_seed.enabled if implemented_seed is not None else async_seed,
        module=stop_mask_cache.__file__,
        seed_implementation=implemented_seed.__file__ if implemented_seed is not None else 'diagnostic prototype')
def setup(*args,**kwargs):
    app=old_setup(*args,**kwargs)
    @app.post('/__pipeline/control',dependencies=[b.Depends(b.check_admin_key)])
    async def control(body:dict):
        global async_seed
        c=b.serving.container
        if c.active_job_ids or c.generator.jobs:raise b.HTTPException(409,'requests active')
        stop_mask_cache.enabled=bool(body['enabled'])
        async_seed=bool(body.get('async_seed',False))
        if implemented_seed is not None:implemented_seed.enabled=async_seed
        return policy_status()
    @app.get('/__pipeline/status',dependencies=[b.Depends(b.check_admin_key)])
    async def status():return policy_status()
    return app
b.server.setup_app=setup
if __name__=='__main__':
    import main
    main.entrypoint()
