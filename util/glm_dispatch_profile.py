"""Native boundary/transfer diagnostics, shared by GLM and Qwen plan gates.

Never use instrumentation for throughput measurements.
"""
from collections import Counter
import sys

from util.decode_dispatch_profile import DispatchProfile


class GlmDispatchProfile(DispatchProfile):
    def __init__(self, model, ranges=False):
        super().__init__(model, ranges)
        self.native_calls = Counter()
        self.transfers = Counter()
        self.eligibility = Counter()
        self.plan_invocations = 0

    def __enter__(self):
        self.old_profile = sys.getprofile()
        if self.old_profile is not None:
            raise RuntimeError('Refusing to replace an existing Python profiler')
        super().__enter__()
        try:
            for module in self.model.modules:
                original = module.prepare_for_device
                def prepare(x, params, original=original, module=module):
                    if self.active and x.device != module.device:
                        self.transfers[(module.key, str(x.device), str(module.device),
                                        tuple(x.shape), str(x.dtype))] += 1
                    return original(x, params)
                name = 'prepare_for_device'
                self.saved.append((module, name, module.__dict__.get(name), name in module.__dict__))
                setattr(module, name, prepare)
            def profile(frame, event, arg):
                if self.active and event == 'return' and frame.f_code.co_name == 'get_native_block':
                    module = frame.f_globals.get('__name__', '')
                    if module in ('exllamav3.modules.native_qwen', 'exllamav3.modules.native_glm'):
                        self.eligibility[(module, 'fallback' if arg is None else 'native')] += 1
                        if type(arg).__name__ == 'BC_BlockPlan':
                            self.plan_invocations += 1
                if self.active and event == 'c_call':
                    owner = getattr(arg, '__self__', None)
                    mod = getattr(arg, '__module__', '') or getattr(type(owner), '__module__', '')
                    if 'exllamav3_ext' in mod:
                        self.native_calls[f'{mod}:{getattr(arg, "__qualname__", str(arg))}'] += 1
            sys.setprofile(profile)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        sys.setprofile(self.old_profile)
        super().__exit__(*exc)

    def report(self):
        return dict(super().report(),
            native_plan_invocations=self.plan_invocations,
            native_fallbacks=sum(v for k, v in self.eligibility.items() if k[1] == 'fallback'),
            eligibility=[dict(module=k[0], result=k[1], count=v) for k, v in self.eligibility.items()],
            native_calls=[dict(name=k, count=v, per_decode=v/max(1, self.tokens))
                          for k, v in self.native_calls.most_common()],
            transfers=[dict(module=k[0], source=k[1], destination=k[2], shape=k[3], dtype=k[4],
                            count=v, per_decode=v/max(1, self.tokens)) for k, v in self.transfers.items()],
            transfer_semantics='Logical prepare_for_device transfers; physical host bounce is visible in CUDA trace')
