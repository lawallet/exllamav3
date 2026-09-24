"""Opt-in, instance-local host dispatch instrumentation for decode experiments.

Inclusive times overlap; exclusive times subtract instrumented children only.
Neither is GPU execution time or proof that the GPU was idle. No synchronization
is inserted. Use the optional CUDA trace to locate actual device gaps.
"""
from collections import defaultdict
from contextlib import nullcontext
from time import perf_counter_ns


class DispatchProfile:
    def __init__(self, model, ranges=False):
        self.model = model
        self.ranges = ranges
        self.rows = defaultdict(lambda: [0, 0, 0])
        self.stack = []
        self.saved = []
        self.active = False
        self.tokens = 0

    def _patch(self, obj, name, root=False):
        original = getattr(obj, name)
        label = f"{type(obj).__name__}:{getattr(obj, 'key', 'model')}.{name}"

        def wrapped(*args, **kwargs):
            previous = self.active
            if root:
                ids = args[0] if args else kwargs['input_ids']
                self.active = tuple(ids.shape) == (1, 1)
                if self.active:
                    self.tokens += 1
            try:
                if not self.active:
                    return original(*args, **kwargs)
                scope = nullcontext()
                if self.ranges:
                    from torch.profiler import record_function
                    scope = record_function(label)
                with scope:
                    frame = [0]
                    self.stack.append(frame)
                    start = perf_counter_ns()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        elapsed = perf_counter_ns() - start
                        self.stack.pop()
                        if self.stack:
                            self.stack[-1][0] += elapsed
                        row = self.rows[label]
                        row[0] += 1
                        row[1] += elapsed
                        row[2] += elapsed - frame[0]
            finally:
                if root:
                    self.active = previous

        self.saved.append((obj, name, obj.__dict__.get(name), name in obj.__dict__))
        setattr(obj, name, wrapped)

    def __enter__(self):
        try:
            seen = set()
            for module in self.model:
                if id(module) in seen:
                    continue
                seen.add(id(module))
                for name in ('forward', 'mix', 'apply_'):
                    if callable(getattr(module, name, None)):
                        self._patch(module, name)
            self._patch(self.model, 'forward_ls')
            self._patch(self.model, 'forward', root=True)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for obj, name, old, owned in reversed(self.saved):
            if owned:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        self.saved.clear()

    def report(self):
        return {
            'decode_forward_calls': self.tokens,
            'timing_semantics': 'host wall time; exclusive subtracts instrumented children; not GPU idle time',
            'calls': [dict(name=name, count=n, calls_per_decode=n / max(1, self.tokens),
                           inclusive_ms=inc / 1e6, exclusive_ms=exc / 1e6)
                      for name, (n, inc, exc) in sorted(self.rows.items(), key=lambda r: -r[1][2])],
        }
