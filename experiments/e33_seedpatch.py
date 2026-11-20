"""E33 helper: run an experiment script on a new random instance without editing it.

apply(k) replaces numpy.random.default_rng by a wrapper that adds k * OFFSET to every integer seed, so every random
draw of a script (random fields, imperfections, patterns, bootstrap resamples) comes from a new stream while the code
path stays the same. k = 0 leaves the seeds unchanged and reproduces the committed results. With `results`, the
output folder common.RESULTS is redirected, so that a rerun never overwrites a committed result file; this must happen
before the experiment module is imported, because the scripts bind RESULTS at import.

pool_factory(k, results, cap) returns a drop-in replacement for ProcessPoolExecutor whose worker processes apply the
same patch before they import the experiment module (the initializer runs before the first task is unpickled), and
then run the script's own initializer, if any. It also caps the number of workers.
"""

import importlib
from pathlib import Path

import numpy as np

OFFSET = 1_000_003
_ORIGINAL = np.random.default_rng


def apply(k, results=None):
    k = int(k)

    def default_rng(seed=None):
        if isinstance(seed, (int, np.integer)) and not isinstance(seed, bool):
            seed = int(seed) + k * OFFSET
        return _ORIGINAL(seed)

    np.random.default_rng = default_rng if k else _ORIGINAL
    if results is not None:
        import common
        common.RESULTS = Path(results)


def worker_init(k, results, init_module=None, init_name=None, init_args=()):
    apply(k, results)
    if init_module is not None:
        getattr(importlib.import_module(init_module), init_name)(*init_args)


def pool_factory(k, results, cap):
    from concurrent.futures import ProcessPoolExecutor

    def make(max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        n = max(1, min(max_workers or cap, cap))
        ref = (initializer.__module__, initializer.__qualname__) if initializer is not None else (None, None)
        return ProcessPoolExecutor(max_workers=n, mp_context=mp_context, initializer=worker_init,
                                   initargs=(k, None if results is None else str(results), *ref, tuple(initargs)),
                                   **kw)

    return make
