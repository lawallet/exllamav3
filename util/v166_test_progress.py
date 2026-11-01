"""Persist pytest outcomes even if a native test terminates its process."""
import json
import os
from pathlib import Path


def pytest_configure(config):
    if os.name == 'nt':
        import ctypes
        ucrt = ctypes.CDLL('ucrtbase.dll')
        before = ucrt._getmaxstdio()
        target = max(before, 4096)
        assert ucrt._setmaxstdio(target) == target
        _write(dict(stdio_limit_before=before, stdio_limit_after=ucrt._getmaxstdio()))


def _write(row):
    path = os.environ.get('EXL3_TEST_PROGRESS')
    if path:
        with Path(path).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')


def pytest_collection_finish(session):
    _write(dict(collection=[item.nodeid for item in session.items]))


def pytest_collection_modifyitems(config, items):
    """Resume only failed/unrun cases after a process-wide resource failure."""
    resume = os.environ.get('EXL3_TEST_RESUME')
    if not resume:
        return
    completed, failed = set(), set()
    for line in Path(resume).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if 'nodeid' not in row or row.get('subtest'):
            continue
        if row['outcome'] == 'failed':
            failed.add(row['nodeid'])
        elif row['outcome'] == 'skipped' or (row['when'] == 'call' and row['outcome'] == 'passed'):
            completed.add(row['nodeid'])
    completed -= failed
    selected = [item for item in items if item.nodeid not in completed]
    deselected = [item for item in items if item.nodeid in completed]
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def pytest_runtest_logreport(report):
    _write(dict(nodeid=report.nodeid, when=report.when, outcome=report.outcome,
                duration=report.duration, subtest=hasattr(report, 'context')))
