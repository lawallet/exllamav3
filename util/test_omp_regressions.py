"""Run shared native gates and wait-policy tests with explicitly selected package/binary."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--package-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    package=a.package_dir.resolve()
    sys.path[:0]=[str(package),str(ROOT/'tests')]
    os.environ['PYTHONPATH']=os.pathsep.join([str(package),str(ROOT/'tests')])
    import exllamav3,torch
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.util.openmp import configure_openmp_for_current_thread,openmp_status
    assert Path(exllamav3.__file__).is_relative_to(package)
    assert Path(ext.__file__).parent==package
    configure_openmp_for_current_thread()
    suite=unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_native*.py'))
    suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_openmp_policy.py'))
    start=time.perf_counter()
    result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
    report=dict(tests=result.testsRun,skipped=[(str(t),s) for t,s in result.skipped],
        errors=[(str(t),s) for t,s in result.errors],failures=[(str(t),s) for t,s in result.failures],
        success=result.wasSuccessful(),seconds=time.perf_counter()-start,package=exllamav3.__file__,extension=ext.__file__,
        policy=openmp_status(),torch_threads=torch.get_num_threads(),
        devices=[dict(name=torch.cuda.get_device_name(i),capability=torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())])
    a.output.write_text(json.dumps(report,indent=2))
    raise SystemExit(not report['success'])


if __name__=='__main__':main()
