"""Run shared native regressions against an explicit isolated extension."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--extension-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--pattern',default='test_native*.py')
a=p.parse_args()
paths=[str(ROOT),str(a.extension_dir.resolve()),str(ROOT/'tests')]
sys.path[:0]=paths
os.environ['PYTHONPATH']=os.pathsep.join(paths)
import torch
from exllamav3.ext import exllamav3_ext as ext
assert Path(ext.__file__).parent==a.extension_dir.resolve()
assert ext.native_block_plan_carry_abi==1
start=time.perf_counter()
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern=a.pattern)
result=unittest.TextTestRunner(stream=sys.stdout,verbosity=2).run(suite)
report=dict(tests=result.testsRun,skipped=result.skipped,errors=[(str(t),s) for t,s in result.errors],
            failures=[(str(t),s) for t,s in result.failures],success=result.wasSuccessful(),
            seconds=time.perf_counter()-start,extension=ext.__file__,
            devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2))
raise SystemExit(not result.wasSuccessful())
