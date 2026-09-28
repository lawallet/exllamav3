"""Check imports and run guards or actual-weight sites from an extracted wheel."""
import argparse
from pathlib import Path
import runpy
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--package', type=Path, required=True)
p.add_argument('--mode', choices=('guards','sites'), required=True)
p.add_argument('--output', type=Path)
a = p.parse_args()
package = a.package.resolve()
sys.path.insert(0, str(package))
import exllamav3
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.version import __version__
assert Path(exllamav3.__file__).resolve().is_relative_to(package)
assert Path(ext.__file__).resolve().is_relative_to(package)
assert __version__ == '1.5.1+nativeplan.deepseekv4.1'
assert hasattr(ext, 'BC_BlockPlan')
print(f'Wheel import verified: {__version__}; {ext.__file__}', flush=True)
if a.mode == 'guards':
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_native*.py')
    result = unittest.TextTestRunner().run(suite)
    assert result.wasSuccessful() and result.testsRun == 113
else:
    assert a.output is not None
    sys.argv = ['check_deepseek_plan_sites.py', '--extension-dir', str(package), '--output', str(a.output)]
    runpy.run_path(str(ROOT/'util/check_deepseek_plan_sites.py'), run_name='__main__')
