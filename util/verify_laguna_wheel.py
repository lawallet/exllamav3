"""Run a validation script with all product imports pinned to an extracted wheel."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys
p = argparse.ArgumentParser()
p.add_argument('--package', type=Path, required=True)
p.add_argument('--script', required=True)
p.add_argument('args', nargs=argparse.REMAINDER)
a = p.parse_args()
package = a.package.resolve()
sys.path.insert(0, str(package))
import exllamav3
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.version import __version__
assert Path(exllamav3.__file__).resolve().is_relative_to(package)
assert Path(ext.__file__).resolve().is_relative_to(package)
assert __version__ == '1.5.1+nativeplan.v2.laguna1'
assert importlib.metadata.version('exllamav3') == __version__
assert ext.native_block_plan_v2_abi == ext.native_block_plan_v2_residual_ops == 1
print(json.dumps(dict(version=__version__, package=exllamav3.__file__, extension=ext.__file__)), flush=True)
args = a.args[1:] if a.args[:1] == ['--'] else a.args
sys.argv = [a.script, '--extension-dir', str(package), *args]
runpy.run_path(str(Path(__file__).resolve().parent/a.script), run_name='__main__')
for name, module in tuple(sys.modules.items()):
    if name == 'exllamav3' or name.startswith('exllamav3.'):
        path = getattr(module, '__file__', None)
        if path: assert Path(path).resolve().is_relative_to(package), (name, path)
print('PASS all product imports came from extracted wheel', flush=True)
