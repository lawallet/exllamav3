"""Qwen exact regression with GLM enabled, importing the packaged GLM wheel."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-dir', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.package_dir.resolve()))
    import exllamav3
    import exllamav3_ext
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.modules import transformer
    from util.check_qwen_native import check
    transformer._native_glm_enabled = True
    config = Config.from_directory(args.model)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=512)
    model.load(device='cuda:0', progressbar=True)
    prompt = (ROOT / 'doc/native_qwen_prompt.txt').read_text(encoding='utf-8')
    result = check(model, cache, tokenizer, prompt)
    result.update(package=exllamav3.__file__, extension=exllamav3_ext.__file__, glm_flag_enabled=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
