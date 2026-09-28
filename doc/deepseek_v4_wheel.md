# DeepSeek V4 native-plan wheel

Local wheel:
`dist/deepseek-v4-nativeplan-1/wheels/exllamav3-1.5.1+nativeplan.deepseekv4.1-cp313-cp313-win_amd64.whl`

Built for Windows x64, CPython 3.13, the pinned Torch 2.9.0+cu128 runtime,
and CUDA architectures 8.9/12.0. Enable DeepSeek plans with `EXL3_NATIVE_PLAN=1`.
The wheel has not been installed into TabbyAPI or published.

Wheel SHA256:
`3f63a74c1496901b9bae1ee0bda9fea356821fe241bf6b3f80cd71adf4ebc4c1`

Packaging copied the exact model-tested extension, without recompiling:
`40ca8e4f2910e72655dc42422af94d72dd629f314746c2d565840c9787c72473`.
Every packaged Python source was checked against the checkout. The wheel is
tagged `cp313-cp313-win_amd64`; its version differs from the earlier IR wheel.

After extraction, package and extension imports were verified to resolve from
the wheel. All 113 native guard tests and 414 actual-weight DeepSeek target/draft
site cases passed using that extracted package. Full model, topology, 64K and
performance results for these sources and the identical extension are in
[the validation report](deepseek_v4_validation.md).

Build proof, source hashes, checksum manifest, extracted package and check logs
are in `dist/deepseek-v4-nativeplan-1`. Reproduce packaging with the pinned
Python using `util/package_deepseek_plan.py --validation native-results/deepseek-v4
--output <new-directory>`. This uses the established isolated setuptools build
and does not install packages or resolve dependencies.
