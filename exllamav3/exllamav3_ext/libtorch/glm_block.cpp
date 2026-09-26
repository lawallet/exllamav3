#include "glm_block.h"
#include "../hc_mix.cuh"
#include "../norm.cuh"
#include <ATen/Functions.h>
#include <c10/cuda/CUDAGuard.h>

BC_GlmBlock::BC_GlmBlock(std::vector<at::Tensor> w, double ne, double he, int64_t it)
    : weights(std::move(w)), norm_eps(ne), hc_eps(he), iterations(it)
{
    TORCH_CHECK(weights.size() == 8, "GLM: expected two fn/base/scale/norm bundles");
    auto device = weights[0].device();
    TORCH_CHECK(device.is_cuda(), "GLM: CUDA weights required");
    c10::cuda::CUDAGuard guard(device);
    for (const auto& t : weights)
        TORCH_CHECK(t.defined() && t.device() == device && t.is_contiguous(), "GLM: invalid weight placement/layout");
    for (int i : {0, 4}) {
        TORCH_CHECK(weights[i].scalar_type() == at::kHalf && weights[i].sizes() == at::IntArrayRef({24, 16384}), "GLM: fn shape/type");
        TORCH_CHECK(weights[i+1].scalar_type() == at::kFloat && weights[i+1].numel() == 24, "GLM: base shape/type");
        TORCH_CHECK(weights[i+2].scalar_type() == at::kFloat && weights[i+2].numel() == 3, "GLM: scale shape/type");
        TORCH_CHECK(weights[i+3].numel() == 4096 && (weights[i+3].scalar_type() == at::kHalf || weights[i+3].scalar_type() == at::kBFloat16 || weights[i+3].scalar_type() == at::kFloat), "GLM: norm shape/type");
    }
    auto opts = weights[1].options();
    partials = at::empty({1, hc_mix_num_chunks(1, 16384), 25}, opts);
    post = at::empty({1, 4}, opts);
    comb = at::empty({1, 4, 4}, opts);
    collapsed = at::empty({1, 4096}, opts.dtype(at::kHalf));
    normalized = at::empty_like(collapsed);
    normalized3 = normalized.view({1, 1, 4096});
}

void BC_GlmBlock::validate(const at::Tensor& x) {
    TORCH_CHECK(x.device() == weights[0].device() && x.scalar_type() == at::kFloat &&
                x.is_contiguous() && x.sizes() == at::IntArrayRef({1, 1, 4, 4096}), "GLM: invalid residual");
}

at::Tensor BC_GlmBlock::mix_norm(const at::Tensor& x, int site) {
    const int i = site * 4;
    hc_mix(x.view({1, 4, 4096}), weights[i], weights[i+1], weights[i+2],
           norm_eps, hc_eps, iterations, partials, post, comb, collapsed);
    rms_norm(collapsed, weights[i+3], normalized, norm_eps, 0.0f, 1.0f, false, false, 1);
    return normalized3;
}

void BC_GlmBlock::apply(const at::Tensor& x, const at::Tensor& y) {
    TORCH_CHECK(y.device() == x.device() && y.is_contiguous() &&
                y.sizes() == at::IntArrayRef({1, 1, 4096}) &&
                (y.scalar_type() == at::kFloat || y.scalar_type() == at::kHalf), "GLM: invalid sublayer output");
    hc_apply(x.view({1, 4, 4096}), y.view({1, 4096}), post, comb);
}

at::Tensor BC_GlmBlock::begin(const at::Tensor& x) {
    validate(x);
    c10::cuda::CUDAGuard guard(x.device());
    return mix_norm(x, 0);
}

at::Tensor BC_GlmBlock::middle(const at::Tensor& x, const at::Tensor& y) {
    validate(x);
    c10::cuda::CUDAGuard guard(x.device());
    apply(x, y);
    return mix_norm(x, 1);
}

void BC_GlmBlock::finish(const at::Tensor& x, const at::Tensor& y) {
    validate(x);
    c10::cuda::CUDAGuard guard(x.device());
    apply(x, y);
}
