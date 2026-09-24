#include "qwen_block.h"
#include "../hc_mix.cuh"
#include "../routing.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/Functions.h>

BC_QwenBlock::BC_QwenBlock(std::vector<at::Tensor> w,
    std::shared_ptr<BC_BlockSparseMLP> m, double ae, double me)
    : weights(std::move(w)), moe(std::move(m)), attn_eps(ae), mlp_eps(me)
{
    // [attn fn/up/w, mlp fn/up/w, gate/gate_t/i8/scales, logits/sel/routing]
    TORCH_CHECK(weights.size() == 13 && moe, "Qwen block: invalid parameter bundle");
    auto device = weights[0].device();
    for (const auto& t : weights)
        TORCH_CHECK(t.defined() && t.is_cuda() && t.device() == device,
                    "Qwen block: all weights/workspaces must be on one CUDA device");
    c10::cuda::CUDAGuard guard(device);
    auto opts = weights[2].options().dtype(at::kFloat);
    const auto h = weights[1].size(0), d = weights[1].size(1) * 4;
    attn_dots = at::empty({1, weights[0].size(0) + 1, h}, opts);
    mlp_dots = at::empty({1, weights[3].size(0) + 1, h}, opts);
    attn_post = at::empty({1, h}, opts);
    mlp_post = at::empty({1, h}, opts);
    attn_mixed = at::empty({1, d}, opts.dtype(at::kHalf));
    mlp_mixed = at::empty({1, d}, opts.dtype(at::kHalf));
}

at::Tensor BC_QwenBlock::begin(const at::Tensor& x)
{
    TORCH_CHECK(x.dim() == 4 && x.size(0) == 1 && x.size(1) == 1 &&
                x.size(2) == attn_post.size(1) && x.size(3) == attn_mixed.size(1) &&
                x.scalar_type() == at::kFloat && x.is_contiguous() &&
                x.device() == weights[0].device(), "Qwen block: invalid residual stack");
    c10::cuda::CUDAGuard guard(x.device());
    gr_mix(x.view({1, x.size(2), x.size(3)}), weights[0], weights[1], weights[2],
           attn_eps, attn_dots, attn_post, attn_mixed);
    return attn_mixed.view({1, 1, x.size(3)});
}

void BC_QwenBlock::finish(at::Tensor x, const at::Tensor& attention_output)
{
    c10::cuda::CUDAGuard guard(x.device());
    auto streams = x.view({1, x.size(2), x.size(3)});
    auto y = attention_output.reshape({1, x.size(3)}).contiguous();
    // Consume attention gates before mixing the updated residual. No kernel math,
    // quantization, stream ordering, or cache operations change here.
    hc_apply(streams, y, attn_post, c10::nullopt);
    gr_mix(streams, weights[3], weights[4], weights[5], mlp_eps,
           mlp_dots, mlp_post, mlp_mixed);
    routing_std(mlp_mixed, weights[6], weights[10], weights[11], weights[12],
                c10::nullopt, weights[7], c10::nullopt, weights[8], weights[9]);
    moe->run_bszN(mlp_mixed, weights[11], weights[12]);
    hc_apply(streams, moe->out_bszn.slice(0, 0, 1), mlp_post, c10::nullopt);
}
