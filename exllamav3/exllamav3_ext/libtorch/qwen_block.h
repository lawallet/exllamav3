#pragma once
#include "blocksparse_mlp.h"

// Experimental batch-one Qwen4Exp orchestration. Attention retains its existing
// Python cache/state preparation; begin/finish own only stateless residual/MoE work.
struct BC_QwenBlock
{
    std::vector<at::Tensor> weights;
    std::shared_ptr<BC_BlockSparseMLP> moe;
    double attn_eps, mlp_eps;
    at::Tensor attn_dots, attn_post, attn_mixed;
    at::Tensor mlp_dots, mlp_post, mlp_mixed;
    BC_QwenBlock(std::vector<at::Tensor> weights,
                 std::shared_ptr<BC_BlockSparseMLP> moe, double attn_eps, double mlp_eps);
    at::Tensor begin(const at::Tensor& x);
    void finish(at::Tensor x, const at::Tensor& attention_output);
};
