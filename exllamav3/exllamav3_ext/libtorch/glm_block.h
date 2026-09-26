#pragma once
#include <ATen/Tensor.h>
#include <vector>

// Single-row mHC/RMSNorm orchestration. Attention and MLP remain stock callers.
struct BC_GlmBlock
{
    std::vector<at::Tensor> weights;
    at::Tensor partials, post, comb, collapsed, normalized, normalized3;
    double norm_eps, hc_eps;
    int64_t iterations, rows;
    BC_GlmBlock(std::vector<at::Tensor> w, double ne, double he, int64_t it, int64_t r = 1);
    at::Tensor begin(const at::Tensor& x);
    at::Tensor middle(const at::Tensor& x, const at::Tensor& y);
    void finish(const at::Tensor& x, const at::Tensor& y);
    void validate(const at::Tensor& x);
    at::Tensor mix_norm(const at::Tensor& x, int site);
    void apply(const at::Tensor& x, const at::Tensor& y);
};
