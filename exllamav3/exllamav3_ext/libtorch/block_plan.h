#pragma once
#include "blocksparse_mlp.h"
#include <array>

// Row-local orchestration only. Python owns all persistent attention state.
struct BC_BlockPlan {
    enum Op { GR_MIX, HC_MIX, APPLY, RMS_NORM, ROUTE, MOE };
    struct Site {
        std::vector<at::Tensor> w;
        double eps, hc_eps;
        int64_t iterations;
        at::Tensor dots, post, comb, mixed, normalized, output;
    };
    using Kernel = void (BC_BlockPlan::*)(Site&, const at::Tensor&, at::Tensor&);
    struct Instruction { Kernel kernel; size_t site; };
    std::vector<Site> sites;
    std::array<std::vector<Instruction>, 3> stages;
    std::vector<at::Tensor> router;
    std::shared_ptr<BC_BlockSparseMLP> moe;
    int64_t batch, query, rows, channels, hidden;
    BC_BlockPlan(std::vector<std::vector<at::Tensor>> weights,
        std::vector<int64_t> kinds, std::vector<double> eps,
        std::vector<double> hc_eps, std::vector<int64_t> iterations,
        std::vector<std::vector<std::pair<int64_t,int64_t>>> code,
        std::vector<at::Tensor> router, std::shared_ptr<BC_BlockSparseMLP> moe,
        int64_t batch, int64_t query);
    at::Tensor begin(const at::Tensor& x);
    at::Tensor middle(const at::Tensor& x, const at::Tensor& y);
    void finish(const at::Tensor& x, const at::Tensor& y);
    at::Tensor execute(size_t stage, const at::Tensor& x, at::Tensor y);
    void gr(Site&, const at::Tensor&, at::Tensor&);
    void hc(Site&, const at::Tensor&, at::Tensor&);
    void apply(Site&, const at::Tensor&, at::Tensor&);
    void norm(Site&, const at::Tensor&, at::Tensor&);
    void route(Site&, const at::Tensor&, at::Tensor&);
    void experts(Site&, const at::Tensor&, at::Tensor&);
};
