#pragma once
#include "blocksparse_mlp.h"
#include "../graph.cuh"
#include <array>

// Row-local orchestration only. Python owns all persistent attention state.
struct BC_BlockPlan {
    enum Op { GR_MIX, HC_MIX, APPLY, RMS_NORM, ROUTE, MOE, HC_CARRY };
    struct Site {
        std::vector<at::Tensor> w;
        double eps, hc_eps;
        int64_t iterations;
        at::Tensor dots, post, comb, mixed, normalized, output, pre;
    };
    using Kernel = void (BC_BlockPlan::*)(Site&, const at::Tensor&, at::Tensor&);
    struct Instruction { Kernel kernel; size_t site; };
    std::vector<Site> sites;
    std::array<std::vector<Instruction>, 3> stages;
    std::vector<at::Tensor> router;
    std::shared_ptr<BC_BlockSparseMLP> moe;
    int64_t batch, query, rows, channels, hidden;
    bool carry_mode = false;
    at::Tensor carry_output;
    struct GraphBinding {
        at::Tensor tensor;
        void* pointer;
        std::vector<int64_t> sizes, strides;
        at::ScalarType dtype;
        at::Device device;
    };
    std::vector<GraphBinding> graph_bindings;
    // Begin has no dynamic boundary; middle specializes its boundary dtype.
    std::array<std::unique_ptr<Graph>, 3> stage_graphs;
    std::array<bool, 2> graph_eligible = {false, false};
    bool graphs_enabled = false;
    uint64_t graph_captures = 0, graph_replays = 0;
    BC_BlockPlan(std::vector<std::vector<at::Tensor>> weights,
        std::vector<int64_t> kinds, std::vector<double> eps,
        std::vector<double> hc_eps, std::vector<int64_t> iterations,
        std::vector<std::vector<std::pair<int64_t,int64_t>>> code,
        std::vector<at::Tensor> router, std::shared_ptr<BC_BlockSparseMLP> moe,
        int64_t batch, int64_t query);
    ~BC_BlockPlan();
    void set_graphs(bool enabled);
    void execute_graph(size_t stage, const at::Tensor& residual, at::Tensor y);
    at::Tensor begin(const at::Tensor& x);
    at::Tensor begin_carry(const at::Tensor& x, const c10::optional<at::Tensor>& pre);
    at::Tensor middle(const at::Tensor& x, const at::Tensor& y);
    void finish(const at::Tensor& x, const at::Tensor& y);
    at::Tensor execute(size_t stage, const at::Tensor& x, at::Tensor y);
    void gr(Site&, const at::Tensor&, at::Tensor&);
    void hc(Site&, const at::Tensor&, at::Tensor&);
    void hc_carry(Site&, const at::Tensor&, at::Tensor&);
    void apply(Site&, const at::Tensor&, at::Tensor&);
    void norm(Site&, const at::Tensor&, at::Tensor&);
    void route(Site&, const at::Tensor&, at::Tensor&);
    void experts(Site&, const at::Tensor&, at::Tensor&);
};
