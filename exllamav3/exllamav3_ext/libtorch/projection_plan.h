#pragma once
#include "../graph.cuh"
#include <array>
#include <memory>

// Stateless native IR boundary: linear -> RMSNorm -> linear, plus a parallel
// input projection. No positions, attention, cache, index or speculative state.
struct BC_ProjectionPlan {
    BC_ProjectionPlan(std::vector<at::Tensor> weights, double epsilon, int64_t query);
    ~BC_ProjectionPlan();
    std::vector<at::Tensor> run(const at::Tensor& x);
    void set_graphs(bool enabled) { graphs_enabled = enabled; }
    uint64_t graph_captures = 0, graph_replays = 0;
private:
    struct Binding {
        at::Tensor tensor;
        void* pointer;
        std::vector<int64_t> sizes, strides;
        at::ScalarType dtype;
        at::Device device;
    };
    std::vector<at::Tensor> weights;
    std::vector<Binding> bindings;
    std::array<at::Tensor, 3> scratch;
    at::Tensor raw_query, query_latent, query_output, kv_output;
    std::vector<at::Tensor> outputs;
    std::unique_ptr<Graph> graph;
    at::Device device{at::kCPU};
    cudaStream_t stream;
    int64_t query, hidden;
    float epsilon;
    bool graphs_enabled = true;
    void execute(const at::Tensor& x, Graph* capture);
};
