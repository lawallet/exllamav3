#pragma once
#include <ATen/Tensor.h>
#include <array>
#include <vector>

// Additive resolved-value frontend. V1's executor/ABI remains untouched.
struct BC_BlockPlanV2 {
    enum Op { RMS_NORM = 3, NORM_ADD = 6 };
    struct Value {
        int64_t role;
        at::ScalarType dtype;
        at::Tensor tensor, output;
    };
    struct Norm { at::Tensor weight; float eps, bias, scale; };
    struct Instruction;
    using Kernel = void (*)(const Instruction&);
    struct Instruction { Kernel kernel; Value* src; Value* dst; Norm* norm; };
    std::vector<Value> values;
    std::vector<Norm> norms;
    std::array<std::vector<Instruction>, 3> stages;
    std::array<Value*, 3> external{}, outputs{};
    int64_t batch, query, hidden;
    c10::Device device{c10::kCPU};
    BC_BlockPlanV2(std::vector<std::pair<int64_t,int64_t>> specs,
        std::vector<at::Tensor> weights, std::vector<std::vector<double>> config,
        std::vector<std::vector<std::vector<int64_t>>> code,
        std::vector<int64_t> returns, int64_t b, int64_t q, int64_t d);
    at::Tensor begin(const at::Tensor& x);
    at::Tensor middle(const at::Tensor& x, const at::Tensor& y);
    void finish(const at::Tensor& x, const at::Tensor& y);
private:
    at::Tensor execute(size_t stage, const at::Tensor& x, const at::Tensor& y);
    static void norm(const Instruction& op);
    static void norm_add(const Instruction& op);
};
