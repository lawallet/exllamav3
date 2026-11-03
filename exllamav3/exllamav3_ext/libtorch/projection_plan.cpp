#include "projection_plan.h"
#include "../norm.cuh"
#include "../quant/exl3_gemm.cuh"
#include <ATen/Functions.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

BC_ProjectionPlan::BC_ProjectionPlan(std::vector<at::Tensor> w, double eps, int64_t q)
    : weights(std::move(w)), query(q), epsilon(static_cast<float>(eps))
{
    TORCH_CHECK(weights.size()==10 && q>=1 && q<=5 && std::isfinite(epsilon) && epsilon>0,
                "Projection plan: invalid specialization");
    TORCH_CHECK(weights[0].defined() && weights[0].is_cuda(), "Projection plan: CUDA weights required");
    device=weights[0].device();
    c10::cuda::CUDAGuard guard(device);
    stream=at::cuda::getCurrentCUDAStream().stream();
    for (const auto& t:weights)
        TORCH_CHECK(t.defined() && t.device()==device && t.is_contiguous(),
                    "Projection plan: invalid weight placement/layout");
    std::array<int64_t,3> in{},out{};
    for (int i=0;i<3;++i) {
        const auto& t=weights[3*i]; const auto& su=weights[3*i+1]; const auto& sv=weights[3*i+2];
        TORCH_CHECK(t.scalar_type()==at::kShort && t.dim()==3 && t.size(2)==48 &&
                    t.size(0)>0 && t.size(1)>0 && t.size(0)<=8192 && t.size(1)<=8192,
                    "Projection plan: requires bounded K3 EXL3 weights");
        in[i]=t.size(0)*16; out[i]=t.size(1)*16;
        TORCH_CHECK(in[i]%128==0 && out[i]%128==0 && su.scalar_type()==at::kHalf &&
                    sv.scalar_type()==at::kHalf && su.sizes()==at::IntArrayRef({in[i]}) &&
                    sv.sizes()==at::IntArrayRef({out[i]}), "Projection plan: invalid sign vectors");
        scratch[i]=at::empty({q,in[i]},su.options());
    }
    TORCH_CHECK(in[1]==out[0] && in[2]==in[0], "Projection plan: incompatible projection topology");
    const auto& norm=weights[9];
    TORCH_CHECK(norm.sizes()==at::IntArrayRef({out[0]}) &&
                (norm.scalar_type()==at::kHalf || norm.scalar_type()==at::kBFloat16),
                "Projection plan: invalid norm binding");
    hidden=in[0]; auto opts=weights[1].options();
    raw_query=at::empty({q,out[0]},opts); query_latent=at::empty_like(raw_query);
    query_output=at::empty({q,out[1]},opts); kv_output=at::empty({q,out[2]},opts);
    outputs={query_latent.view({1,q,out[0]}),query_output.view({1,q,out[1]}),kv_output.view({1,q,out[2]})};
    auto bind=[this](const at::Tensor& t) {
        bindings.push_back({t,t.data_ptr(),t.sizes().vec(),t.strides().vec(),t.scalar_type(),t.device()});
    };
    for (const auto& t:weights) bind(t);
    for (const auto& t:scratch) bind(t);
    for (const auto& t:{raw_query,query_latent,query_output,kv_output}) bind(t);
    for (const auto& t:outputs) bind(t);
}

BC_ProjectionPlan::~BC_ProjectionPlan() {
    if (graph) { c10::cuda::CUDAGuard guard(device); graph.reset(); }
}

void BC_ProjectionPlan::execute(const at::Tensor& x, Graph* capture) {
    auto linear=[&](int i,const at::Tensor& input,at::Tensor& output) {
        exl3_gemm_gr(input,weights[3*i],output,weights[3*i+1],scratch[i],weights[3*i+2],
                     -1,false,true,0,capture);
    };
    linear(0,x,raw_query);
    rms_norm_gr(raw_query,weights[9],query_latent,epsilon,0.0f,1.0f,capture);
    linear(1,query_latent,query_output);
    linear(2,x,kv_output);
}

std::vector<at::Tensor> BC_ProjectionPlan::run(const at::Tensor& x) {
    TORCH_CHECK(x.device()==device && x.scalar_type()==at::kHalf && x.is_contiguous() &&
                x.sizes()==at::IntArrayRef({1,query,hidden}), "Projection plan: invalid input");
    c10::cuda::CUDAGuard guard(device);
    TORCH_CHECK(stream==at::cuda::getCurrentCUDAStream().stream(), "Projection plan: changed execution stream");
    for (const auto& b:bindings)
        TORCH_CHECK(b.tensor.device()==b.device && b.tensor.scalar_type()==b.dtype &&
                    b.tensor.sizes()==at::IntArrayRef(b.sizes) && b.tensor.strides()==at::IntArrayRef(b.strides) &&
                    b.tensor.data_ptr()==b.pointer, "Projection plan: changed fixed binding");
    for (const auto& t:{raw_query,query_latent,query_output,kv_output}) at::assert_no_overlap(x,t);
    if (!graph) graph=std::make_unique<Graph>();
    if (!graphs_enabled || graph->disabled) { execute(x,nullptr); return outputs; }
    if (!graph->ready) {
        // Resolve the exact upstream kernel/autotuner and device workspaces
        // before capture; the following graph uses those same primitives.
        execute(x,nullptr);
        GraphCapture capture(*graph);
        execute(x,graph.get()); capture.finish(); ++graph_captures;
    }
    graph->launch({{GP_gemm_A,x.data_ptr()},{GP_gemm_A,query_latent.data_ptr()},
                   {GP_gemm_A,x.data_ptr()}},stream);
    ++graph_replays;
    return outputs;
}
