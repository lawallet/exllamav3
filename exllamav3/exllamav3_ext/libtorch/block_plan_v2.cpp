#include "block_plan_v2.h"
#include "../norm.cuh"
#include <ATen/Functions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <limits>

BC_BlockPlanV2::BC_BlockPlanV2(std::vector<std::pair<int64_t,int64_t>> specs,
    std::vector<at::Tensor> weights, std::vector<std::vector<double>> config,
    std::vector<std::vector<std::vector<int64_t>>> code,
    std::vector<int64_t> returns, int64_t b, int64_t q, int64_t d)
    : batch(b), query(q), hidden(d)
{
    TORCH_CHECK(b>=1 && b<=12 && q>=1 && q<=12 && b*q<=12 && d>0 &&
        d<=std::numeric_limits<int>::max() && d%4==0 && !weights.empty() &&
        weights.size()==config.size() && code.size()==3 && returns.size()==3 &&
        !specs.empty(), "Block plan v2: invalid specialization");
    TORCH_CHECK(weights[0].defined() && weights[0].is_cuda(), "Block plan v2: CUDA weights required");
    device=weights[0].device();
    for (size_t i=0; i<weights.size(); ++i) {
        const auto& w=weights[i]; const auto& c=config[i];
        TORCH_CHECK(w.defined() && w.device()==device && w.is_contiguous() &&
            w.sizes()==at::IntArrayRef({d}) &&
            (w.scalar_type()==at::kHalf || w.scalar_type()==at::kBFloat16),
            "Block plan v2: invalid norm weight");
        TORCH_CHECK(c.size()==3 && std::isfinite(static_cast<float>(c[0])) && c[0]>0 &&
            static_cast<float>(c[0])>0 && std::isfinite(static_cast<float>(c[1])) &&
            std::isfinite(static_cast<float>(c[2])), "Block plan v2: invalid norm config");
    }
    std::array<int,3> roles{};
    for (auto [role,dtype]:specs) {
        TORCH_CHECK(role>=-1 && role<=2 && (dtype==0 || dtype==1), "Block plan v2: invalid value");
        if (role>=0) ++roles[role];
        TORCH_CHECK(role!=0 || dtype==1, "Block plan v2: residual must be FP32");
    }
    TORCH_CHECK(roles[0]==1 && roles[1]==1 && roles[2]==1, "Block plan v2: external value roles");
    // Verify every stage before allocating scratch or resolving pointers.
    for (size_t s=0; s<3; ++s) {
        TORCH_CHECK(!code[s].empty(), "Block plan v2: empty stage");
        std::vector<bool> live;
        for (auto [role,dtype]:specs) live.push_back(role==0 || (s>0 && role==s));
        for (const auto& ins:code[s]) {
            TORCH_CHECK(ins.size()==4, "Block plan v2: instruction arity");
            auto op=ins[0], src=ins[1], dst=ins[2], n=ins[3];
            TORCH_CHECK((op==RMS_NORM || op==NORM_ADD || op==RES_ADD_NORM || op==RES_ADD) &&
                src>=0 && src<specs.size() && dst>=0 && dst<specs.size() &&
                (op==RES_ADD ? n==-1 : n>=0 && n<weights.size()), "Block plan v2: invalid operand");
            TORCH_CHECK(live[src] && src!=dst, "Block plan v2: uninitialized/aliased operand");
            TORCH_CHECK(((op==RMS_NORM || op==RES_ADD_NORM) && specs[dst].first==-1) ||
                ((op==NORM_ADD || op==RES_ADD) && specs[dst].first==0), "Block plan v2: invalid destination");
            TORCH_CHECK(op!=RES_ADD_NORM || specs[src].first!=0, "Block plan v2: residual source alias");
            live[dst]=true;
        }
        auto r=returns[s];
        TORCH_CHECK(s==2 ? r==-1 : (r>=0 && r<specs.size() && live[r] && specs[r].first==-1),
                    "Block plan v2: invalid stage return");
    }
    c10::cuda::CUDAGuard guard(device);
    values.reserve(specs.size()); norms.reserve(weights.size());
    for (auto [role,dtype]:specs) {
        Value v{role, dtype==0 ? at::kHalf : at::kFloat, {}, {}};
        if (role==-1) {
            v.tensor=at::empty({b*q,d}, weights[0].options().dtype(v.dtype));
            v.output=v.tensor.view({b,q,d});
        }
        values.push_back(std::move(v));
    }
    for (size_t i=0; i<weights.size(); ++i)
        norms.push_back({weights[i],static_cast<float>(config[i][0]),
            static_cast<float>(config[i][1]),static_cast<float>(config[i][2])});
    for (auto& v:values) if (v.role>=0) external[v.role]=&v;
    for (size_t s=0; s<3; ++s) {
        for (const auto& ins:code[s]) stages[s].push_back({
            ins[0]==RMS_NORM ? &norm : ins[0]==NORM_ADD ? &norm_add :
                ins[0]==RES_ADD_NORM ? &res_add_norm : &res_add,
            &values[ins[1]], &values[ins[2]], ins[3]>=0 ? &norms[ins[3]] : nullptr, external[0]});
        if (returns[s]>=0) outputs[s]=&values[returns[s]];
    }
}

at::Tensor BC_BlockPlanV2::execute(size_t stage,const at::Tensor& x,const at::Tensor& y) {
    auto valid=[&](const at::Tensor& t,const Value* v) {
        return t.defined() && t.device()==device &&
            t.scalar_type()==v->dtype && t.is_contiguous() &&
            t.sizes()==at::IntArrayRef({batch,query,hidden});
    };
    TORCH_CHECK(valid(x,external[0]), "Block plan v2: invalid residual");
    if (stage) TORCH_CHECK(valid(y,external[stage]), "Block plan v2: invalid boundary output");
    auto aliases=[](const at::Tensor& a,const at::Tensor& b) { return a.is_alias_of(b); };
    if (stage) TORCH_CHECK(!aliases(x,y), "Block plan v2: external alias");
    for (const auto& n:norms) {
        TORCH_CHECK(n.weight.device()==device && n.weight.is_contiguous() &&
            n.weight.sizes()==at::IntArrayRef({hidden}) &&
            (n.weight.scalar_type()==at::kHalf || n.weight.scalar_type()==at::kBFloat16),
            "Block plan v2: changed norm binding");
        TORCH_CHECK(!aliases(x,n.weight) && (!stage || !aliases(y,n.weight)), "Block plan v2: weight alias");
    }
    for (const auto& v:values) if (v.role==-1) {
        TORCH_CHECK(!aliases(x,v.tensor) && (!stage || !aliases(y,v.tensor)), "Block plan v2: scratch alias");
    }
    c10::cuda::CUDAGuard guard(x.device());
    external[0]->tensor=x.view({batch*query,hidden});
    if (stage) external[stage]->tensor=y.view({batch*query,hidden});
    for (const auto& op:stages[stage]) op.kernel(op);
    // Only owned scratch/weights persist between successful stage calls.
    external[0]->tensor=at::Tensor();
    if (stage) external[stage]->tensor=at::Tensor();
    return outputs[stage] ? outputs[stage]->output : at::Tensor();
}
at::Tensor BC_BlockPlanV2::begin(const at::Tensor& x) { return execute(0,x,{}); }
at::Tensor BC_BlockPlanV2::middle(const at::Tensor& x,const at::Tensor& y) { return execute(1,x,y); }
void BC_BlockPlanV2::finish(const at::Tensor& x,const at::Tensor& y) { execute(2,x,y); }
void BC_BlockPlanV2::norm(const Instruction& op) {
    rms_norm(op.src->tensor,op.norm->weight,op.dst->tensor,op.norm->eps,
             op.norm->bias,op.norm->scale,false,false,1);
}
void BC_BlockPlanV2::norm_add(const Instruction& op) {
    rms_norm(op.src->tensor,op.norm->weight,op.dst->tensor,op.norm->eps,
             op.norm->bias,op.norm->scale,false,true,1);
}
void BC_BlockPlanV2::res_add_norm(const Instruction& op) {
    rms_norm_res_in(op.src->tensor,op.norm->weight,op.dst->tensor,op.residual->tensor,
                    op.norm->eps,op.norm->bias,op.norm->scale);
}
void BC_BlockPlanV2::res_add(const Instruction& op) {
    op.dst->tensor.add_(op.src->tensor);
}
