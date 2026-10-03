#include "block_plan.h"
#include "../hc_mix.cuh"
#include "../norm.cuh"
#include "../routing.cuh"
#include <ATen/Functions.h>
#include <c10/cuda/CUDAGuard.h>

BC_BlockPlan::BC_BlockPlan(std::vector<std::vector<at::Tensor>> weights,
    std::vector<int64_t> kinds, std::vector<double> eps,
    std::vector<double> hc_eps, std::vector<int64_t> iterations,
    std::vector<std::vector<std::pair<int64_t,int64_t>>> code,
    std::vector<at::Tensor> r, std::shared_ptr<BC_BlockSparseMLP> m,
    int64_t b, int64_t q)
    : router(std::move(r)), moe(std::move(m)), batch(b), query(q), rows(b*q)
{
    const auto n = weights.size();
    TORCH_CHECK(n && kinds.size()==n && eps.size()==n && hc_eps.size()==n &&
        iterations.size()==n && code.size()==3 && b>=1 && q>=1 && rows<=12,
        "Block plan: invalid specialization");
    TORCH_CHECK(!weights[0].empty() && weights[0][0].defined() && weights[0][0].is_cuda(),
        "Block plan: CUDA weights required");
    const auto device = weights[0][0].device();
    c10::cuda::CUDAGuard guard(device);
    channels = hidden = 0;
    for (size_t i=0; i<n; ++i) {
        auto& w = weights[i];
        TORCH_CHECK((kinds[i]==GR_MIX && w.size()==3 && rows<=5) ||
                    (kinds[i]==HC_MIX && w.size()==4), "Block plan: invalid mix bundle");
        for (const auto& t : w)
            TORCH_CHECK(t.defined() && t.device()==device && t.is_contiguous(),
                        "Block plan: invalid tensor placement/layout");
        int64_t h, d;
        if (kinds[i]==GR_MIX) {
            TORCH_CHECK(w[0].dim()==2 && w[1].dim()==4 && w[2].dim()==1 &&
                w[0].scalar_type()==at::kHalf && w[1].scalar_type()==at::kHalf &&
                w[2].scalar_type()==at::kHalf, "Block plan: invalid gated mix weights");
            h=w[1].size(0); d=w[1].size(1)*4;
            TORCH_CHECK(h==4 && d%8==0 && w[1].size(3)==4 &&
                w[0].size(0)==w[1].size(2)+h && w[0].size(1)==h*d &&
                w[2].numel()==h*d, "Block plan: inconsistent gated mix dimensions");
        } else {
            h=4; d=4096;
            TORCH_CHECK(w[0].scalar_type()==at::kHalf && w[0].sizes()==at::IntArrayRef({24,16384}) &&
                w[1].scalar_type()==at::kFloat && w[1].numel()==24 &&
                w[2].scalar_type()==at::kFloat && w[2].numel()==3 && w[3].numel()==d &&
                (w[3].scalar_type()==at::kHalf || w[3].scalar_type()==at::kBFloat16 ||
                 w[3].scalar_type()==at::kFloat), "Block plan: invalid HC weights");
        }
        TORCH_CHECK(i==0 || (h==channels && d==hidden), "Block plan: inconsistent residual dimensions");
        channels=h; hidden=d;
        Site s; s.w=std::move(w); s.eps=eps[i]; s.hc_eps=hc_eps[i]; s.iterations=iterations[i];
        auto opts=s.w[0].options().dtype(at::kFloat);
        s.post=at::empty({rows,h},opts);
        s.mixed=at::empty({rows,d},opts.dtype(at::kHalf));
        if (kinds[i]==GR_MIX) {
            s.dots=at::empty({rows,s.w[0].size(0)+1,h},opts);
            s.output=s.mixed.view({b,q,d});
        } else {
            s.dots=at::empty({rows,hc_mix_num_chunks(rows,h*d),25},opts);
            s.comb=at::empty({rows,h,h},opts);
            s.normalized=at::empty_like(s.mixed);
            s.output=s.normalized.view({b,q,d});
        }
        sites.push_back(std::move(s));
    }
    if (!router.empty()) {
        TORCH_CHECK(router.size()==7 && moe, "Block plan: invalid router bundle");
        TORCH_CHECK(moe->out_bszn.device()==device && moe->out_bszn.size(0)>=rows &&
            moe->out_bszn.size(1)==hidden, "Block plan: incompatible MoE workspace");
        for (const auto& t:router)
            TORCH_CHECK(t.defined() && t.device()==device && t.is_contiguous(), "Block plan: router placement/layout");
        if (rows>1) for (int i=4;i<7;++i)
            router[i]=at::empty({rows,router[i].size(1)},router[i].options());
    }
    const Kernel kernels[]={&BC_BlockPlan::gr,&BC_BlockPlan::hc,&BC_BlockPlan::apply,
        &BC_BlockPlan::norm,&BC_BlockPlan::route,&BC_BlockPlan::experts};
    std::vector<bool> mixed(n,false), applied(n,false);
    bool routed=false, value=false;
    for (size_t stage=0;stage<3;++stage) {
        if (stage<2 && !code[stage].empty()) {
            auto last=code[stage].back().first;
            TORCH_CHECK(last==GR_MIX || last==RMS_NORM,
                "Block plan: Python boundary must return a mixed/normalized activation");
        }
        if (stage>0) value=true; // Explicit Python boundary supplies the current activation.
        for (auto [op,site]:code[stage]) {
            TORCH_CHECK(op>=GR_MIX && op<=MOE && site>=0 && site<n, "Block plan: invalid instruction");
            if (op==GR_MIX || op==HC_MIX) {
                TORCH_CHECK(op==kinds[site] && !mixed[site], "Block plan: invalid mix ordering");
                mixed[site]=true; value=true;
            } else if (op==APPLY) {
                TORCH_CHECK(mixed[site] && !applied[site] && value, "Block plan: apply without live gates");
                applied[site]=true;
            } else if (op==RMS_NORM) {
                TORCH_CHECK(kinds[site]==HC_MIX && mixed[site] && !applied[site], "Block plan: invalid norm");
            } else if (op==ROUTE) {
                TORCH_CHECK(!router.empty() && value, "Block plan: route without activation"); routed=true;
            } else {
                TORCH_CHECK(moe && routed, "Block plan: MoE without routing");
            }
            stages[stage].push_back({kernels[op],static_cast<size_t>(site)});
        }
    }
    TORCH_CHECK(!stages[0].empty() && !stages[2].empty(), "Block plan: empty boundary");
    for (size_t i=0;i<n;++i) TORCH_CHECK(mixed[i] && applied[i], "Block plan: unconsumed site");
}

at::Tensor BC_BlockPlan::execute(size_t stage,const at::Tensor& x,at::Tensor y) {
    TORCH_CHECK(!stages[stage].empty(), "Block plan: unused Python boundary");
    TORCH_CHECK(x.device()==sites[0].w[0].device() && x.scalar_type()==at::kFloat &&
        x.is_contiguous() && x.sizes()==at::IntArrayRef({batch,query,channels,hidden}),
        "Block plan: invalid residual");
    if (stage) {
        TORCH_CHECK(y.defined() && y.device()==x.device() &&
            y.sizes()==at::IntArrayRef({batch,query,hidden}) &&
            (y.scalar_type()==at::kHalf || y.scalar_type()==at::kFloat),
            "Block plan: invalid boundary output");
        y=y.reshape({rows,hidden}).contiguous();
    }
    c10::cuda::CUDAGuard guard(x.device());
    auto residual=x.view({rows,channels,hidden});
    for (const auto& op:stages[stage]) (this->*op.kernel)(sites[op.site],residual,y);
    return stage<2 ? sites[stages[stage].back().site].output : at::Tensor();
}
at::Tensor BC_BlockPlan::begin(const at::Tensor& x) { return execute(0,x,{}); }
at::Tensor BC_BlockPlan::middle(const at::Tensor& x,const at::Tensor& y) { return execute(1,x,y); }
void BC_BlockPlan::finish(const at::Tensor& x,const at::Tensor& y) { execute(2,x,y); }
void BC_BlockPlan::gr(Site& s,const at::Tensor& x,at::Tensor& y) {
    gr_mix(x,c10::nullopt,s.w[0],s.w[1],s.w[2],s.eps,s.dots,s.post,s.mixed); y=s.mixed;
}
void BC_BlockPlan::hc(Site& s,const at::Tensor& x,at::Tensor& y) {
    hc_mix(x,s.w[0],s.w[1],s.w[2],s.eps,s.hc_eps,s.iterations,s.dots,s.post,s.comb,s.mixed); y=s.mixed;
}
void BC_BlockPlan::apply(Site& s,const at::Tensor& x,at::Tensor& y) {
    hc_apply(x,y,s.post,s.comb.defined() ? c10::optional<at::Tensor>(s.comb) : c10::nullopt,
             c10::nullopt,c10::nullopt);
}
void BC_BlockPlan::norm(Site& s,const at::Tensor&,at::Tensor& y) {
    rms_norm(s.mixed,s.w[3],s.normalized,s.eps,0.0f,1.0f,false,false,1); y=s.normalized;
}
void BC_BlockPlan::route(Site&,const at::Tensor&,at::Tensor& y) {
    routing_std(y,router[0],router[4],router[5],router[6],c10::nullopt,router[1],c10::nullopt,router[2],router[3]);
}
void BC_BlockPlan::experts(Site&,const at::Tensor&,at::Tensor& y) {
    moe->run_bszN(y,router[5],router[6]); y=moe->out_bszn.slice(0,0,rows);
}
