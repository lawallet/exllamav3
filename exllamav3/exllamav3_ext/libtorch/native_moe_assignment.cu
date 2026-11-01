#include "native_moe_assignment.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

namespace {
constexpr int THREADS = 1024;

__device__ __forceinline__ int local_id(int64_t e, int experts) {
    return e >= 0 && e < experts ? (int)e : experts;
}

// A bounded histogram + prefix scan. Integer atomics affect only counts, never
// floating-point values or assignment order. The following kernel packs stably.
__global__ void metadata_kernel(const int64_t* selected, int64_t* counts,
    int64_t* tables, float* output, int experts, int assignments, int outputs) {
    __shared__ int histogram[THREADS];
    __shared__ int warp_sums[32];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    histogram[t] = 0;
    __syncthreads();
    if (t < assignments) atomicAdd(histogram + local_id(selected[t], experts), 1);
    __syncthreads();
    int count = histogram[t], sum = count;
    for (int d = 1; d < 32; d *= 2) {
        int v = __shfl_up_sync(0xffffffff, sum, d);
        if (lane >= d) sum += v;
    }
    if (lane == 31) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        int w = warp_sums[lane];
        for (int d = 1; d < 32; d *= 2) {
            int v = __shfl_up_sync(0xffffffff, w, d);
            if (lane >= d) w += v;
        }
        warp_sums[lane] = w;
    }
    __syncthreads();
    if (t <= experts) {
        int start = sum - count + (warp ? warp_sums[warp - 1] : 0);
        counts[t] = count;
        tables[t] = start;                         // contribution slot base
        tables[experts + 1 + t] = start;           // packed expert start
        tables[2 * (experts + 1) + t] = t < experts && count > 0 ? 1 : 0;
    }
    for (int i = t; i < outputs; i += THREADS) output[i] = 0.0f;
}

__global__ void pack_kernel(const int64_t* selected, const half* weights,
    const int64_t* starts, int64_t* tokens, half* packed_weights, int64_t* inverse,
    int experts, int assignments, int topk) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= assignments) return;
    int e = local_id(selected[a], experts), rank = 0;
    for (int i = 0; i < a; ++i) rank += local_id(selected[i], experts) == e;
    int dst = (int)starts[e] + rank;
    tokens[dst] = a / topk;
    packed_weights[dst] = weights[a];
    inverse[a] = dst;
}

void tensor_check(const at::Tensor& t, at::ScalarType dtype, const at::Tensor& ref) {
    TORCH_CHECK(t.is_cuda() && t.device() == ref.device() && t.scalar_type() == dtype && t.is_contiguous(),
                "native_moe_assign: device, dtype or contiguous layout mismatch");
}
} // namespace

void native_moe_assign(const at::Tensor& selected, const at::Tensor& weights,
    at::Tensor counts, at::Tensor tables, at::Tensor tokens,
    at::Tensor packed_weights, at::Tensor inverse, at::Tensor output) {
    TORCH_CHECK(selected.is_cuda() && selected.dim() == 2, "native_moe_assign: selected must be CUDA [R,K]");
    tensor_check(selected, at::kLong, selected);
    tensor_check(weights, at::kHalf, selected);
    tensor_check(counts, at::kLong, selected);
    tensor_check(tables, at::kLong, selected);
    tensor_check(tokens, at::kLong, selected);
    tensor_check(packed_weights, at::kHalf, selected);
    tensor_check(inverse, at::kLong, selected);
    tensor_check(output, at::kFloat, selected);
    int rows = (int)selected.size(0), topk = (int)selected.size(1);
    int experts = (int)counts.numel() - 1, assignments = rows * topk;
    TORCH_CHECK(rows >= 1 && rows <= 16 && topk >= 1 && topk <= 32 && experts >= 1 && experts <= 512,
                "native_moe_assign: bounded shape exceeded");
    TORCH_CHECK(weights.sizes() == selected.sizes() && counts.dim() == 1 &&
                tables.dim() == 2 && tables.size(0) == 3 && tables.size(1) == experts + 1 &&
                tokens.dim() == 1 && tokens.numel() == assignments &&
                packed_weights.dim() == 1 && packed_weights.numel() == assignments &&
                inverse.dim() == 1 && inverse.numel() == assignments &&
                output.dim() == 2 && output.size(0) == rows && output.size(1) >= 1 && output.size(1) <= 65536,
                "native_moe_assign: buffer shape mismatch");
    // Reject overlapping input/output buffers before any launch. Workspace views
    // are allowed if their addressed byte ranges do not overlap.
    const at::Tensor ts[] = {selected, weights, counts, tables, tokens, packed_weights, inverse, output};
    for (int i = 2; i < 8; ++i) for (int j = 0; j < i; ++j) {
        auto a = (uintptr_t)ts[i].data_ptr(), b = (uintptr_t)ts[j].data_ptr();
        TORCH_CHECK(a + ts[i].nbytes() <= b || b + ts[j].nbytes() <= a,
                    "native_moe_assign: overlapping buffers");
    }
    const c10::cuda::CUDAGuard guard(selected.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    metadata_kernel<<<1, THREADS, 0, stream>>>(selected.data_ptr<int64_t>(), counts.data_ptr<int64_t>(),
        tables.data_ptr<int64_t>(), output.data_ptr<float>(), experts, assignments, (int)output.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    pack_kernel<<<(assignments + 127) / 128, 128, 0, stream>>>(selected.data_ptr<int64_t>(),
        (const half*)weights.data_ptr(), tables.data_ptr<int64_t>(), tokens.data_ptr<int64_t>(),
        (half*)packed_weights.data_ptr(), inverse.data_ptr<int64_t>(), experts, assignments, topk);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
