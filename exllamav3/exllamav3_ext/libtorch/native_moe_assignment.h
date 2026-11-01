#pragma once
#include <ATen/ATen.h>

// Native IR metadata operation, ABI 1. Stable grouping in original assignment
// order; invalid IDs map to the non-executed sentinel bucket E. This operation
// does not establish an expert kernel's occupancy bound: that is the caller's
// capability contract. No expert arithmetic or routing is performed here.
void native_moe_assign(
    const at::Tensor& selected, const at::Tensor& weights,
    at::Tensor counts, at::Tensor tables, at::Tensor tokens,
    at::Tensor packed_weights, at::Tensor inverse, at::Tensor output);
