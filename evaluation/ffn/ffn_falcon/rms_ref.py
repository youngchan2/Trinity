import triton
import triton.language as tl
import torch

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_K': 32}),
        triton.Config({'BLOCK_K': 64}),
        triton.Config({'BLOCK_K': 128})
    ], key=[]
)
@triton.jit
def kernel_0(
    O_FF_ptr,
    O_FF_stride0: tl.constexpr,
    O_FF_stride1: tl.constexpr,
    O_FF1_ptr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr
):
    O_FF1 = tl.zeros((M,), dtype=tl.float32)
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, N, BLOCK_K):
        k_indices = k + tl.arange(0, BLOCK_K)
        mask = k_indices[None, :] < N
        offset_1 = (tl.arange(0, M))[:, None] * O_FF_stride0 + k_indices[None, :] * O_FF_stride1
        temp_1 = tl.load(O_FF_ptr + offset_1, mask=mask, other=0.0)
        O_FF1 += tl.sum((temp_1 * temp_1).to(tl.float32), axis=1)
    
    offset_2 = tl.arange(0, M)
    tl.store(O_FF1_ptr + offset_2, O_FF1.to(tl.float16))



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_K': 32}),
        triton.Config({'BLOCK_K': 64}),
        triton.Config({'BLOCK_K': 128})
    ], key=[]
)
@triton.jit
def kernel_1(
    O_FF_ptr,
    O_FF_stride0: tl.constexpr,
    O_FF_stride1: tl.constexpr,
    O_FF1_ptr,
    O_FF1_stride0: tl.constexpr,
    O_FF_norm_ptr,
    O_FF_norm_stride0: tl.constexpr,
    O_FF_norm_stride1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr
):
    # Parallel loop k from 0 to O_FF_norm_dim1 with tile size BLOCK_K
    # Executed across grid dimension 0
    k = 0 + tl.program_id(0) * BLOCK_K
    
    k_indices = k + tl.arange(0, BLOCK_K)
    mask = k_indices[None, :] < N
    
    offset_0 = (tl.arange(0, M))[:, None] * O_FF_stride0 + k_indices[None, :] * O_FF_stride1
    temp_0 = tl.load(O_FF_ptr + offset_0, mask=mask, other=0.0)
    offset_1 = (tl.arange(0, M)) * O_FF1_stride0
    temp_1 = tl.load(O_FF1_ptr + offset_1)
    offset_2 = (tl.arange(0, M))[:, None] * O_FF_norm_stride0 + k_indices[None, :] * O_FF_norm_stride1
    tl.store(O_FF_norm_ptr + offset_2, (temp_0 / tl.sqrt((temp_1 / N).to(tl.float32))[:, None]), mask=mask)


# Metadata for benchmark.py
TENSOR_PARAMS = ['O_FF', 'O_FF1', 'O_FF_norm']
BLOCK_PARAMS = ['block_k']

def forward(O_FF, O_FF1, O_FF_norm, block_k=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    # Launch kernel_0
    grid = (1,)
    kernel_0[grid](
        O_FF,
        O_FF.stride(0),
        O_FF.stride(1),
        O_FF1,
        # BLOCK_K are provided by autotune,
        # BLOCK_K is automatically set by autotune,
        K=4544,
        M=16,
        N=4544
    )

    # Launch kernel_1
    grid = ((4544 - 0 + block_k - 1) // block_k,)
    kernel_1[grid](
        O_FF,
        O_FF.stride(0),
        O_FF.stride(1),
        O_FF1,
        O_FF1.stride(0),
        O_FF_norm,
        O_FF_norm.stride(0),
        O_FF_norm.stride(1),
        # BLOCK_K are provided by autotune,
        # BLOCK_K is automatically set by autotune,
        K=4544,
        M=16,
        N=4544
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
