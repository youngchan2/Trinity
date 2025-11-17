import triton
import triton.language as tl
import torch

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_K': 512}),
        triton.Config({'BLOCK_K': 1024}),
        triton.Config({'BLOCK_K': 2048}),
        triton.Config({'BLOCK_K': 4096})
    ], key=[]
)
@triton.jit
def kernel_0(
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
    
    offset_0 = (tl.arange(0, M))[:, None] * O_FF_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * O_FF_stride1
    temp_0 = tl.load(O_FF_ptr + offset_0)
    offset_1 = (tl.arange(0, M)) * O_FF1_stride0
    temp_1 = tl.load(O_FF1_ptr + offset_1)
    offset_2 = (tl.arange(0, M))[:, None] * O_FF_norm_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * O_FF_norm_stride1
    tl.store(O_FF_norm_ptr + offset_2, (temp_0 / tl.sqrt((temp_1 / 4096).to(tl.float16).to(tl.float32)).to(tl.float16)[:, None]).to(tl.float16))


# Metadata for benchmark.py
TENSOR_PARAMS = ['O_FF', 'O_FF1', 'O_FF_norm']
BLOCK_PARAMS = ['block_k']

def forward(O_FF, O_FF1, O_FF_norm, block_k=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    # Launch kernel_0
    grid = ((4096 - 0 + block_k - 1) // block_k,)
    kernel_0[grid](
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
        K=4096,
        M=16,
        N=4096
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
