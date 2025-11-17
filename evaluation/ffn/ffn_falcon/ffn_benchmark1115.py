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
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr
):
    # Allocate intermediate tensors
    O_FF1 = tl.zeros((M,), dtype=tl.float16)

    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_0 = (tl.arange(0, M))[:, None] * O_FF_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * O_FF_stride1
        temp_0 = tl.load(O_FF_ptr + offset_0)
        O_FF1 = ((O_FF1 * 1).to(tl.float16) + tl.sum((temp_0 * temp_0).to(tl.float16), axis=1, dtype=tl.float16)).to(tl.float16)
    offset_out = tl.arange(0, M)
    tl.store(O_FF1_ptr + offset_out, O_FF1)


# Metadata for benchmark.py
TENSOR_PARAMS = ['O_FF', 'O_FF1']
BLOCK_PARAMS = ['block_k']

def forward(O_FF, O_FF1, block_k=16):
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
        K=4096,
        M=16,
        N=4096
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
