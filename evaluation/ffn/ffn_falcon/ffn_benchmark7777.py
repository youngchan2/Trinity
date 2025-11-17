import triton
import triton.language as tl
import torch
@triton.jit
def kernel_0(
    O_FF_ptr,
    O_FF_stride0: tl.constexpr,
    O_FF_stride1: tl.constexpr,
    O_FF_norm_ptr,
    O_FF_norm_stride0: tl.constexpr,
    O_FF_norm_stride1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr
):
    # Allocate intermediate tensors
    O_FF1 = tl.zeros((M,), dtype=tl.float16)

    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * O_FF_stride0 + (tl.arange(0, 8192))[None, :] * O_FF_stride1
        fulltile_1_indices = tl.arange(0, 8192)
        mask_0 = (fulltile_1_indices < N)[None, :]
        temp_0 = tl.load(O_FF_ptr + offset_0, mask=mask_0, other=0.0)
        O_FF1 = tl.sum((temp_0 * temp_0).to(tl.float16), axis=1, dtype=tl.float16)
        offset_1 = (tl.arange(0, 16))[:, None] * O_FF_norm_stride0 + (tl.arange(0, 8192))[None, :] * O_FF_norm_stride1
        tl.store(O_FF_norm_ptr + offset_1, (temp_0 / tl.sqrt((O_FF1 / 4544).to(tl.float16).to(tl.float32)).to(tl.float16)[:, None]).to(tl.float16), mask=mask_0)


# Metadata for benchmark.py
TENSOR_PARAMS = ['O_FF', 'O_FF_norm']
BLOCK_PARAMS = []

def forward(O_FF, O_FF_norm):
    """
    Wrapper function that executes all kernels sequentially.
    """
    # Launch kernel_0
    grid = (1,)
    kernel_0[grid](
        O_FF,
        O_FF.stride(0),
        O_FF.stride(1),
        O_FF_norm,
        O_FF_norm.stride(0),
        O_FF_norm.stride(1),
        BLOCK_K=4544,
        K=4544,
        M=16,
        N=4544
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
