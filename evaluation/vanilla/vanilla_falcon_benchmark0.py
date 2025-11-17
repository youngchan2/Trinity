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
    O2_ptr,
    O2_stride0: tl.constexpr,
    O2_stride1: tl.constexpr,
    WK_ptr,
    WK_stride0: tl.constexpr,
    WK_stride1: tl.constexpr,
    WQ_ptr,
    WQ_stride0: tl.constexpr,
    WQ_stride1: tl.constexpr,
    WV_ptr,
    WV_stride0: tl.constexpr,
    WV_stride1: tl.constexpr,
    X_ptr,
    X_stride0: tl.constexpr,
    X_stride1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr
):
    # Allocate intermediate tensors
    K1 = tl.zeros((M, BLOCK_N), dtype=tl.float16)
    Q1 = tl.zeros((M, BLOCK_N), dtype=tl.float16)
    V1 = tl.zeros((M, BLOCK_N), dtype=tl.float16)

    # Parallel loop n from 0 to Q1_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_0 = (k_indices < N)[None, :]
        temp_0 = tl.load(X_ptr + offset_0, mask=mask_0, other=0.0)
        offset_1 = (k + tl.arange(0, BLOCK_K))[:, None] * WQ_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WQ_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_1 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_1 = tl.load(WQ_ptr + offset_1, mask=mask_1, other=0.0)
        Q1 = (tl.dot(temp_0, temp_1).to(tl.float16) + (1 * Q1).to(tl.float16)).to(tl.float16)
        offset_2 = (k + tl.arange(0, BLOCK_K))[:, None] * WK_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WK_stride1
        mask_2 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_2 = tl.load(WK_ptr + offset_2, mask=mask_2, other=0.0)
        K1 = (tl.dot(temp_0, temp_2).to(tl.float16) + (1 * K1).to(tl.float16)).to(tl.float16)
        offset_3 = (k + tl.arange(0, BLOCK_K))[:, None] * WV_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WV_stride1
        mask_3 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_3 = tl.load(WV_ptr + offset_3, mask=mask_3, other=0.0)
        V1 = (tl.dot(temp_0, temp_3).to(tl.float16) + (1 * V1).to(tl.float16)).to(tl.float16)
    temp_4 = tl.expand_dims(Q1, 1)
    Q = tl.permute(temp_4, (1, 0, 2))
    temp_5 = tl.expand_dims(K1, 1)
    K = tl.permute(temp_5, (1, 0, 2))
    temp_6 = tl.expand_dims(V1, 1)
    V = tl.permute(temp_6, (1, 0, 2))
    temp_7 = tl.permute(K, (0, 2, 1))
    C_exp = tl.exp(tl.dot(Q, temp_7).to(tl.float32))
    C_sum = tl.sum(C_exp, axis=2)
    O = tl.dot(C_exp, V.to(tl.float32))
    O = (O / C_sum[:, :, None])
    temp_8 = tl.permute(O, (1, 0, 2))
    offset_4 = (tl.arange(0, 16))[:, None] * O2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * O2_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_4 = (n_indices < N)[None, :]
    tl.store(O2_ptr + offset_4, tl.reshape(temp_8, (M, D)).to(tl.float16), mask=mask_4)


# Metadata for benchmark.py
TENSOR_PARAMS = ['O2', 'WK', 'WQ', 'WV', 'X']
BLOCK_PARAMS = ['block_k']

def forward(O2, WK, WQ, WV, X, block_k=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[((4544 - 0 + 64 - 1) // 64,)](
        O2,
        O2.stride(0),
        O2.stride(1),
        WK,
        WK.stride(0),
        WK.stride(1),
        WQ,
        WQ.stride(0),
        WQ.stride(1),
        WV,
        WV.stride(0),
        WV.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        # BLOCK_K are provided by autotune,
        BLOCK_N=64,
        # BLOCK_K is automatically set by autotune,
        D=64,
        H=71,
        M=16,
        N=4544,
        P=1024
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
