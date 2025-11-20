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
    K1_ptr,
    K1_stride0: tl.constexpr,
    K1_stride1: tl.constexpr,
    Q1_ptr,
    Q1_stride0: tl.constexpr,
    Q1_stride1: tl.constexpr,
    V1_ptr,
    V1_stride0: tl.constexpr,
    V1_stride1: tl.constexpr,
    X_ptr,
    X_stride0: tl.constexpr,
    X_stride1: tl.constexpr,
    dO2_ptr,
    dO2_stride0: tl.constexpr,
    dO2_stride1: tl.constexpr,
    dWK_ptr,
    dWK_stride0: tl.constexpr,
    dWK_stride1: tl.constexpr,
    dWQ_ptr,
    dWQ_stride0: tl.constexpr,
    dWQ_stride1: tl.constexpr,
    dWV_ptr,
    dWV_stride0: tl.constexpr,
    dWV_stride1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr
):
    # Allocate intermediate tensors
    dC = tl.zeros((1, M, M), dtype=tl.float32)
    dC_exp = tl.zeros((1, M, M), dtype=tl.float32)
    dK = tl.zeros((1, M, D), dtype=tl.float32)
    dK1 = tl.zeros((M, BLOCK_N), dtype=tl.float32)
    dQ = tl.zeros((1, M, D), dtype=tl.float32)
    dQ1 = tl.zeros((M, BLOCK_N), dtype=tl.float32)
    dV1 = tl.zeros((M, BLOCK_N), dtype=tl.float32)

    # Parallel loop n from 0 to Q1_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    offset_0 = (tl.arange(0, 16))[:, None] * Q1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * Q1_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_0 = (n_indices < N)[None, :]
    temp_0 = tl.load(Q1_ptr + offset_0, mask=mask_0, other=0.0)
    temp_1 = tl.expand_dims(temp_0, 1)
    temp_2 = temp_1
    Q = tl.permute(temp_2, (1, 0, 2))
    offset_1 = (tl.arange(0, 16))[:, None] * K1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * K1_stride1
    mask_1 = (n_indices < N)[None, :]
    temp_3 = tl.load(K1_ptr + offset_1, mask=mask_1, other=0.0)
    temp_4 = tl.expand_dims(temp_3, 1)
    temp_5 = temp_4
    K = tl.permute(temp_5, (1, 0, 2))
    offset_2 = (tl.arange(0, 16))[:, None] * V1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * V1_stride1
    mask_2 = (n_indices < N)[None, :]
    temp_6 = tl.load(V1_ptr + offset_2, mask=mask_2, other=0.0)
    temp_7 = tl.expand_dims(temp_6, 1)
    temp_8 = temp_7
    V = tl.permute(temp_8, (1, 0, 2))
    temp_9 = tl.permute(K, (0, 2, 1))
    C_exp = tl.exp(tl.dot(Q, temp_9))
    offset_3 = (tl.arange(0, 16))[:, None] * dO2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dO2_stride1
    mask_3 = (n_indices < N)[None, :]
    temp_10 = tl.load(dO2_ptr + offset_3, mask=mask_3, other=0.0)
    temp_11 = tl.expand_dims(temp_10, 1)
    temp_12 = tl.permute(temp_11, (1, 0, 2))
    temp_13 = tl.permute(V, (0, 2, 1))
    dC_exp = (dC_exp + (tl.dot(temp_12, temp_13) / tl.sum(C_exp, axis=2, dtype=tl.float32)[:, :, None]))
    temp_15 = tl.expand_dims(temp_10, 1)
    temp_16 = tl.permute(temp_15, (1, 0, 2))
    dV = tl.dot((C_exp / tl.sum(C_exp, axis=2, dtype=tl.float32)[:, :, None]), temp_16)
    dC = ((dC_exp * C_exp) + dC)
    dQ = (dQ + tl.dot(dC, K))
    temp_17 = tl.permute(dC, (0, 2, 1))
    dK = (dK + tl.dot(temp_17, Q))
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        temp_18 = tl.permute(dQ, (1, 0, 2))
        # Squeeze dimension 1 from dQ
        temp_19 = tl.reshape(temp_18, (M, D))
        dQ1 = temp_19
        temp_20 = tl.permute(dK, (1, 0, 2))
        # Squeeze dimension 1 from dK
        temp_21 = tl.reshape(temp_20, (M, D))
        dK1 = temp_21
        temp_22 = tl.permute(dV, (1, 0, 2))
        # Squeeze dimension 1 from dV
        temp_23 = tl.reshape(temp_22, (M, D))
        dV1 = temp_23
        offset_4 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_4 = (k_indices < N)[None, :]
        temp_24 = tl.load(X_ptr + offset_4, mask=mask_4, other=0.0)
        temp_25 = tl.trans(temp_24)
        offset_5 = (k + tl.arange(0, BLOCK_K))[:, None] * dWV_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWV_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_5 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWV_ptr + offset_5, tl.dot(temp_25, dV1).to(tl.float32), mask=mask_5)
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_6 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_6 = (k_indices < N)[None, :]
        temp_26 = tl.load(X_ptr + offset_6, mask=mask_6, other=0.0)
        temp_27 = tl.trans(temp_26)
        offset_7 = (k + tl.arange(0, BLOCK_K))[:, None] * dWQ_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWQ_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_7 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWQ_ptr + offset_7, tl.dot(temp_27, dQ1).to(tl.float32), mask=mask_7)
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_8 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_8 = (k_indices < N)[None, :]
        temp_28 = tl.load(X_ptr + offset_8, mask=mask_8, other=0.0)
        temp_29 = tl.trans(temp_28)
        offset_9 = (k + tl.arange(0, BLOCK_K))[:, None] * dWK_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWK_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_9 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWK_ptr + offset_9, tl.dot(temp_29, dK1).to(tl.float32), mask=mask_9)


# Metadata for benchmark.py
TENSOR_PARAMS = ['K1', 'Q1', 'V1', 'X', 'dO2', 'dWK', 'dWQ', 'dWV']
BLOCK_PARAMS = ['block_k']

def forward(K1, Q1, V1, X, dO2, dWK, dWQ, dWV, block_k=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[((4544 - 0 + 64 - 1) // 64,)](
        K1,
        K1.stride(0),
        K1.stride(1),
        Q1,
        Q1.stride(0),
        Q1.stride(1),
        V1,
        V1.stride(0),
        V1.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        dO2,
        dO2.stride(0),
        dO2.stride(1),
        dWK,
        dWK.stride(0),
        dWK.stride(1),
        dWQ,
        dWQ.stride(0),
        dWQ.stride(1),
        dWV,
        dWV.stride(0),
        dWV.stride(1),
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
