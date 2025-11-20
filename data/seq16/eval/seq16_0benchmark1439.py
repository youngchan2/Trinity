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
    K_ptr,
    K_stride0: tl.constexpr,
    K_stride1: tl.constexpr,
    K_stride2: tl.constexpr,
    Q_ptr,
    Q_stride0: tl.constexpr,
    Q_stride1: tl.constexpr,
    Q_stride2: tl.constexpr,
    V_ptr,
    V_stride0: tl.constexpr,
    V_stride1: tl.constexpr,
    V_stride2: tl.constexpr,
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
    K1 = tl.zeros((M, BLOCK_N), dtype=tl.float32)
    Q1 = tl.zeros((M, BLOCK_N), dtype=tl.float32)
    V1 = tl.zeros((M, BLOCK_N), dtype=tl.float32)

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
        Q1 = ((1 * Q1).to(tl.float32) + tl.dot(temp_0, temp_1).to(tl.float32)).to(tl.float32)
        offset_2 = (k + tl.arange(0, BLOCK_K))[:, None] * WK_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WK_stride1
        mask_2 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_2 = tl.load(WK_ptr + offset_2, mask=mask_2, other=0.0)
        K1 = ((1 * K1).to(tl.float32) + tl.dot(temp_0, temp_2).to(tl.float32)).to(tl.float32)
        offset_3 = (k + tl.arange(0, BLOCK_K))[:, None] * WV_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WV_stride1
        mask_3 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_3 = tl.load(WV_ptr + offset_3, mask=mask_3, other=0.0)
        V1 = ((1 * V1).to(tl.float32) + tl.dot(temp_0, temp_3).to(tl.float32)).to(tl.float32)
    temp_4 = tl.expand_dims(Q1, 1)
    temp_5 = temp_4
    offset_4 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * Q_stride0 + (tl.arange(0, 16))[None, :, None] * Q_stride1 + (tl.arange(0, 64))[None, None, :] * Q_stride2
    tl.store(Q_ptr + offset_4, tl.permute(temp_5, (1, 0, 2)))
    temp_6 = tl.expand_dims(K1, 1)
    temp_7 = temp_6
    offset_5 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * K_stride0 + (tl.arange(0, 16))[None, :, None] * K_stride1 + (tl.arange(0, 64))[None, None, :] * K_stride2
    tl.store(K_ptr + offset_5, tl.permute(temp_7, (1, 0, 2)))
    temp_8 = tl.expand_dims(V1, 1)
    temp_9 = temp_8
    offset_6 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * V_stride0 + (tl.arange(0, 16))[None, :, None] * V_stride1 + (tl.arange(0, 64))[None, None, :] * V_stride2
    tl.store(V_ptr + offset_6, tl.permute(temp_9, (1, 0, 2)))



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_K': 32}),
        triton.Config({'BLOCK_K': 64}),
        triton.Config({'BLOCK_K': 128})
    ], key=[]
)
@triton.jit
def kernel_1(
    K_ptr,
    K_stride0: tl.constexpr,
    K_stride1: tl.constexpr,
    K_stride2: tl.constexpr,
    Q_ptr,
    Q_stride0: tl.constexpr,
    Q_stride1: tl.constexpr,
    Q_stride2: tl.constexpr,
    V_ptr,
    V_stride0: tl.constexpr,
    V_stride1: tl.constexpr,
    V_stride2: tl.constexpr,
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

    # Parallel loop n from 0 to dO2_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    offset_0 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * Q_stride0 + (tl.arange(0, 16))[None, :, None] * Q_stride1 + (tl.arange(0, 64))[None, None, :] * Q_stride2
    temp_0 = tl.load(Q_ptr + offset_0)
    offset_1 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * K_stride0 + (tl.arange(0, 16))[None, :, None] * K_stride1 + (tl.arange(0, 64))[None, None, :] * K_stride2
    temp_1 = tl.load(K_ptr + offset_1)
    temp_2 = tl.permute(temp_1, (0, 2, 1))
    C_exp = tl.exp(tl.dot(temp_0, temp_2))
    offset_2 = (tl.arange(0, 16))[:, None] * dO2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dO2_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_4 = (n_indices < N)[None, :]
    temp_3 = tl.load(dO2_ptr + offset_2, mask=mask_4, other=0.0)
    temp_4 = tl.expand_dims(temp_3, 1)
    temp_5 = tl.permute(temp_4, (1, 0, 2))
    offset_3 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * V_stride0 + (tl.arange(0, 16))[None, :, None] * V_stride1 + (tl.arange(0, 64))[None, None, :] * V_stride2
    temp_6 = tl.load(V_ptr + offset_3)
    temp_7 = tl.permute(temp_6, (0, 2, 1))
    dC_exp = ((tl.dot(temp_5, temp_7) / tl.sum(C_exp, axis=2, dtype=tl.float32)[:, :, None]) + dC_exp)
    temp_9 = tl.expand_dims(temp_3, 1)
    temp_10 = tl.permute(temp_9, (1, 0, 2))
    dV = tl.dot((C_exp / tl.sum(C_exp, axis=2, dtype=tl.float32)[:, :, None]), temp_10)
    dC = (dC + (dC_exp * C_exp))
    dQ = (dQ + tl.dot(dC, temp_1))
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        temp_11 = tl.permute(dC, (0, 2, 1))
        offset_4 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * Q_stride0 + (tl.arange(0, 16))[None, :, None] * Q_stride1 + (tl.arange(0, 64))[None, None, :] * Q_stride2
        temp_12 = tl.load(Q_ptr + offset_4)
        dK = (tl.dot(temp_11, temp_12) + dK)
        temp_13 = tl.permute(dQ, (1, 0, 2))
        # Squeeze dimension 1 from dQ
        temp_14 = tl.reshape(temp_13, (M, D))
        dQ1 = temp_14
        temp_15 = tl.permute(dK, (1, 0, 2))
        # Squeeze dimension 1 from dK
        temp_16 = tl.reshape(temp_15, (M, D))
        dK1 = temp_16
        temp_17 = tl.permute(dV, (1, 0, 2))
        # Squeeze dimension 1 from dV
        temp_18 = tl.reshape(temp_17, (M, D))
        dV1 = temp_18
        offset_5 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_5 = (k_indices < N)[None, :]
        temp_19 = tl.load(X_ptr + offset_5, mask=mask_5, other=0.0)
        temp_20 = tl.trans(temp_19)
        offset_6 = (k + tl.arange(0, BLOCK_K))[:, None] * dWQ_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWQ_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_6 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWQ_ptr + offset_6, tl.dot(temp_20, dQ1).to(tl.float32), mask=mask_6)
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_7 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_7 = (k_indices < N)[None, :]
        temp_21 = tl.load(X_ptr + offset_7, mask=mask_7, other=0.0)
        temp_22 = tl.trans(temp_21)
        offset_8 = (k + tl.arange(0, BLOCK_K))[:, None] * dWK_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWK_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_8 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWK_ptr + offset_8, tl.dot(temp_22, dK1).to(tl.float32), mask=mask_8)
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_9 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_9 = (k_indices < N)[None, :]
        temp_23 = tl.load(X_ptr + offset_9, mask=mask_9, other=0.0)
        temp_24 = tl.trans(temp_23)
        offset_10 = (k + tl.arange(0, BLOCK_K))[:, None] * dWV_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWV_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_10 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWV_ptr + offset_10, tl.dot(temp_24, dV1).to(tl.float32), mask=mask_10)


# Metadata for benchmark.py
TENSOR_PARAMS = ['K', 'Q', 'V', 'WK', 'WQ', 'WV', 'X', 'dO2', 'dWK', 'dWQ', 'dWV']
BLOCK_PARAMS = ['block_k']

def forward(K, Q, V, WK, WQ, WV, X, dO2, dWK, dWQ, dWV, block_k=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[((4544 - 0 + 64 - 1) // 64,)](
        K,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        Q,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        V,
        V.stride(0),
        V.stride(1),
        V.stride(2),
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

    kernel_1[((4544 - 0 + 64 - 1) // 64,)](
        K,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        Q,
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        V,
        V.stride(0),
        V.stride(1),
        V.stride(2),
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
