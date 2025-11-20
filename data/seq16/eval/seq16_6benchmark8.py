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
    C_exp_ptr,
    C_exp_stride0: tl.constexpr,
    C_exp_stride1: tl.constexpr,
    C_exp_stride2: tl.constexpr,
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
    K1 = tl.zeros((M, BLOCK_N), dtype=tl.float32)
    Q1 = tl.zeros((M, BLOCK_N), dtype=tl.float32)
    V1 = tl.zeros((M, BLOCK_N), dtype=tl.float32)
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
    Q = tl.permute(temp_5, (1, 0, 2))
    temp_6 = tl.expand_dims(K1, 1)
    temp_7 = temp_6
    K = tl.permute(temp_7, (1, 0, 2))
    temp_8 = tl.expand_dims(V1, 1)
    temp_9 = temp_8
    V = tl.permute(temp_9, (1, 0, 2))
    offset_4 = (tl.arange(0, 16))[:, None] * dO2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dO2_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_4 = (n_indices < N)[None, :]
    temp_10 = tl.load(dO2_ptr + offset_4, mask=mask_4, other=0.0)
    temp_11 = tl.expand_dims(temp_10, 1)
    temp_12 = tl.permute(temp_11, (1, 0, 2))
    temp_13 = tl.permute(V, (0, 2, 1))
    offset_5 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_stride1 + (tl.arange(0, 16))[None, None, :] * C_exp_stride2
    temp_14 = tl.load(C_exp_ptr + offset_5)
    dC_exp = (dC_exp + (tl.dot(temp_12, temp_13).to(tl.float32) / tl.sum(temp_14, axis=2, dtype=tl.float32)[:, :, None]).to(tl.float32)).to(tl.float32)
    temp_16 = tl.expand_dims(temp_10, 1)
    temp_17 = tl.permute(temp_16, (1, 0, 2))
    dV = tl.dot((temp_14 / tl.sum(temp_14, axis=2, dtype=tl.float32)[:, :, None]).to(tl.float32), temp_17).to(tl.float32)
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_6 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_stride1 + (tl.arange(0, 16))[None, None, :] * C_exp_stride2
        temp_18 = tl.load(C_exp_ptr + offset_6)
        dC = ((dC_exp * temp_18).to(tl.float32) + dC).to(tl.float32)
        dQ = (tl.dot(dC, K).to(tl.float32) + dQ).to(tl.float32)
        temp_19 = tl.permute(dC, (0, 2, 1))
        dK = (tl.dot(temp_19, Q).to(tl.float32) + dK).to(tl.float32)
        temp_20 = tl.permute(dQ, (1, 0, 2))
        # Squeeze dimension 1 from dQ
        temp_21 = tl.reshape(temp_20, (M, D))
        dQ1 = temp_21
        temp_22 = tl.permute(dK, (1, 0, 2))
        # Squeeze dimension 1 from dK
        temp_23 = tl.reshape(temp_22, (M, D))
        dK1 = temp_23
        temp_24 = tl.permute(dV, (1, 0, 2))
        # Squeeze dimension 1 from dV
        temp_25 = tl.reshape(temp_24, (M, D))
        dV1 = temp_25
        offset_7 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_5 = (k_indices < N)[None, :]
        temp_26 = tl.load(X_ptr + offset_7, mask=mask_5, other=0.0)
        temp_27 = tl.trans(temp_26)
        offset_8 = (k + tl.arange(0, BLOCK_K))[:, None] * dWQ_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWQ_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_6 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWQ_ptr + offset_8, tl.dot(temp_27, dQ1).to(tl.float32), mask=mask_6)
        temp_28 = tl.trans(temp_26)
        offset_9 = (k + tl.arange(0, BLOCK_K))[:, None] * dWK_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWK_stride1
        mask_7 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWK_ptr + offset_9, tl.dot(temp_28, dK1).to(tl.float32), mask=mask_7)
        temp_29 = tl.trans(temp_26)
        offset_10 = (k + tl.arange(0, BLOCK_K))[:, None] * dWV_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWV_stride1
        mask_8 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWV_ptr + offset_10, tl.dot(temp_29, dV1).to(tl.float32), mask=mask_8)


# Metadata for benchmark.py
TENSOR_PARAMS = ['C_exp', 'WK', 'WQ', 'WV', 'X', 'dO2', 'dWK', 'dWQ', 'dWV']
BLOCK_PARAMS = ['block_k']

def forward(C_exp, WK, WQ, WV, X, dO2, dWK, dWQ, dWV, block_k=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[((4544 - 0 + 64 - 1) // 64,)](
        C_exp,
        C_exp.stride(0),
        C_exp.stride(1),
        C_exp.stride(2),
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
