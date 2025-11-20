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
    dK_ptr,
    dK_stride0: tl.constexpr,
    dK_stride1: tl.constexpr,
    dK_stride2: tl.constexpr,
    dO2_ptr,
    dO2_stride0: tl.constexpr,
    dO2_stride1: tl.constexpr,
    dO_tmp_ptr,
    dO_tmp_stride0: tl.constexpr,
    dO_tmp_stride1: tl.constexpr,
    dO_tmp_stride2: tl.constexpr,
    dQ_ptr,
    dQ_stride0: tl.constexpr,
    dQ_stride1: tl.constexpr,
    dQ_stride2: tl.constexpr,
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
    dC_sum = tl.zeros((1, M), dtype=tl.float32)

    # Initialize kernel accumulators
    dK = tl.zeros((1, 16, 64), dtype=tl.float32)
    dO_tmp = tl.zeros((1, 16, 64), dtype=tl.float32)
    dQ = tl.zeros((1, 16, 64), dtype=tl.float32)
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
        Q1 = (tl.dot(temp_0, temp_1).to(tl.float32) + (1 * Q1).to(tl.float32)).to(tl.float32)
        offset_2 = (k + tl.arange(0, BLOCK_K))[:, None] * WK_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WK_stride1
        mask_2 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_2 = tl.load(WK_ptr + offset_2, mask=mask_2, other=0.0)
        K1 = (tl.dot(temp_0, temp_2).to(tl.float32) + (1 * K1).to(tl.float32)).to(tl.float32)
        offset_3 = (k + tl.arange(0, BLOCK_K))[:, None] * WV_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WV_stride1
        mask_3 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_3 = tl.load(WV_ptr + offset_3, mask=mask_3, other=0.0)
        V1 = (tl.dot(temp_0, temp_3).to(tl.float32) + (1 * V1).to(tl.float32)).to(tl.float32)
    temp_4 = tl.expand_dims(Q1, 1)
    Q = tl.permute(temp_4, (1, 0, 2))
    temp_5 = tl.expand_dims(K1, 1)
    K = tl.permute(temp_5, (1, 0, 2))
    temp_6 = tl.expand_dims(V1, 1)
    V = tl.permute(temp_6, (1, 0, 2))
    temp_7 = tl.permute(K, (0, 2, 1))
    offset_4 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_stride1 + (tl.arange(0, 16))[None, None, :] * C_exp_stride2
    tl.store(C_exp_ptr + offset_4, tl.exp(tl.dot(Q, temp_7).to(tl.float32)))
    offset_5 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_stride1 + (tl.arange(0, 16))[None, None, :] * C_exp_stride2
    temp_8 = tl.load(C_exp_ptr + offset_5)
    C_sum = tl.sum(temp_8, axis=2)
    O = tl.dot(temp_8, V)
    O = (O / C_sum[:, :, None])
    temp_9 = tl.permute(O, (1, 0, 2))
    # Squeeze dimension 1 from O
    temp_10 = tl.reshape(temp_9, (M, D))
    offset_6 = (tl.arange(0, 16))[:, None] * O2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * O2_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_4 = (n_indices < N)[None, :]
    tl.store(O2_ptr + offset_6, temp_10.to(tl.float32), mask=mask_4)
    offset_7 = (tl.arange(0, 16))[:, None] * dO2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dO2_stride1
    mask_5 = (n_indices < N)[None, :]
    temp_11 = tl.load(dO2_ptr + offset_7, mask=mask_5, other=0.0)
    temp_12 = tl.expand_dims(temp_11, 1)
    dO = tl.permute(temp_12, (1, 0, 2))
    dO_tmp = (dO_tmp + ((1 / C_sum[:, :, None]) * dO))
    dC_sum = (dC_sum + (0 - tl.sum((dO * (O / C_sum[:, :, None])), axis=2, dtype=tl.float32)))
    temp_13 = tl.permute(V, (0, 2, 1))
    dC_exp = (tl.dot(dO_tmp, temp_13.to(tl.float32)) + dC_exp)
    dC_exp = (dC_sum[:, :, None] + dC_exp)
    dC = ((temp_8 * dC_exp) + dC)
    dQ = (dQ + tl.dot(dC, K.to(tl.float32)))
    temp_14 = tl.permute(dC, (0, 2, 1))
    dK = (dK + tl.dot(temp_14, Q.to(tl.float32)))
    # Store kernel accumulators
    offset_8 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * dK_stride0 + (tl.arange(0, 16))[None, :, None] * dK_stride1 + (tl.arange(0, 64))[None, None, :] * dK_stride2
    tl.store(dK_ptr + offset_8, dK.to(tl.float32))
    offset_9 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * dO_tmp_stride0 + (tl.arange(0, 16))[None, :, None] * dO_tmp_stride1 + (tl.arange(0, 64))[None, None, :] * dO_tmp_stride2
    tl.store(dO_tmp_ptr + offset_9, dO_tmp.to(tl.float32))
    offset_10 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * dQ_stride0 + (tl.arange(0, 16))[None, :, None] * dQ_stride1 + (tl.arange(0, 64))[None, None, :] * dQ_stride2
    tl.store(dQ_ptr + offset_10, dQ.to(tl.float32))



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_K': 32}),
        triton.Config({'BLOCK_K': 64}),
        triton.Config({'BLOCK_K': 128})
    ], key=[]
)
@triton.jit
def kernel_1(
    C_exp_ptr,
    C_exp_stride0: tl.constexpr,
    C_exp_stride1: tl.constexpr,
    C_exp_stride2: tl.constexpr,
    X_ptr,
    X_stride0: tl.constexpr,
    X_stride1: tl.constexpr,
    dK_ptr,
    dK_stride0: tl.constexpr,
    dK_stride1: tl.constexpr,
    dK_stride2: tl.constexpr,
    dO_tmp_ptr,
    dO_tmp_stride0: tl.constexpr,
    dO_tmp_stride1: tl.constexpr,
    dO_tmp_stride2: tl.constexpr,
    dQ_ptr,
    dQ_stride0: tl.constexpr,
    dQ_stride1: tl.constexpr,
    dQ_stride2: tl.constexpr,
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

    # Parallel loop n from 0 to dQ1_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    offset_0 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_stride1 + (tl.arange(0, 16))[None, None, :] * C_exp_stride2
    temp_0 = tl.load(C_exp_ptr + offset_0)
    temp_1 = tl.permute(temp_0, (0, 2, 1))
    offset_1 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * dO_tmp_stride0 + (tl.arange(0, 16))[None, :, None] * dO_tmp_stride1 + (tl.arange(0, 64))[None, None, :] * dO_tmp_stride2
    temp_2 = tl.load(dO_tmp_ptr + offset_1)
    dV = tl.dot(temp_1, temp_2).to(tl.float32)
    offset_2 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * dQ_stride0 + (tl.arange(0, 16))[None, :, None] * dQ_stride1 + (tl.arange(0, 64))[None, None, :] * dQ_stride2
    temp_3 = tl.load(dQ_ptr + offset_2)
    temp_4 = tl.permute(temp_3, (1, 0, 2))
    # Squeeze dimension 1 from dQ
    temp_5 = tl.reshape(temp_4, (M, D))
    dQ1 = temp_5
    offset_3 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * dK_stride0 + (tl.arange(0, 16))[None, :, None] * dK_stride1 + (tl.arange(0, 64))[None, None, :] * dK_stride2
    temp_6 = tl.load(dK_ptr + offset_3)
    temp_7 = tl.permute(temp_6, (1, 0, 2))
    # Squeeze dimension 1 from dK
    temp_8 = tl.reshape(temp_7, (M, D))
    dK1 = temp_8
    temp_9 = tl.permute(dV, (1, 0, 2))
    # Squeeze dimension 1 from dV
    temp_10 = tl.reshape(temp_9, (M, D))
    dV1 = temp_10
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_4 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_6 = (k_indices < N)[None, :]
        temp_11 = tl.load(X_ptr + offset_4, mask=mask_6, other=0.0)
        temp_12 = tl.trans(temp_11)
        offset_5 = (k + tl.arange(0, BLOCK_K))[:, None] * dWQ_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWQ_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_7 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWQ_ptr + offset_5, tl.dot(temp_12, dQ1).to(tl.float32), mask=mask_7)
        temp_13 = tl.trans(temp_11)
        offset_6 = (k + tl.arange(0, BLOCK_K))[:, None] * dWV_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWV_stride1
        mask_8 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWV_ptr + offset_6, tl.dot(temp_13, dV1).to(tl.float32), mask=mask_8)
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_7 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_9 = (k_indices < N)[None, :]
        temp_14 = tl.load(X_ptr + offset_7, mask=mask_9, other=0.0)
        temp_15 = tl.trans(temp_14)
        offset_8 = (k + tl.arange(0, BLOCK_K))[:, None] * dWK_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWK_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_10 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWK_ptr + offset_8, tl.dot(temp_15, dK1).to(tl.float32), mask=mask_10)


# Metadata for benchmark.py
TENSOR_PARAMS = ['C_exp', 'O2', 'WK', 'WQ', 'WV', 'X', 'dK', 'dO2', 'dO_tmp', 'dQ', 'dWK', 'dWQ', 'dWV']
BLOCK_PARAMS = ['block_k']

def forward(C_exp, O2, WK, WQ, WV, X, dK, dO2, dO_tmp, dQ, dWK, dWQ, dWV, block_k=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[((4544 - 0 + 64 - 1) // 64,)](
        C_exp,
        C_exp.stride(0),
        C_exp.stride(1),
        C_exp.stride(2),
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
        dK,
        dK.stride(0),
        dK.stride(1),
        dK.stride(2),
        dO2,
        dO2.stride(0),
        dO2.stride(1),
        dO_tmp,
        dO_tmp.stride(0),
        dO_tmp.stride(1),
        dO_tmp.stride(2),
        dQ,
        dQ.stride(0),
        dQ.stride(1),
        dQ.stride(2),
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
        C_exp,
        C_exp.stride(0),
        C_exp.stride(1),
        C_exp.stride(2),
        X,
        X.stride(0),
        X.stride(1),
        dK,
        dK.stride(0),
        dK.stride(1),
        dK.stride(2),
        dO_tmp,
        dO_tmp.stride(0),
        dO_tmp.stride(1),
        dO_tmp.stride(2),
        dQ,
        dQ.stride(0),
        dQ.stride(1),
        dQ.stride(2),
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
