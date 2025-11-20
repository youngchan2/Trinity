import triton
import triton.language as tl
import torch
@triton.jit
def kernel_0(
    C_exp_ptr,
    C_exp_stride0: tl.constexpr,
    C_exp_stride1: tl.constexpr,
    C_exp_stride2: tl.constexpr,
    K1_ptr,
    K1_stride0: tl.constexpr,
    K1_stride1: tl.constexpr,
    Q1_ptr,
    Q1_stride0: tl.constexpr,
    Q1_stride1: tl.constexpr,
    V1_ptr,
    V1_stride0: tl.constexpr,
    V1_stride1: tl.constexpr,
    dK_ptr,
    dK_stride0: tl.constexpr,
    dK_stride1: tl.constexpr,
    dK_stride2: tl.constexpr,
    dO2_ptr,
    dO2_stride0: tl.constexpr,
    dO2_stride1: tl.constexpr,
    dQ_ptr,
    dQ_stride0: tl.constexpr,
    dQ_stride1: tl.constexpr,
    dQ_stride2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr
):
    # Allocate intermediate tensors
    dC = tl.zeros((1, M, M), dtype=tl.float32)
    dC_exp = tl.zeros((1, M, M), dtype=tl.float32)

    # Initialize kernel accumulators
    dK = tl.zeros((1, 16, 64), dtype=tl.float32)
    dQ = tl.zeros((1, 16, 64), dtype=tl.float32)
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
    offset_3 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_stride1 + (tl.arange(0, 16))[None, None, :] * C_exp_stride2
    tl.store(C_exp_ptr + offset_3, tl.exp(tl.dot(Q, temp_9)))
    offset_4 = (tl.arange(0, 16))[:, None] * dO2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dO2_stride1
    mask_3 = (n_indices < N)[None, :]
    temp_10 = tl.load(dO2_ptr + offset_4, mask=mask_3, other=0.0)
    temp_11 = tl.expand_dims(temp_10, 1)
    temp_12 = tl.permute(temp_11, (1, 0, 2))
    temp_13 = tl.permute(V, (0, 2, 1))
    offset_5 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_stride1 + (tl.arange(0, 16))[None, None, :] * C_exp_stride2
    temp_14 = tl.load(C_exp_ptr + offset_5)
    dC_exp = (dC_exp + (tl.dot(temp_12, temp_13) / tl.sum(temp_14, axis=2, dtype=tl.float32)[:, :, None]))
    dC = (dC + (dC_exp * temp_14))
    dQ = (dQ + tl.dot(dC, K.to(tl.float32)))
    temp_16 = tl.permute(dC, (0, 2, 1))
    dK = (tl.dot(temp_16, Q.to(tl.float32)) + dK)
    # Store kernel accumulators
    offset_6 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * dK_stride0 + (tl.arange(0, 16))[None, :, None] * dK_stride1 + (tl.arange(0, 64))[None, None, :] * dK_stride2
    tl.store(dK_ptr + offset_6, dK.to(tl.float32))
    offset_7 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * dQ_stride0 + (tl.arange(0, 16))[None, :, None] * dQ_stride1 + (tl.arange(0, 64))[None, None, :] * dQ_stride2
    tl.store(dQ_ptr + offset_7, dQ.to(tl.float32))



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
    dO2_ptr,
    dO2_stride0: tl.constexpr,
    dO2_stride1: tl.constexpr,
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

    # Parallel loop n from 0 to dO2_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    offset_0 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * C_exp_stride0 + (tl.arange(0, 16))[None, :, None] * C_exp_stride1 + (tl.arange(0, 16))[None, None, :] * C_exp_stride2
    temp_0 = tl.load(C_exp_ptr + offset_0)
    offset_1 = (tl.arange(0, 16))[:, None] * dO2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dO2_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_4 = (n_indices < N)[None, :]
    temp_1 = tl.load(dO2_ptr + offset_1, mask=mask_4, other=0.0)
    temp_2 = tl.expand_dims(temp_1, 1)
    temp_3 = tl.permute(temp_2, (1, 0, 2))
    dV = tl.dot((temp_0 / tl.sum(temp_0, axis=2, dtype=tl.float32)[:, :, None]).to(tl.float32), temp_3).to(tl.float32)
    offset_2 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * dQ_stride0 + (tl.arange(0, 16))[None, :, None] * dQ_stride1 + (tl.arange(0, 64))[None, None, :] * dQ_stride2
    temp_4 = tl.load(dQ_ptr + offset_2)
    temp_5 = tl.permute(temp_4, (1, 0, 2))
    # Squeeze dimension 1 from dQ
    temp_6 = tl.reshape(temp_5, (M, D))
    dQ1 = temp_6
    offset_3 = (((n // BLOCK_N)+tl.arange(0, 1)))[:, None, None] * dK_stride0 + (tl.arange(0, 16))[None, :, None] * dK_stride1 + (tl.arange(0, 64))[None, None, :] * dK_stride2
    temp_7 = tl.load(dK_ptr + offset_3)
    temp_8 = tl.permute(temp_7, (1, 0, 2))
    # Squeeze dimension 1 from dK
    temp_9 = tl.reshape(temp_8, (M, D))
    dK1 = temp_9
    temp_10 = tl.permute(dV, (1, 0, 2))
    # Squeeze dimension 1 from dV
    temp_11 = tl.reshape(temp_10, (M, D))
    dV1 = temp_11
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_4 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_5 = (k_indices < N)[None, :]
        temp_12 = tl.load(X_ptr + offset_4, mask=mask_5, other=0.0)
        temp_13 = tl.trans(temp_12)
        offset_5 = (k + tl.arange(0, BLOCK_K))[:, None] * dWQ_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWQ_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_6 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWQ_ptr + offset_5, tl.dot(temp_13, dQ1).to(tl.float32), mask=mask_6)
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_6 = (tl.arange(0, 16))[:, None] * X_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * X_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_7 = (k_indices < N)[None, :]
        temp_14 = tl.load(X_ptr + offset_6, mask=mask_7, other=0.0)
        temp_15 = tl.trans(temp_14)
        offset_7 = (k + tl.arange(0, BLOCK_K))[:, None] * dWK_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWK_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_8 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWK_ptr + offset_7, tl.dot(temp_15, dK1).to(tl.float32), mask=mask_8)
        temp_16 = tl.trans(temp_14)
        offset_8 = (k + tl.arange(0, BLOCK_K))[:, None] * dWV_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * dWV_stride1
        mask_9 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        tl.store(dWV_ptr + offset_8, tl.dot(temp_16, dV1).to(tl.float32), mask=mask_9)


# Metadata for benchmark.py
TENSOR_PARAMS = ['C_exp', 'K1', 'Q1', 'V1', 'X', 'dK', 'dO2', 'dQ', 'dWK', 'dWQ', 'dWV']
BLOCK_PARAMS = ['block_k']

def forward(C_exp, K1, Q1, V1, X, dK, dO2, dQ, dWK, dWQ, dWV, block_k=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[((4544 - 0 + 64 - 1) // 64,)](
        C_exp,
        C_exp.stride(0),
        C_exp.stride(1),
        C_exp.stride(2),
        K1,
        K1.stride(0),
        K1.stride(1),
        Q1,
        Q1.stride(0),
        Q1.stride(1),
        V1,
        V1.stride(0),
        V1.stride(1),
        dK,
        dK.stride(0),
        dK.stride(1),
        dK.stride(2),
        dO2,
        dO2.stride(0),
        dO2.stride(1),
        dQ,
        dQ.stride(0),
        dQ.stride(1),
        dQ.stride(2),
        BLOCK_N=64,
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
        dO2,
        dO2.stride(0),
        dO2.stride(1),
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
