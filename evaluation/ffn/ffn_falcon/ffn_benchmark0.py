import triton
import triton.language as tl
import torch

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_N': 32, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128})
    ], key=[]
)
@triton.jit
def kernel_0(
    O2_ptr,
    O2_stride0: tl.constexpr,
    O2_stride1: tl.constexpr,
    WO_ptr,
    WO_stride0: tl.constexpr,
    WO_stride1: tl.constexpr,
    attn_O1_ptr,
    attn_O1_stride0: tl.constexpr,
    attn_O1_stride1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr
):
    # Parallel loop n from 0 to attn_O1_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop k from 0 to 4096 with tile size BLOCK_K
    for k in range(0, 4096, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * O2_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * O2_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_0 = (k_indices < N)[None, :]
        temp_0 = tl.load(O2_ptr + offset_0, mask=mask_0, other=0.0)
        offset_1 = (k + tl.arange(0, BLOCK_K))[:, None] * WO_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WO_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_1 = (k_indices < N)[:, None] & (n_indices < N)[None, :]
        temp_1 = tl.load(WO_ptr + offset_1, mask=mask_1, other=0.0)
        offset_2 = (tl.arange(0, 16))[:, None] * attn_O1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O1_stride1
        mask_2 = (n_indices < N)[None, :]
        temp_2 = tl.load(attn_O1_ptr + offset_2, mask=mask_2, other=0.0)
        offset_3 = (tl.arange(0, 16))[:, None] * attn_O1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O1_stride1
        tl.store(attn_O1_ptr + offset_3, (tl.dot(temp_0, temp_1).to(tl.float16) + (temp_2 * 1).to(tl.float16)).to(tl.float16), mask=mask_2)



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_N': 32}),
        triton.Config({'BLOCK_N': 64}),
        triton.Config({'BLOCK_N': 128})
    ], key=[]
)
@triton.jit
def kernel_1(
    X_ptr,
    X_stride0: tl.constexpr,
    X_stride1: tl.constexpr,
    attn_O1_ptr,
    attn_O1_stride0: tl.constexpr,
    attn_O1_stride1: tl.constexpr,
    attn_O2_ptr,
    attn_O2_stride0: tl.constexpr,
    attn_O2_stride1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr
):
    # Parallel loop n from 0 to attn_O2_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    offset_0 = (tl.arange(0, 16))[:, None] * X_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * X_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_3 = (n_indices < N)[None, :]
    temp_0 = tl.load(X_ptr + offset_0, mask=mask_3, other=0.0)
    offset_1 = (tl.arange(0, 16))[:, None] * attn_O1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O1_stride1
    temp_1 = tl.load(attn_O1_ptr + offset_1, mask=mask_3, other=0.0)
    offset_2 = (tl.arange(0, 16))[:, None] * attn_O2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O2_stride1
    tl.store(attn_O2_ptr + offset_2, (temp_0 + temp_1).to(tl.float16), mask=mask_3)



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_K': 32}),
        triton.Config({'BLOCK_K': 64}),
        triton.Config({'BLOCK_K': 128})
    ], key=[]
)
@triton.jit
def kernel_2(
    attn_O2_ptr,
    attn_O2_stride0: tl.constexpr,
    attn_O2_stride1: tl.constexpr,
    attn_O3_ptr,
    attn_O3_stride0: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr
):
    # Sequential loop k from 0 to 4096 with tile size BLOCK_K
    for k in range(0, 4096, BLOCK_K):
        offset_0 = (tl.arange(0, 16)) * attn_O3_stride0
        temp_0 = tl.load(attn_O3_ptr + offset_0)
        offset_1 = (tl.arange(0, 16))[:, None] * attn_O2_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * attn_O2_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_4 = (k_indices < N)[None, :]
        temp_1 = tl.load(attn_O2_ptr + offset_1, mask=mask_4, other=0.0)
        offset_2 = (tl.arange(0, 16)) * attn_O3_stride0
        tl.store(attn_O3_ptr + offset_2, ((temp_0 * 1).to(tl.float16) + tl.sum((temp_1 * temp_1).to(tl.float16), axis=1, dtype=tl.float16)).to(tl.float16))



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_P': 32, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_P': 32, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_P': 32, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_P': 64, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_P': 64, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_P': 64, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_P': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_P': 128, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_P': 128, 'BLOCK_K': 128})
    ], key=[]
)
@triton.jit
def kernel_3(
    FF1a_ptr,
    FF1a_stride0: tl.constexpr,
    FF1a_stride1: tl.constexpr,
    FF1b_ptr,
    FF1b_stride0: tl.constexpr,
    FF1b_stride1: tl.constexpr,
    WFF1a_ptr,
    WFF1a_stride0: tl.constexpr,
    WFF1a_stride1: tl.constexpr,
    WFF1b_ptr,
    WFF1b_stride0: tl.constexpr,
    WFF1b_stride1: tl.constexpr,
    attn_O2_ptr,
    attn_O2_stride0: tl.constexpr,
    attn_O2_stride1: tl.constexpr,
    attn_O3_ptr,
    attn_O3_stride0: tl.constexpr,
    attn_O_norm_ptr,
    attn_O_norm_stride0: tl.constexpr,
    attn_O_norm_stride1: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr
):
    # Parallel loop p from 0 to FF1b_dim1 with tile size BLOCK_P
    # Executed across grid dimension 0
    p = 0 + tl.program_id(0) * BLOCK_P
    
    # Sequential loop k from 0 to 4096 with tile size BLOCK_K
    for k in range(0, 4096, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * attn_O2_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * attn_O2_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_5 = (k_indices < N)[None, :]
        temp_0 = tl.load(attn_O2_ptr + offset_0, mask=mask_5, other=0.0)
        offset_1 = (tl.arange(0, 16)) * attn_O3_stride0
        temp_1 = tl.load(attn_O3_ptr + offset_1)
        offset_2 = (tl.arange(0, 16))[:, None] * attn_O_norm_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * attn_O_norm_stride1
        tl.store(attn_O_norm_ptr + offset_2, (temp_0 / tl.sqrt((temp_1 / 4096).to(tl.float16).to(tl.float32)).to(tl.float16)[:, None]).to(tl.float16), mask=mask_5)
        offset_3 = (k + tl.arange(0, BLOCK_K))[:, None] * WFF1b_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * WFF1b_stride1
        p_indices = p + tl.arange(0, BLOCK_P)
        mask_6 = (k_indices < N)[:, None] & (p_indices < P)[None, :]
        temp_2 = tl.load(WFF1b_ptr + offset_3, mask=mask_6, other=0.0)
        offset_4 = (tl.arange(0, 16))[:, None] * FF1b_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * FF1b_stride1
        mask_7 = (p_indices < N)[None, :]
        temp_3 = tl.load(FF1b_ptr + offset_4, mask=mask_7, other=0.0)
        offset_5 = (tl.arange(0, 16))[:, None] * FF1b_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * FF1b_stride1
        tl.store(FF1b_ptr + offset_5, (tl.dot((temp_0 / tl.sqrt((temp_1 / 4096).to(tl.float16).to(tl.float32)).to(tl.float16)[:, None]).to(tl.float16), temp_2).to(tl.float16) + (temp_3 * 1).to(tl.float16)).to(tl.float16), mask=mask_7)
        offset_6 = (tl.arange(0, 16))[:, None] * FF1a_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * FF1a_stride1
        temp_4 = tl.load(FF1a_ptr + offset_6, mask=mask_7, other=0.0)
        offset_7 = (k + tl.arange(0, BLOCK_K))[:, None] * WFF1a_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * WFF1a_stride1
        temp_5 = tl.load(WFF1a_ptr + offset_7, mask=mask_6, other=0.0)
        offset_8 = (tl.arange(0, 16))[:, None] * FF1a_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * FF1a_stride1
        tl.store(FF1a_ptr + offset_8, ((temp_4 * 1).to(tl.float16) + tl.dot(temp_0, temp_5).to(tl.float16)).to(tl.float16), mask=mask_7)



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_N': 32, 'BLOCK_P': 32}),
        triton.Config({'BLOCK_N': 32, 'BLOCK_P': 64}),
        triton.Config({'BLOCK_N': 32, 'BLOCK_P': 128}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_P': 32}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_P': 64}),
        triton.Config({'BLOCK_N': 64, 'BLOCK_P': 128}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_P': 32}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_P': 64}),
        triton.Config({'BLOCK_N': 128, 'BLOCK_P': 128})
    ], key=[]
)
@triton.jit
def kernel_4(
    FF1a_ptr,
    FF1a_stride0: tl.constexpr,
    FF1a_stride1: tl.constexpr,
    FF1b_ptr,
    FF1b_stride0: tl.constexpr,
    FF1b_stride1: tl.constexpr,
    O_FF_ptr,
    O_FF_stride0: tl.constexpr,
    O_FF_stride1: tl.constexpr,
    WFF2_ptr,
    WFF2_stride0: tl.constexpr,
    WFF2_stride1: tl.constexpr,
    attn_O3_ptr,
    attn_O3_stride0: tl.constexpr,
    attn_O_norm_ptr,
    attn_O_norm_stride0: tl.constexpr,
    attn_O_norm_stride1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr
):
    # Allocate intermediate tensors
    FF2 = tl.zeros((M, BLOCK_N), dtype=tl.float16)

    # Parallel loop n from 0 to FF2_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop p from 0 to 14336 with tile size BLOCK_P
    for p in range(0, 14336, BLOCK_P):
        offset_0 = (tl.arange(0, 16))[:, None] * FF1a_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * FF1a_stride1
        p_indices = p + tl.arange(0, BLOCK_P)
        mask_8 = (p_indices < N)[None, :]
        temp_0 = tl.load(FF1a_ptr + offset_0, mask=mask_8, other=0.0)
        offset_1 = (tl.arange(0, 16)) * attn_O3_stride0
        temp_1 = tl.load(attn_O3_ptr + offset_1)
        offset_2 = (tl.arange(0, 16))[:, None] * FF1a_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * FF1a_stride1
        tl.store(FF1a_ptr + offset_2, (temp_0 / tl.sqrt((temp_1 / 4096).to(tl.float16).to(tl.float32)).to(tl.float16)[:, None]).to(tl.float16), mask=mask_8)
        offset_3 = (tl.arange(0, 16))[:, None] * FF1b_stride0 + (p + tl.arange(0, BLOCK_P))[None, :] * FF1b_stride1
        temp_2 = tl.load(FF1b_ptr + offset_3, mask=mask_8, other=0.0)
        offset_4 = (p + tl.arange(0, BLOCK_P))[:, None] * WFF2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WFF2_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_9 = (p_indices < P)[:, None] & (n_indices < N)[None, :]
        temp_3 = tl.load(WFF2_ptr + offset_4, mask=mask_9, other=0.0)
        FF2 = ((FF2 * 1).to(tl.float16) + tl.dot(((tl.sigmoid(temp_2.to(tl.float32)).to(tl.float16) * temp_2).to(tl.float16) * temp_0).to(tl.float16), temp_3).to(tl.float16)).to(tl.float16)
    offset_5 = (tl.arange(0, 16))[:, None] * attn_O_norm_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O_norm_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_10 = (n_indices < N)[None, :]
    temp_4 = tl.load(attn_O_norm_ptr + offset_5, mask=mask_10, other=0.0)
    offset_6 = (tl.arange(0, 16))[:, None] * O_FF_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * O_FF_stride1
    tl.store(O_FF_ptr + offset_6, (FF2 + temp_4).to(tl.float16), mask=mask_10)


# Metadata for benchmark.py
TENSOR_PARAMS = ['FF1a', 'FF1b', 'O2', 'O_FF', 'WFF1a', 'WFF1b', 'WFF2', 'WO', 'X', 'attn_O1', 'attn_O2', 'attn_O3', 'attn_O_norm']
BLOCK_PARAMS = ['block_k', 'block_n', 'block_p']

def forward(FF1a, FF1b, O2, O_FF, WFF1a, WFF1b, WFF2, WO, X, attn_O1, attn_O2, attn_O3, attn_O_norm, block_k=16, block_n=16, block_p=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    # Launch kernel_0
    grid = ((4096 - 0 + block_n - 1) // block_n,)
    kernel_0[grid](
        O2,
        O2.stride(0),
        O2.stride(1),
        WO,
        WO.stride(0),
        WO.stride(1),
        attn_O1,
        attn_O1.stride(0),
        attn_O1.stride(1),
        # BLOCK_K, BLOCK_N are provided by autotune,
        # BLOCK_N is automatically set by autotune,
        # BLOCK_K is automatically set by autotune,
        M=16,
        N=4096,
        P=14336
    )

    # Launch kernel_1
    grid = ((4096 - 0 + block_n - 1) // block_n,)
    kernel_1[grid](
        X,
        X.stride(0),
        X.stride(1),
        attn_O1,
        attn_O1.stride(0),
        attn_O1.stride(1),
        attn_O2,
        attn_O2.stride(0),
        attn_O2.stride(1),
        # BLOCK_N are provided by autotune,
        # BLOCK_N is automatically set by autotune,
        M=16,
        N=4096,
        P=14336
    )

    # Launch kernel_2
    grid = (1,)
    kernel_2[grid](
        attn_O2,
        attn_O2.stride(0),
        attn_O2.stride(1),
        attn_O3,
        attn_O3.stride(0),
        # BLOCK_K are provided by autotune,
        # BLOCK_K is automatically set by autotune,
        M=16,
        N=4096,
        P=14336
    )

    # Launch kernel_3
    grid = ((14336 - 0 + block_p - 1) // block_p,)
    kernel_3[grid](
        FF1a,
        FF1a.stride(0),
        FF1a.stride(1),
        FF1b,
        FF1b.stride(0),
        FF1b.stride(1),
        WFF1a,
        WFF1a.stride(0),
        WFF1a.stride(1),
        WFF1b,
        WFF1b.stride(0),
        WFF1b.stride(1),
        attn_O2,
        attn_O2.stride(0),
        attn_O2.stride(1),
        attn_O3,
        attn_O3.stride(0),
        attn_O_norm,
        attn_O_norm.stride(0),
        attn_O_norm.stride(1),
        # BLOCK_K, BLOCK_P are provided by autotune,
        # BLOCK_P is automatically set by autotune,
        # BLOCK_K is automatically set by autotune,
        M=16,
        N=4096,
        P=14336
    )

    # Launch kernel_4
    grid = ((4096 - 0 + block_n - 1) // block_n,)
    kernel_4[grid](
        FF1a,
        FF1a.stride(0),
        FF1a.stride(1),
        FF1b,
        FF1b.stride(0),
        FF1b.stride(1),
        O_FF,
        O_FF.stride(0),
        O_FF.stride(1),
        WFF2,
        WFF2.stride(0),
        WFF2.stride(1),
        attn_O3,
        attn_O3.stride(0),
        attn_O_norm,
        attn_O_norm.stride(0),
        attn_O_norm.stride(1),
        # BLOCK_N, BLOCK_P are provided by autotune,
        # BLOCK_N is automatically set by autotune,
        # BLOCK_P is automatically set by autotune,
        M=16,
        N=4096,
        P=14336
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
