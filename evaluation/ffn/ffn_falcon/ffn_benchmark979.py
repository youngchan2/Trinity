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
    X_ptr,
    X_stride0: tl.constexpr,
    X_stride1: tl.constexpr,
    attn_O2_ptr,
    attn_O2_stride0: tl.constexpr,
    attn_O2_stride1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr
):
    # Allocate intermediate tensors
    attn_O1 = tl.zeros((M, BLOCK_N), dtype=tl.float16)

    # Parallel loop n from 0 to attn_O1_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * O2_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * O2_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_0 = (k_indices < K)[None, :]
        temp_0 = tl.load(O2_ptr + offset_0, mask=mask_0, other=0.0)
        offset_1 = (k + tl.arange(0, BLOCK_K))[:, None] * WO_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WO_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_1 = (k_indices < K)[:, None] & (n_indices < N)[None, :]
        temp_1 = tl.load(WO_ptr + offset_1, mask=mask_1, other=0.0)
        attn_O1 = ((attn_O1 * 1).to(tl.float16) + tl.dot(temp_0, temp_1).to(tl.float16)).to(tl.float16)
    offset_2 = (tl.arange(0, 16))[:, None] * X_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * X_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_2 = (n_indices < N)[None, :]
    temp_2 = tl.load(X_ptr + offset_2, mask=mask_2, other=0.0)
    offset_3 = (tl.arange(0, 16))[:, None] * attn_O2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O2_stride1
    tl.store(attn_O2_ptr + offset_3, (attn_O1 + temp_2).to(tl.float16), mask=mask_2)



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
def kernel_1(
    FF1_ptr,
    FF1_stride0: tl.constexpr,
    FF1_stride1: tl.constexpr,
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
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr
):
    # Allocate intermediate tensors
    FF1a = tl.zeros((M, BLOCK_N), dtype=tl.float16)
    FF1b = tl.zeros((M, BLOCK_N), dtype=tl.float16)

    # Initialize kernel accumulators
    attn_O3 = tl.zeros((16,), dtype=tl.float16)
    # Parallel loop n from 0 to FF1a_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * attn_O2_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * attn_O2_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_3 = (k_indices < N)[None, :]
        temp_0 = tl.load(attn_O2_ptr + offset_0, mask=mask_3, other=0.0)
        attn_O3 = ((attn_O3 * 1).to(tl.float16) + tl.sum((temp_0 * temp_0).to(tl.float16), axis=1, dtype=tl.float16)).to(tl.float16)
        offset_1 = (k + tl.arange(0, BLOCK_K))[:, None] * WFF1a_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WFF1a_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_4 = (k_indices < K)[:, None] & (n_indices < N)[None, :]
        temp_1 = tl.load(WFF1a_ptr + offset_1, mask=mask_4, other=0.0)
        FF1a = ((FF1a * 1).to(tl.float16) + tl.dot(temp_0, temp_1).to(tl.float16)).to(tl.float16)
        offset_2 = (k + tl.arange(0, BLOCK_K))[:, None] * WFF1b_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WFF1b_stride1
        temp_2 = tl.load(WFF1b_ptr + offset_2, mask=mask_4, other=0.0)
        FF1b = (tl.dot(temp_0, temp_2).to(tl.float16) + (FF1b * 1).to(tl.float16)).to(tl.float16)
    FF1a = (FF1a / tl.sqrt((attn_O3 / 4544).to(tl.float16).to(tl.float32)).to(tl.float16)[:, None]).to(tl.float16)
    FF1b = (FF1b / tl.sqrt((attn_O3 / 4544).to(tl.float16).to(tl.float32)).to(tl.float16)[:, None]).to(tl.float16)
    offset_3 = (tl.arange(0, 16))[:, None] * FF1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * FF1_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_5 = (n_indices < N)[None, :]
    tl.store(FF1_ptr + offset_3, (FF1a * (FF1b * tl.sigmoid(FF1b.to(tl.float32)).to(tl.float16)).to(tl.float16)).to(tl.float16), mask=mask_5)
    # Store kernel accumulators
    offset_4 = (tl.arange(0, 16)) * attn_O3_stride0
    tl.store(attn_O3_ptr + offset_4, attn_O3)



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
    attn_O_norm_ptr,
    attn_O_norm_stride0: tl.constexpr,
    attn_O_norm_stride1: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr
):
    # Parallel loop k from 0 to attn_O_norm_dim1 with tile size BLOCK_K
    # Executed across grid dimension 0
    k = 0 + tl.program_id(0) * BLOCK_K
    
    offset_0 = (tl.arange(0, 16))[:, None] * attn_O2_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * attn_O2_stride1
    k_indices = k + tl.arange(0, BLOCK_K)
    mask_6 = (k_indices < N)[None, :]
    temp_0 = tl.load(attn_O2_ptr + offset_0, mask=mask_6, other=0.0)
    offset_1 = (tl.arange(0, 16)) * attn_O3_stride0
    temp_1 = tl.load(attn_O3_ptr + offset_1)
    offset_2 = (tl.arange(0, 16))[:, None] * attn_O_norm_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * attn_O_norm_stride1
    tl.store(attn_O_norm_ptr + offset_2, (temp_0 / tl.sqrt((temp_1 / 4544).to(tl.float16).to(tl.float32)).to(tl.float16)[:, None]).to(tl.float16), mask=mask_6)



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
def kernel_3(
    FF1_ptr,
    FF1_stride0: tl.constexpr,
    FF1_stride1: tl.constexpr,
    O_FF_ptr,
    O_FF_stride0: tl.constexpr,
    O_FF_stride1: tl.constexpr,
    WFF2_ptr,
    WFF2_stride0: tl.constexpr,
    WFF2_stride1: tl.constexpr,
    attn_O_norm_ptr,
    attn_O_norm_stride0: tl.constexpr,
    attn_O_norm_stride1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr
):
    # Allocate intermediate tensors
    FF2 = tl.zeros((M, BLOCK_N), dtype=tl.float16)

    # Parallel loop n from 0 to FF2_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * FF1_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * FF1_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_7 = (k_indices < N)[None, :]
        temp_0 = tl.load(FF1_ptr + offset_0, mask=mask_7, other=0.0)
        offset_1 = (k + tl.arange(0, BLOCK_K))[:, None] * WFF2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WFF2_stride1
        n_indices = n + tl.arange(0, BLOCK_N)
        mask_8 = (k_indices < K)[:, None] & (n_indices < N)[None, :]
        temp_1 = tl.load(WFF2_ptr + offset_1, mask=mask_8, other=0.0)
        FF2 = ((1 * FF2).to(tl.float16) + tl.dot(temp_0, temp_1).to(tl.float16)).to(tl.float16)
    offset_2 = (tl.arange(0, 16))[:, None] * attn_O_norm_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O_norm_stride1
    n_indices = n + tl.arange(0, BLOCK_N)
    mask_9 = (n_indices < K)[None, :]
    temp_2 = tl.load(attn_O_norm_ptr + offset_2, mask=mask_9, other=0.0)
    offset_3 = (tl.arange(0, 16))[:, None] * O_FF_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * O_FF_stride1
    tl.store(O_FF_ptr + offset_3, (temp_2 + FF2).to(tl.float16), mask=mask_9)



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_K': 32}),
        triton.Config({'BLOCK_K': 64}),
        triton.Config({'BLOCK_K': 128})
    ], key=[]
)
@triton.jit
def kernel_4(
    O_FF_ptr,
    O_FF_stride0: tl.constexpr,
    O_FF_stride1: tl.constexpr,
    O_FF1_ptr,
    O_FF1_stride0: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr
):
    # Initialize kernel accumulators
    O_FF1 = tl.zeros((16,), dtype=tl.float16)
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_0 = (tl.arange(0, 16))[:, None] * O_FF_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * O_FF_stride1
        k_indices = k + tl.arange(0, BLOCK_K)
        mask_10 = (k_indices < N)[None, :]
        temp_0 = tl.load(O_FF_ptr + offset_0, mask=mask_10, other=0.0)
        O_FF1 = ((O_FF1 * 1).to(tl.float16) + tl.sum((temp_0 * temp_0).to(tl.float16), axis=1, dtype=tl.float16)).to(tl.float16)
    # Store kernel accumulators
    offset_1 = (tl.arange(0, 16)) * O_FF1_stride0
    tl.store(O_FF1_ptr + offset_1, O_FF1)



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_K': 32}),
        triton.Config({'BLOCK_K': 64}),
        triton.Config({'BLOCK_K': 128})
    ], key=[]
)
@triton.jit
def kernel_5(
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
    
    offset_0 = (tl.arange(0, 16))[:, None] * O_FF_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * O_FF_stride1
    k_indices = k + tl.arange(0, BLOCK_K)
    mask_11 = (k_indices < N)[None, :]
    temp_0 = tl.load(O_FF_ptr + offset_0, mask=mask_11, other=0.0)
    offset_1 = (tl.arange(0, 16)) * O_FF1_stride0
    temp_1 = tl.load(O_FF1_ptr + offset_1)
    offset_2 = (tl.arange(0, 16))[:, None] * O_FF_norm_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * O_FF_norm_stride1
    tl.store(O_FF_norm_ptr + offset_2, (temp_0 / tl.sqrt((temp_1 / 4544).to(tl.float16).to(tl.float32)).to(tl.float16)[:, None]).to(tl.float16), mask=mask_11)


# Metadata for benchmark.py
TENSOR_PARAMS = ['FF1', 'O2', 'O_FF', 'O_FF1', 'O_FF_norm', 'WFF1a', 'WFF1b', 'WFF2', 'WO', 'X', 'attn_O2', 'attn_O3', 'attn_O_norm']
BLOCK_PARAMS = ['block_k', 'block_n']

def forward(FF1, O2, O_FF, O_FF1, O_FF_norm, WFF1a, WFF1b, WFF2, WO, X, attn_O2, attn_O3, attn_O_norm, block_k=16, block_n=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    kernel_0[lambda meta: ((4544 - 0 + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],)](
        O2,
        O2.stride(0),
        O2.stride(1),
        WO,
        WO.stride(0),
        WO.stride(1),
        X,
        X.stride(0),
        X.stride(1),
        attn_O2,
        attn_O2.stride(0),
        attn_O2.stride(1),
        # BLOCK_K, BLOCK_N are provided by autotune,
        # BLOCK_N is automatically set by autotune,
        # BLOCK_K is automatically set by autotune,
        K=4544,
        M=16,
        N=4544
    )

    kernel_1[lambda meta: ((4544 - 0 + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],)](
        FF1,
        FF1.stride(0),
        FF1.stride(1),
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
        # BLOCK_K, BLOCK_N are provided by autotune,
        # BLOCK_N is automatically set by autotune,
        # BLOCK_K is automatically set by autotune,
        K=4544,
        M=16,
        N=4544
    )

    kernel_2[lambda meta: ((4544 - 0 + meta["BLOCK_K"] - 1) // meta["BLOCK_K"],)](
        attn_O2,
        attn_O2.stride(0),
        attn_O2.stride(1),
        attn_O3,
        attn_O3.stride(0),
        attn_O_norm,
        attn_O_norm.stride(0),
        attn_O_norm.stride(1),
        # BLOCK_K are provided by autotune,
        # BLOCK_K is automatically set by autotune,
        K=4544,
        M=16,
        N=4544
    )

    kernel_3[lambda meta: ((4544 - 0 + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],)](
        FF1,
        FF1.stride(0),
        FF1.stride(1),
        O_FF,
        O_FF.stride(0),
        O_FF.stride(1),
        WFF2,
        WFF2.stride(0),
        WFF2.stride(1),
        attn_O_norm,
        attn_O_norm.stride(0),
        attn_O_norm.stride(1),
        # BLOCK_K, BLOCK_N are provided by autotune,
        # BLOCK_N is automatically set by autotune,
        # BLOCK_K is automatically set by autotune,
        K=4544,
        M=16,
        N=4544
    )

    kernel_4[(1,)](
        O_FF,
        O_FF.stride(0),
        O_FF.stride(1),
        O_FF1,
        O_FF1.stride(0),
        # BLOCK_K are provided by autotune,
        # BLOCK_K is automatically set by autotune,
        K=4544,
        M=16,
        N=4544
    )

    kernel_5[lambda meta: ((4544 - 0 + meta["BLOCK_K"] - 1) // meta["BLOCK_K"],)](
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
        K=4544,
        M=16,
        N=4544
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
