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
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr
):
    # Parallel loop n from 0 to attn_O1_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_0 = (tl.arange(0, M))[:, None] * O2_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * O2_stride1
        temp_0 = tl.load(O2_ptr + offset_0)
        offset_1 = (k + tl.arange(0, BLOCK_K))[:, None] * WO_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WO_stride1
        temp_1 = tl.load(WO_ptr + offset_1)
        offset_2 = (tl.arange(0, M))[:, None] * attn_O1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O1_stride1
        temp_2 = tl.load(attn_O1_ptr + offset_2)
        offset_3 = (tl.arange(0, M))[:, None] * attn_O1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O1_stride1
        tl.store(attn_O1_ptr + offset_3, (tl.dot(temp_0, temp_1).to(tl.float16) + (1 * temp_2).to(tl.float16)).to(tl.float16))



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
    attn_O3_ptr,
    attn_O3_stride0: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr
):
    # Sequential loop n from 0 to 4544 with tile size BLOCK_N
    for n in range(0, 4544, BLOCK_N):
        offset_0 = (tl.arange(0, M))[:, None] * attn_O1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O1_stride1
        temp_0 = tl.load(attn_O1_ptr + offset_0)
        offset_1 = (tl.arange(0, M))[:, None] * X_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * X_stride1
        temp_1 = tl.load(X_ptr + offset_1)
        offset_2 = (tl.arange(0, M))[:, None] * attn_O2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O2_stride1
        tl.store(attn_O2_ptr + offset_2, (temp_0 + temp_1).to(tl.float16))
        offset_3 = (tl.arange(0, M)) * attn_O3_stride0
        temp_2 = tl.load(attn_O3_ptr + offset_3)
        offset_4 = (tl.arange(0, M)) * attn_O3_stride0
        tl.store(attn_O3_ptr + offset_4, (tl.sum(((temp_0 + temp_1).to(tl.float16) * (temp_0 + temp_1).to(tl.float16)).to(tl.float16), axis=1, dtype=tl.float16) + (1 * temp_2).to(tl.float16)).to(tl.float16))



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
def kernel_2(
    FF1_ptr,
    FF1_stride0: tl.constexpr,
    FF1_stride1: tl.constexpr,
    FF2_ptr,
    FF2_stride0: tl.constexpr,
    FF2_stride1: tl.constexpr,
    WFF1a_ptr,
    WFF1a_stride0: tl.constexpr,
    WFF1a_stride1: tl.constexpr,
    WFF1b_ptr,
    WFF1b_stride0: tl.constexpr,
    WFF1b_stride1: tl.constexpr,
    WFF2_ptr,
    WFF2_stride0: tl.constexpr,
    WFF2_stride1: tl.constexpr,
    attn_O2_ptr,
    attn_O2_stride0: tl.constexpr,
    attn_O2_stride1: tl.constexpr,
    attn_O3_ptr,
    attn_O3_stride0: tl.constexpr,
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
    FF1a = tl.zeros((M, BLOCK_N), dtype=tl.float16)
    FF1b = tl.zeros((M, BLOCK_N), dtype=tl.float16)

    # Parallel loop n from 0 to attn_O_norm_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    offset_0 = (tl.arange(0, M))[:, None] * attn_O2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O2_stride1
    temp_0 = tl.load(attn_O2_ptr + offset_0)
    offset_1 = (tl.arange(0, M)) * attn_O3_stride0
    temp_1 = tl.load(attn_O3_ptr + offset_1)
    offset_2 = (tl.arange(0, M))[:, None] * attn_O_norm_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O_norm_stride1
    tl.store(attn_O_norm_ptr + offset_2, (temp_0 / tl.sqrt((temp_1 / 4544).to(tl.float16).to(tl.float32)).to(tl.float16)[:, None]).to(tl.float16))
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_3 = (tl.arange(0, M))[:, None] * attn_O_norm_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * attn_O_norm_stride1
        temp_2 = tl.load(attn_O_norm_ptr + offset_3)
        offset_4 = (k + tl.arange(0, BLOCK_K))[:, None] * WFF1a_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WFF1a_stride1
        temp_3 = tl.load(WFF1a_ptr + offset_4)
        FF1a = ((FF1a * 1).to(tl.float16) + tl.dot(temp_2, temp_3).to(tl.float16)).to(tl.float16)
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_5 = (tl.arange(0, M))[:, None] * attn_O_norm_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * attn_O_norm_stride1
        temp_4 = tl.load(attn_O_norm_ptr + offset_5)
        offset_6 = (k + tl.arange(0, BLOCK_K))[:, None] * WFF1b_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WFF1b_stride1
        temp_5 = tl.load(WFF1b_ptr + offset_6)
        FF1b = ((1 * FF1b).to(tl.float16) + tl.dot(temp_4, temp_5).to(tl.float16)).to(tl.float16)
    offset_7 = (tl.arange(0, M))[:, None] * FF1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * FF1_stride1
    tl.store(FF1_ptr + offset_7, (FF1a * (FF1b * tl.sigmoid(FF1b.to(tl.float32)).to(tl.float16)).to(tl.float16)).to(tl.float16))
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_8 = (tl.arange(0, M))[:, None] * FF1_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * FF1_stride1
        temp_6 = tl.load(FF1_ptr + offset_8)
        offset_9 = (k + tl.arange(0, BLOCK_K))[:, None] * WFF2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WFF2_stride1
        temp_7 = tl.load(WFF2_ptr + offset_9)
        offset_10 = (tl.arange(0, M))[:, None] * FF2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * FF2_stride1
        temp_8 = tl.load(FF2_ptr + offset_10)
        offset_11 = (tl.arange(0, M))[:, None] * FF2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * FF2_stride1
        tl.store(FF2_ptr + offset_11, (tl.dot(temp_6, temp_7).to(tl.float16) + (1 * temp_8).to(tl.float16)).to(tl.float16))



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_N': 32}),
        triton.Config({'BLOCK_N': 64}),
        triton.Config({'BLOCK_N': 128})
    ], key=[]
)
@triton.jit
def kernel_3(
    FF2_ptr,
    FF2_stride0: tl.constexpr,
    FF2_stride1: tl.constexpr,
    O_FF_ptr,
    O_FF_stride0: tl.constexpr,
    O_FF_stride1: tl.constexpr,
    O_FF1_ptr,
    O_FF1_stride0: tl.constexpr,
    attn_O_norm_ptr,
    attn_O_norm_stride0: tl.constexpr,
    attn_O_norm_stride1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr
):
    # Sequential loop n from 0 to 4544 with tile size BLOCK_N
    for n in range(0, 4544, BLOCK_N):
        offset_0 = (tl.arange(0, M))[:, None] * attn_O_norm_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O_norm_stride1
        temp_0 = tl.load(attn_O_norm_ptr + offset_0)
        offset_1 = (tl.arange(0, M))[:, None] * FF2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * FF2_stride1
        temp_1 = tl.load(FF2_ptr + offset_1)
        offset_2 = (tl.arange(0, M))[:, None] * O_FF_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * O_FF_stride1
        tl.store(O_FF_ptr + offset_2, (temp_0 + temp_1).to(tl.float16))
        offset_3 = (tl.arange(0, M)) * O_FF1_stride0
        temp_2 = tl.load(O_FF1_ptr + offset_3)
        offset_4 = (tl.arange(0, M)) * O_FF1_stride0
        tl.store(O_FF1_ptr + offset_4, (tl.sum(((temp_0 + temp_1).to(tl.float16) * (temp_0 + temp_1).to(tl.float16)).to(tl.float16), axis=1, dtype=tl.float16) + (1 * temp_2).to(tl.float16)).to(tl.float16))



@triton.autotune(
    configs = [
        triton.Config({'BLOCK_N': 32}),
        triton.Config({'BLOCK_N': 64}),
        triton.Config({'BLOCK_N': 128})
    ], key=[]
)
@triton.jit
def kernel_4(
    O_FF_ptr,
    O_FF_stride0: tl.constexpr,
    O_FF_stride1: tl.constexpr,
    O_FF1_ptr,
    O_FF1_stride0: tl.constexpr,
    O_FF_norm_ptr,
    O_FF_norm_stride0: tl.constexpr,
    O_FF_norm_stride1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr
):
    # Parallel loop n from 0 to O_FF_norm_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    offset_0 = (tl.arange(0, M))[:, None] * O_FF_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * O_FF_stride1
    temp_0 = tl.load(O_FF_ptr + offset_0)
    offset_1 = (tl.arange(0, M)) * O_FF1_stride0
    temp_1 = tl.load(O_FF1_ptr + offset_1)
    offset_2 = (tl.arange(0, M))[:, None] * O_FF_norm_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * O_FF_norm_stride1
    tl.store(O_FF_norm_ptr + offset_2, (temp_0 / tl.sqrt((temp_1 / 4544).to(tl.float16).to(tl.float32)).to(tl.float16)[:, None]).to(tl.float16))


# Metadata for benchmark.py
TENSOR_PARAMS = ['FF1', 'FF2', 'O2', 'O_FF', 'O_FF1', 'O_FF_norm', 'WFF1a', 'WFF1b', 'WFF2', 'WO', 'X', 'attn_O1', 'attn_O2', 'attn_O3', 'attn_O_norm']
BLOCK_PARAMS = ['block_k', 'block_n']

def forward(FF1, FF2, O2, O_FF, O_FF1, O_FF_norm, WFF1a, WFF1b, WFF2, WO, X, attn_O1, attn_O2, attn_O3, attn_O_norm, block_k=16, block_n=16):
    """
    Wrapper function that executes all kernels sequentially.
    """
    # Launch kernel_0
    grid = ((4544 - 0 + block_n - 1) // block_n,)
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
        K=4544,
        M=16,
        N=4544
    )

    # Launch kernel_1
    grid = (1,)
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
        attn_O3,
        attn_O3.stride(0),
        # BLOCK_N are provided by autotune,
        # BLOCK_N is automatically set by autotune,
        K=4544,
        M=16,
        N=4544
    )

    # Launch kernel_2
    grid = ((4544 - 0 + block_n - 1) // block_n,)
    kernel_2[grid](
        FF1,
        FF1.stride(0),
        FF1.stride(1),
        FF2,
        FF2.stride(0),
        FF2.stride(1),
        WFF1a,
        WFF1a.stride(0),
        WFF1a.stride(1),
        WFF1b,
        WFF1b.stride(0),
        WFF1b.stride(1),
        WFF2,
        WFF2.stride(0),
        WFF2.stride(1),
        attn_O2,
        attn_O2.stride(0),
        attn_O2.stride(1),
        attn_O3,
        attn_O3.stride(0),
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

    # Launch kernel_3
    grid = (1,)
    kernel_3[grid](
        FF2,
        FF2.stride(0),
        FF2.stride(1),
        O_FF,
        O_FF.stride(0),
        O_FF.stride(1),
        O_FF1,
        O_FF1.stride(0),
        attn_O_norm,
        attn_O_norm.stride(0),
        attn_O_norm.stride(1),
        # BLOCK_N are provided by autotune,
        # BLOCK_N is automatically set by autotune,
        K=4544,
        M=16,
        N=4544
    )

    # Launch kernel_4
    grid = ((4544 - 0 + block_n - 1) // block_n,)
    kernel_4[grid](
        O_FF,
        O_FF.stride(0),
        O_FF.stride(1),
        O_FF1,
        O_FF1.stride(0),
        O_FF_norm,
        O_FF_norm.stride(0),
        O_FF_norm.stride(1),
        # BLOCK_N are provided by autotune,
        # BLOCK_N is automatically set by autotune,
        K=4544,
        M=16,
        N=4544
    )

    # Return output tensors if needed
    # This depends on your specific use case
    pass
