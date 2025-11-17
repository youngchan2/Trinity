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
        offset_0 = (tl.arange(0, M))[:, None] * O2_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * O2_stride1
        temp_0 = tl.load(O2_ptr + offset_0)
        offset_1 = (k + tl.arange(0, BLOCK_K))[:, None] * WO_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WO_stride1
        temp_1 = tl.load(WO_ptr + offset_1)
        attn_O1 = ((attn_O1 * 1).to(tl.float16) + tl.dot(temp_0, temp_1).to(tl.float16)).to(tl.float16)
    offset_2 = (tl.arange(0, M))[:, None] * X_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * X_stride1
    temp_2 = tl.load(X_ptr + offset_2)
    offset_3 = (tl.arange(0, M))[:, None] * attn_O2_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * attn_O2_stride1
    tl.store(attn_O2_ptr + offset_3, (attn_O1 + temp_2).to(tl.float16))



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

    # Parallel loop n from 0 to FF1a_dim1 with tile size BLOCK_N
    # Executed across grid dimension 0
    n = 0 + tl.program_id(0) * BLOCK_N
    
    # Sequential loop k from 0 to 4544 with tile size BLOCK_K
    for k in range(0, 4544, BLOCK_K):
        offset_0 = (tl.arange(0, M)) * attn_O3_stride0
        temp_0 = tl.load(attn_O3_ptr + offset_0)
        offset_1 = (tl.arange(0, M))[:, None] * attn_O2_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * attn_O2_stride1
        temp_1 = tl.load(attn_O2_ptr + offset_1)
        offset_2 = (tl.arange(0, M)) * attn_O3_stride0
        tl.store(attn_O3_ptr + offset_2, ((temp_0 * 1).to(tl.float16) + tl.sum((temp_1 * temp_1).to(tl.float16), axis=1, dtype=tl.float16)).to(tl.float16))
        offset_3 = (k + tl.arange(0, BLOCK_K))[:, None] * WFF1a_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WFF1a_stride1
        temp_2 = tl.load(WFF1a_ptr + offset_3)
        FF1a = ((FF1a * 1).to(tl.float16) + tl.dot(temp_1, temp_2).to(tl.float16)).to(tl.float16)
        offset_4 = (k + tl.arange(0, BLOCK_K))[:, None] * WFF1b_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * WFF1b_stride1
        temp_3 = tl.load(WFF1b_ptr + offset_4)
        FF1b = (tl.dot(temp_1, temp_3).to(tl.float16) + (FF1b * 1).to(tl.float16)).to(tl.float16)
    offset_5 = (tl.arange(0, M)) * attn_O3_stride0
    temp_4 = tl.load(attn_O3_ptr + offset_5)
    FF1a = (FF1a / tl.sqrt((temp_4 / 4544).to(tl.float16).to(tl.float32)).to(tl.float16)[:, None]).to(tl.float16)
    FF1b = (FF1b / tl.sqrt((temp_4 / 4544).to(tl.float16).to(tl.float32)).to(tl.float16)[:, None]).to(tl.float16)
    offset_6 = (tl.arange(0, M))[:, None] * FF1_stride0 + (n + tl.arange(0, BLOCK_N))[None, :] * FF1_stride1
    tl.store(FF1_ptr + offset_6, (FF1a * (FF1b * tl.sigmoid(FF1b.to(tl.float32)).to(tl.float16)).to(tl.float16)).to(tl.float16))



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
    
    offset_0 = (tl.arange(0, M))[:, None] * attn_O2_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * attn_O2_stride1
    temp_0 = tl.load(attn_O2_ptr + offset_0)
    offset_1 = (tl.arange(0, M)) * attn_O3_stride0
    temp_1 = tl.load(attn_O3_ptr + offset_1)
    offset_2 = (tl.arange(0, M))[:, None] * attn_O_norm_stride0 + (k + tl.arange(0, BLOCK_K))[None, :] * attn_O_norm_stride1
    tl.store(attn_O_norm_ptr + offset_2, (temp_0 / tl.sqrt((temp_1 / 4544).to(tl.float16).to(tl.float32)).to(tl.float16)[:, None]).to(tl.float16))