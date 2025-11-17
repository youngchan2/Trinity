from convert_module import convert_ir_to_triton
from ffn_ref import TRT_Llama_Ffn, SimpleFFN
from baseline.inductor import benchmark_ffn_inductor
import argparse
import torch
import math
import importlib.util
import sys

parser = argparse.ArgumentParser(description="Convert Attacc IR to Triton kernel")
parser.add_argument("--n", type=int, default=0, help="Case number to convert")
parser.add_argument("--o", type=int, default=0, help="0 only convert, 1 only test, 2 both convert and test")
args = parser.parse_args()

num = args.n
option = args.o
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
dtype = torch.float16

case_file = f"./benchmark_ffn_llama/ffn_llama_case{num}.txt"
output_file = f"./benchmark_ffn_llama/ffn_llama_benchmark{num}.py"
module_name = f"falcon{num}"

with open(case_file, "r") as f:
    llama_ir = f.read().strip()

M = 16
D = 128
N = 4096
P = 14336
H = 32

tensor_shapes = {
    'O2': ('M', 'N'),
    'WO': ('N', 'N'),
    'X': ('M', 'N'),
    'attn_O1': ('M', 'N'),
    'attn_O2': ('M', 'N'),
    'attn_O3': ('M'),
    'attn_O_norm': ('M', 'N'),
    'WFF1a': ('N', 'P'),
    'WFF1b': ('N', 'P'),
    'FF1a': ('M', 'P'),
    'FF1b': ('M', 'P'),
    'FF1b_silu': ('M', 'P'),
    'FF1': ('M', 'P'),
    'FF2': ('M', 'N'),
    'WFF2': ('P', 'N'),
    'O_FF': ('M', 'N'),
    'O_FF1': ('M'),
    'O_FF_norm': ('M', 'N')
}

constants = {
    'M': 16,
    'N': 4096,
    'P': 14336
}


# Convert IR to Triton kernel
def start_conversion():

    triton_code = convert_ir_to_triton(llama_ir, tensor_shapes, constants)

    # Save the generated kernel
    with open(output_file, "w") as f:
        f.write(triton_code)

    print("=" * 50)
    print("✓ Triton kernel generated successfully!")

def start_test():
    BLOCK_N = 64
    BLOCK_K = 32
    BLOCK_P = 16

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Use much smaller values to avoid numerical overflow
    # Scale down inputs to prevent overflow in float16
    scale_factor = 0.001
    O2 = torch.randn(M, N, dtype=dtype, device=device).clamp(-1, 1) * scale_factor
    O_FF = torch.zeros(M, N, dtype=dtype, device=device)
    O_FF1 = torch.zeros(M, dtype=torch.float16, device=device)
    O_FF_norm = torch.zeros(M, N, dtype=torch.float16, device=device)

    WFF1a = torch.randn(N, P, dtype=dtype, device=device).clamp(-1, 1) * scale_factor
    WFF1b = torch.randn(N, P, dtype=dtype, device=device).clamp(-1, 1) * scale_factor
    WFF2 = torch.randn(P, N, dtype=dtype, device=device).clamp(-1, 1) * scale_factor
    WO = torch.randn(N, N, dtype=dtype, device=device).clamp(-1, 1) * scale_factor
    X = torch.randn(M, N, dtype=dtype, device=device).clamp(-1, 1) * scale_factor
    attn_O1 = torch.zeros(M, N, dtype=dtype, device=device)
    attn_O2 = torch.zeros(M, N, dtype=dtype, device=device)
    attn_O3 = torch.zeros(M, dtype=dtype, device=device)

    # Add new tensors that were promoted to memory
    FF1a = torch.zeros(M, P, dtype=dtype, device=device)
    FF1b = torch.zeros(M, P, dtype=dtype, device=device)
    FF1b_silu = torch.zeros(M, P, dtype=dtype, device=device)
    FF1 = torch.zeros(M, P, dtype=dtype, device=device)
    FF2 = torch.zeros(M, N, dtype=dtype, device=device)
    attn_O_norm = torch.zeros(M, N, dtype=dtype, device=device)

    print("=" * 50)
    print("Starting kernel execution...")
    print(f"Tensor shapes:")
    print(f"  attn_O1: {attn_O1.shape}")
    print(f"  O2: {O2.shape}")
    print(f"  WO: {WO.shape}")

    spec = importlib.util.spec_from_file_location(module_name, output_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    forward = getattr(module, "forward")
    
    tensor_params = getattr(module, 'TENSOR_PARAMS', ['K1', 'V1', 'Q1', 
                                                        'K2', 'V2', 'Q2', 
                                                        'K', 'V', 'Q',
                                                        'WK', 'WV', 'WQ',
                                                        'K_cache', 'V_cache',
                                                        'C', 'C_sum',
                                                        'X', 'O', 'O1', 'O2'])
    block_params = getattr(module, 'BLOCK_PARAMS', ['block_k', 'block_n'])
    
    # Create tensors dictionary
    tensors = {
        'O2': O2,
        'O_FF': O_FF,
        'O_FF1': O_FF1,
        'O_FF_norm': O_FF_norm,
        'WFF1a': WFF1a,
        'WFF1b': WFF1b,
        'WFF2': WFF2,
        'WO': WO,
        'X': X,
        'attn_O1': attn_O1,
        'attn_O2': attn_O2,
        'attn_O3': attn_O3,
        'FF1a': FF1a,
        'FF1b': FF1b,
        'FF1b_silu': FF1b_silu,
        'FF1': FF1,
        'FF2': FF2,
        'attn_O_norm': attn_O_norm
    }

    blocks = {
        'block_k': BLOCK_K,
        'block_n': BLOCK_N,
        'block_p': BLOCK_P
    }
    
    # Build argument list based on metadata
    args = []
    for param in tensor_params:
        if param in tensors:
            args.append(tensors[param])
        else:
            raise ValueError(f"Unknown tensor parameter: {param}")
    # for param in block_params:
    #     if param in blocks:
    #         args.append(blocks[param])
    #     else:
    #         raise ValueError(f"Unknown tensor parameter: {param}")

    print("\nTesting Tile...")
    stream = torch.cuda.Stream(device)

    with torch.cuda.stream(stream):
        for _ in range(10):
            forward(*args)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            forward(*args)
    stream.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        s.record()
        for _ in range(100):
            graph.replay()
        e.record()
        stream.synchronize()
    avg = s.elapsed_time(e) / 100
    print(f"Tile:\t{avg:.4f}ms")
    # forward(O2, O_FF, O_FF1, O_FF_norm, WFF1a, WFF1b, WFF2, WO, X, attn_O1, attn_O2, attn_O3, attn_O_norm, block_k=16, block_n=16)
    print("Kernel execution completed!")
    print(FF2)
    print(torch.isnan(FF2).any().item())
    print("="*50)
    # Create TensorRT model with the same weights as our kernels
    
    print("\nTesting TensorRT...")
    trt_model = TRT_Llama_Ffn(M, N, P, WO=WO, WFF1a=WFF1a, WFF1b=WFF1b, WFF2=WFF2).to(device=device)
    trt_model.half()

    with torch.no_grad():
        for _ in range(10):
            trt_model(O2, X)
        torch.cuda.synchronize()

        s.record()
        for _ in range(100):
            out = trt_model(O2, X)
        e.record()
        torch.cuda.synchronize()
        print(out)

        avg = s.elapsed_time(e) / 100
        print(f"TensorRT:\t{avg:.4f}ms")
    print("="*50)
    print("\nComparing results...")
    if torch.allclose(FF2, out, rtol=1e-3, atol=1e-4):
        print("✓ Results match!")
    else:
        print("✗ Results do not match!")
        max_diff = torch.abs(FF2 - out).max()
        print(f"Maximum difference: {max_diff}")

    print("\nTesting Torch Inductor Implementation...")
    benchmark_ffn_inductor(M, N, P, X, O2, WO, WFF1a, WFF1b, WFF2)

print(f"[Case{num}]")
if option == 0:
    start_conversion()
elif option == 1:
    start_test()
else:
    start_conversion()
    start_test()