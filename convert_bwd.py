from codegen.convert_module import convert_ir_to_triton
from utils.shapes import get_backward_shape_dict
from utils.config import load_model_config, setup_directories
# from baseline.inductor import benchmark_rms
import argparse
import torch
import importlib.util
import sys

parser = argparse.ArgumentParser(description="Convert IR to Triton kernel")
parser.add_argument("--n", type=int, default=0, help="BWD Case number to convert")
parser.add_argument("--m", type=str, default="falcon", help="Input model type")
parser.add_argument("--c", type=int, help="BWD List case number")
parser.add_argument("--s", type=int, help="Sequence Length")
parser.add_argument("--o", type=int, default=0, help="0 only convert, 1 only test, 2 both convert and test")
parser.add_argument("--d", type=int, default=2)
args = parser.parse_args()

num = args.n
list_case = args.c
option = args.o
model = args.m
seq = args.s
gpu = args.d
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
dtype = torch.float32

def initialize_paths_and_configs(num, list_case, seq, model):
    """
    Initialize file paths, model configurations, and load IR code.

    Returns:
        tuple: (fwd_file, fwd_output, fwd_name, bwd_file, bwd_output,
                bwd_name, fwd_ir, bwd_ir, constants, tensor_shapes, M, N, D, H, P)
    """
    # Setup file paths
    fwd_file = f"/home/chani227/Trinity/CodeGen/Training/data/fwd/seq{seq}_fwd{list_case}.txt"
    fwd_output = f"/home/chani227/Trinity/CodeGen/Training/data/fwd/seq{seq}_fwd{list_case}.py"
    fwd_name = f"forward_{num}"

    bwd_file = f"/home/chani227/Trinity/evaluation/backward/bwd/seq{seq}_bwd{list_case}_case{num}.txt"
    bwd_output = f"/home/chani227/Trinity/evaluation/backward/bwd/seq{seq}_{list_case}benchmark{num}.py"
    bwd_name = f"backward_{num}"

    # Load IR code
    with open(fwd_file, "r") as f:
        fwd_ir = f.read().strip()

    with open(bwd_file, "r") as f:
        bwd_ir = f.read().strip()

    # Load model-specific constants from JSON
    constants = load_model_config(model)
    M = constants['M']
    D = constants['D']
    N = constants['N']
    H = constants['H']
    P = constants['P']

    # Get backward tensor shapes from utils
    tensor_shapes = get_backward_shape_dict()

    return (fwd_file, fwd_output, fwd_name, bwd_file, bwd_output,
            bwd_name, fwd_ir, bwd_ir, constants, tensor_shapes, M, N, D, H, P)

# Convert IR to Triton kernel
def start_conversion(fwd_ir, bwd_ir, fwd_output, bwd_output, tensor_shapes, constants):
    """
    Convert IR code to Triton kernels and save them.

    Args:
        fwd_ir: Forward pass IR code
        bwd_ir: Backward pass IR code
        fwd_output: Output path for forward kernel
        bwd_output: Output path for backward kernel
        tensor_shapes: Tensor shape dictionary
        constants: Model constants dictionary
    """
    fwd_code = convert_ir_to_triton(fwd_ir, tensor_shapes, constants)
    bwd_code = convert_ir_to_triton(bwd_ir, tensor_shapes, constants)

    # Save the generated kernel
    with open(fwd_output, "w") as f:
        f.write(fwd_code)

    with open(bwd_output, "w") as f:
        f.write(bwd_code)

    print("=" * 50)
    print("✓ Triton kernel generated successfully!")

def start_test(fwd_output, bwd_output, fwd_name, bwd_name, M, N, D, H, P):
    """
    Execute generated kernels and compare with PyTorch reference.

    Args:
        fwd_output: Path to forward kernel file
        bwd_output: Path to backward kernel file
        fwd_name: Forward kernel module name
        bwd_name: Backward kernel module name
        M, N, D, H, P: Model dimension parameters
    """
    BLOCK_N = 64
    BLOCK_K = 32
    BLOCK_P = 16

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Use much smaller values to avoid numerical overflow
    std = 0.01
    X = torch.randn((M, N), device=device, dtype=dtype) * std
    X2 = torch.randn(M, device=device, dtype=dtype)
    X_norm = torch.zeros((M, N), device=device, dtype=dtype)
    WQ = torch.randn((N, N), device=device, dtype=dtype) * std
    WK = torch.randn((N, N), device=device, dtype=dtype) * std
    WV = torch.randn((N, N), device=device, dtype=dtype) * std
    K_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std
    V_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std
    C = torch.zeros((H, M, P), device=device, dtype=dtype)
    C_exp = torch.zeros((H, M, M), device=device, dtype=dtype)
    C_sum = torch.zeros((H, M), device=device, dtype=torch.float32)
    K = torch.zeros((H, M, D), device=device, dtype=dtype)
    K1 = torch.zeros((M, N), device=device, dtype=dtype)
    K2 = torch.zeros((M, H, D), device=device, dtype=dtype)
    O = torch.zeros((H, M, D), device=device, dtype=dtype)
    O1 = torch.zeros((M, H, D), device=device, dtype=dtype)
    O2 = torch.zeros((M, N), device=device, dtype=dtype)
    Q = torch.zeros((H, M, D), device=device, dtype=dtype)
    Q1 = torch.zeros((M, N), device=device, dtype=dtype)
    Q2 = torch.zeros((M, H, D), device=device, dtype=dtype)
    V = torch.zeros((H, M, D), device=device, dtype=dtype)
    V1 = torch.zeros((M, N), device=device, dtype=dtype)
    V2 = torch.zeros((M, H, D), device=device, dtype=dtype)

    # Gradient
    dWQ = torch.zeros((N, N), device=device, dtype=dtype)
    dWK = torch.zeros((N, N), device=device, dtype=dtype)
    dWV = torch.zeros((N, N), device=device, dtype=dtype)

    dO2 = torch.ones((M, N), device=device, dtype=dtype)
    dO = torch.zeros((H, M, D), device=device, dtype=dtype)
    dO_tmp = torch.zeros((H, M, D), device=device, dtype=dtype)
    dC_exp = torch.zeros((H, M, M), device=device, dtype=dtype)
    dC_sum = torch.zeros((H, M), device=device, dtype=dtype)
    dC = torch.zeros((H, M, M), dtype=dtype, device=device)

    dQ = torch.zeros((H, M, D), device=device, dtype=dtype)
    dK = torch.zeros((H, M, D), device=device, dtype=dtype)
    dV = torch.zeros((H, M, D), device=device, dtype=dtype)
    dQ1 = torch.zeros((M, N), dtype=dtype, device=device)
    dK1 = torch.zeros((M, N), dtype=dtype, device=device)
    dV1 = torch.zeros((M, N), dtype=dtype, device=device)

    # Key Former
    noise = torch.randn((H, M, P+M), device=device, dtype=dtype)
    C_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    C_exp_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    C_sum_perturb = torch.zeros((H, M), device=device, dtype=dtype)
    C_div_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    C_out = torch.zeros((H, P+M), device=device, dtype=dtype)

    # QKNorm
    Q_norm = torch.zeros((H, M, D), device=device, dtype=dtype)
    K_norm = torch.zeros((H, M, D), device=device, dtype=dtype)

    # RoCo
    C_out1 = torch.zeros((H, P+M), device=device, dtype=dtype)
    C_out2 = torch.zeros((H, P+M), device=device, dtype=dtype)

    print("=" * 50)
    print("Starting kernel execution...")

    fwd_spec = importlib.util.spec_from_file_location(fwd_name, fwd_output)
    fwd_module = importlib.util.module_from_spec(fwd_spec)
    sys.modules[fwd_name] = fwd_module
    fwd_spec.loader.exec_module(fwd_module)
    forward = getattr(fwd_module, "forward")

    fwd_tensor_params = getattr(fwd_module, 'TENSOR_PARAMS')
    fwd_block_params = getattr(fwd_module, 'BLOCK_PARAMS')

    bwd_spec = importlib.util.spec_from_file_location(bwd_name, bwd_output)
    bwd_module = importlib.util.module_from_spec(bwd_spec)
    sys.modules[bwd_name] = bwd_module
    bwd_spec.loader.exec_module(bwd_module)
    backward = getattr(bwd_module, "forward")

    bwd_tensor_params = getattr(bwd_module, 'TENSOR_PARAMS')
    bwd_block_params = getattr(bwd_module, 'BLOCK_PARAMS')

    tensors = {
        'X': X,
        'X2': X2,
        'X_norm': X_norm,
        'WQ': WQ,
        'WK': WK,
        'WV': WV,
        'K_cache': K_cache,
        'V_cache': V_cache,
        'C': C,
        'C_exp': C_exp,
        'C_sum': C_sum,
        'K': K,
        'K1': K1,
        'K2': K2,
        'O': O,
        'O1': O1,
        'O2': O2,
        'Q': Q,
        'Q1': Q1,
        'Q2': Q2,
        'V': V,
        'V1': V1,
        'V2': V2,

        'dWQ': dWQ,
        'dWK': dWK,
        'dWV': dWV,
        'dO2': dO2,
        'dO': dO,
        'dO_tmp': dO_tmp,
        'dC': dC,
        'dC_exp': dC_exp,
        'dC_sum': dC_sum,
        'dQ': dQ,
        'dK': dK,
        'dV': dV,
        'dQ1': dQ1,
        'dK1': dK1,
        'dV1': dV1,

        'noise': noise,
        'C_perturb': C_perturb,
        'C_exp_perturb': C_exp_perturb,
        'C_sum_perturb': C_sum_perturb,
        'C_div_perturb': C_div_perturb,
        'C_out': C_out,

        'Q_norm': Q_norm,
        'K_norm': K_norm,

        'C_out1': C_out1,
        'C_out2': C_out2
    }

    blocks = {
        'block_k': BLOCK_K,
        'block_n': BLOCK_N,
        'block_p': BLOCK_P
    }

    fwd_args = []
    for param in fwd_tensor_params:
        if param in tensors:
            fwd_args.append(tensors[param])
        else:
            raise ValueError(f"Unknown tensor parameter: {param}")
    for param in fwd_block_params:
        if param in blocks:
            fwd_args.append(blocks[param])
        else:
            raise ValueError(f"Unknown tensor parameter: {param}")

    bwd_args = []
    for param in bwd_tensor_params:
        if param in tensors:
            bwd_args.append(tensors[param])
        else:
            raise ValueError(f"Unknown tensor parameter: {param}")
    for param in bwd_block_params:
        if param in blocks:
            bwd_args.append(blocks[param])
        else:
            raise ValueError(f"Unknown tensor parameter: {param}")

    forward(*fwd_args)
    print(Q1)
    # ------------------- Check Only BWD -------------------
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        for _ in range(10):
            backward(*bwd_args)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            backward(*bwd_args)
    stream.synchronize()

    start_rms = torch.cuda.Event(enable_timing=True)
    end_rms = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        start_rms.record()
        for _ in range(100):
            graph.replay()
        end_rms.record()  
    stream.synchronize()      

    time = start_rms.elapsed_time(end_rms) / 100

    print(f"Kernel execution completed!: {time}ms")
    print("=" * 50)
    print("Triton Results:")
    print(f"dWQ:\n{dWQ}")
    print(f"dWK:\n{dWK}")
    print(f"dWV:\n{dWV}")

    # Compare with PyTorch reference
    print("=" * 50)
    print("Computing PyTorch reference...")
    from ref.ref_backward import compute_gradients

    # Warmup for PyTorch
    for _ in range(10):
        ref_dWQ, ref_dWK, ref_dWV, Q1_ret = compute_gradients(M, N, D, H, X.clone(), WQ.clone(), WK.clone(), WV.clone(), dO2.clone(), device, dtype)
    torch.cuda.synchronize()

    # Time measurement for PyTorch
    start_torch = torch.cuda.Event(enable_timing=True)
    end_torch = torch.cuda.Event(enable_timing=True)

    start_torch.record()
    for _ in range(100):
        ref_dWQ, ref_dWK, ref_dWV, Q1_ret = compute_gradients(M, N, D, H, X.clone(), WQ.clone(), WK.clone(), WV.clone(), dO2.clone(), device, dtype)
    end_torch.record()
    torch.cuda.synchronize()
    print("ret")
    print(Q1_ret)
    torch_time = start_torch.elapsed_time(end_torch) / 100
    print(f"PyTorch execution completed!: {torch_time}ms")

    print("=" * 50)
    print("PyTorch Reference:")
    print(f"dWQ:\n{ref_dWQ}")
    print(f"dWK:\n{ref_dWK}")
    print(f"dWV:\n{ref_dWV}")

    print("=" * 50)
    print("Comparison:")
    dWQ_match = torch.allclose(dWQ, ref_dWQ, rtol=1e-2, atol=1e-3)
    dWK_match = torch.allclose(dWK, ref_dWK, rtol=1e-2, atol=1e-3)
    dWV_match = torch.allclose(dWV, ref_dWV, rtol=1e-2, atol=1e-3)

    print(f"dWQ match: {'✓' if dWQ_match else '✗'} (max diff: {torch.abs(dWQ - ref_dWQ).max():.6f})")
    print(f"dWK match: {'✓' if dWK_match else '✗'} (max diff: {torch.abs(dWK - ref_dWK).max():.6f})")
    print(f"dWV match: {'✓' if dWV_match else '✗'} (max diff: {torch.abs(dWV - ref_dWV).max():.6f})")
    print("=" * 50)
    print("Performance Comparison:")
    print(f"Triton:  {time:.4f}ms")
    print(f"PyTorch: {torch_time:.4f}ms")
    print(f"Speedup: {torch_time/time:.2f}x")
    print("=" * 50)

print(f"[Case{num}]")

# Setup directories and initialize configurations
setup_directories(seq)
(fwd_file, fwd_output, fwd_name, bwd_file, bwd_output,
 bwd_name, fwd_ir, bwd_ir, constants, tensor_shapes, M, N, D, H, P) = initialize_paths_and_configs(
    num, list_case, seq, model
)

# Execute based on option
if option == 0:
    start_conversion(fwd_ir, bwd_ir, fwd_output, bwd_output, tensor_shapes, constants)
elif option == 1:
    start_test(fwd_output, bwd_output, fwd_name, bwd_name, M, N, D, H, P)
else:
    start_conversion(fwd_ir, bwd_ir, fwd_output, bwd_output, tensor_shapes, constants)
    start_test(fwd_output, bwd_output, fwd_name, bwd_name, M, N, D, H, P)