from codegen.convert_module import convert_ir_to_triton
from utils.shapes import get_forward_shape_dict
from utils.config import load_model_config, setup_directories
# from baseline.inductor import benchmark_rms
import argparse
import torch
import importlib.util
import sys

parser = argparse.ArgumentParser(description="Convert IR to Triton kernel")
parser.add_argument("--n", type=int, default=0, help="Case number to convert")
parser.add_argument("--m", type=str, default="falcon", help="Input model type")
parser.add_argument("--t", type=str, default="vanilla", help="RMS Type")
parser.add_argument("--s", type=int, help="Sequence Length")
parser.add_argument("--o", type=int, default=0, help="0 only convert, 1 only test, 2 both convert and test")
parser.add_argument("--d", type=int, default=0, help="CUDA device number")
parser.add_argument("--pre", action="store_true", help="Whether to use prenorm or not")
args = parser.parse_args()

num = args.n
option = args.o
model = args.m
rms = args.t
seq = args.s
pre = args.pre
gpu = args.d
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
dtype = torch.float32

def initialize_paths_and_configs(num, seq, model, rms, pre):
    """
    Initialize file paths, model configurations, and load IR code.

    Returns:
        tuple: (case_file, output_file, module_name, ir_code, constants,
                tensor_shapes, M, N, D, H, P)
    """
    # Setup file paths
    case_file = f"/home/chani227/Trinity/CodeGen/Training/data/fwd/seq{seq}_fwd{num}.txt"
    output_file = f"/home/chani227/Trinity/CodeGen/Training/data/fwd/seq{seq}_fwd{num}.py"
    module_name = f"{rms}_{model}_best" if not pre else f"{rms}_{model}_prenorm_best"

    # Load IR code
    with open(case_file, "r") as f:
        ir_code = f.read().strip()

    # Load model-specific constants from JSON
    constants = load_model_config(model)
    M = constants['M']
    D = constants['D']
    N = constants['N']
    H = constants['H']
    P = constants['P']

    # Get forward tensor shapes from utils
    tensor_shapes = get_forward_shape_dict()

    return (case_file, output_file, module_name, ir_code, constants,
            tensor_shapes, M, N, D, H, P)

# Convert IR to Triton kernel
def start_conversion(ir_code, output_file, tensor_shapes, constants):
    """
    Convert IR code to Triton kernel and save it.

    Args:
        ir_code: IR code string
        output_file: Output path for generated kernel
        tensor_shapes: Tensor shape dictionary
        constants: Model constants dictionary
    """
    triton_code = convert_ir_to_triton(ir_code, tensor_shapes, constants)

    # Save the generated kernel
    with open(output_file, "w") as f:
        f.write(triton_code)

    print("=" * 50)
    print("✓ Triton kernel generated successfully!")

def torch_baseline(X, WQ, WK, WV, M, N, D, H):
    """PyTorch baseline implementation for seq16_fwd0"""
    # Q1, K1, V1 = X @ WQ, X @ WK, X @ WV
    Q1 = torch.matmul(X, WQ)
    K1 = torch.matmul(X, WK)
    V1 = torch.matmul(X, WV)

    # Reshape to (M, H, D) and permute to (H, M, D)
    Q = Q1.view(M, H, D).permute(1, 0, 2)
    K = K1.view(M, H, D).permute(1, 0, 2)
    V = V1.view(M, H, D).permute(1, 0, 2)

    # Attention: C_exp = exp(Q @ K^T)
    K_T = K.permute(0, 2, 1)  # (H, D, M)
    C_exp = torch.exp(torch.matmul(Q, K_T))  # (H, M, M)

    # Softmax: C_exp / sum(C_exp, dim=-1)
    C_sum = torch.sum(C_exp, dim=2, dtype=torch.float32)  # (H, M)
    C_div = C_exp / C_sum.unsqueeze(2)  # (H, M, M)

    # Output: O = C_div @ V
    O = torch.matmul(C_div.to(V.dtype), V)  # (H, M, D)

    # Reshape back: permute to (M, H, D) and reshape to (M, N)
    O_permuted = O.permute(1, 0, 2)  # (M, H, D)
    O2 = O_permuted.reshape(M, N)  # (M, N)

    return O2

def start_test(output_file, module_name, M, N, D, H, P):
    """
    Execute generated kernel and compare with PyTorch baseline.

    Args:
        output_file: Path to generated kernel file
        module_name: Module name for importing
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
    C = torch.zeros((H, M, M), device=device, dtype=dtype)
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

    # Key Former
    # noise = torch.randn((H, M, P+M), device=device, dtype=dtype)
    # C_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    # C_exp_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    # C_sum_perturb = torch.zeros((H, M), device=device, dtype=dtype)
    # C_div_perturb = torch.zeros((H, M, P+M), device=device, dtype=dtype)
    # C_out = torch.zeros((H, P+M), device=device, dtype=dtype)

    # QKNorm
    # Q_norm = torch.zeros((H, M, D), device=device, dtype=dtype)
    # K_norm = torch.zeros((H, M, D), device=device, dtype=dtype)

    # RoCo
    # C_out1 = torch.zeros((H, P+M), device=device, dtype=dtype)
    # C_out2 = torch.zeros((H, P+M), device=device, dtype=dtype)

    print("=" * 50)
    print("Starting kernel execution...")

    spec = importlib.util.spec_from_file_location(module_name, output_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    forward = getattr(module, "forward")

    tensor_params = getattr(module, 'TENSOR_PARAMS')
    block_params = getattr(module, 'BLOCK_PARAMS')

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

        # 'noise': noise,
        # 'C_perturb': C_perturb,
        # 'C_exp_perturb': C_exp_perturb,
        # 'C_sum_perturb': C_sum_perturb,
        # 'C_div_perturb': C_div_perturb,
        # 'C_out': C_out,

        # 'Q_norm': Q_norm,
        # 'K_norm': K_norm,

        # 'C_out1': C_out1,
        # 'C_out2': C_out2
    }

    blocks = {
        'block_k': BLOCK_K,
        'block_p': BLOCK_P
    }

    args = []
    for param in tensor_params:
        if param in tensors:
            args.append(tensors[param])
        else:
            raise ValueError(f"Unknown tensor parameter: {param}")
    for param in block_params:
        if param in blocks:
            args.append(blocks[param])
        else:
            raise ValueError(f"Unknown tensor parameter: {param}")

    # ------------------- Tile -------------------
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

    # Run PyTorch baseline for comparison
    print("=" * 50)
    print("Running PyTorch baseline...")
    with torch.no_grad():
        baseline_O2 = torch_baseline(X, WQ, WK, WV, M, N, D, H)

    print("\nResults comparison:")
    print(f"Triton O2:\n{O2}")
    print(f"\nBaseline O2:\n{baseline_O2}")

    # Check if results match
    diff = torch.abs(O2 - baseline_O2)
    max_diff = diff.max()
    mean_diff = diff.mean()

    # Find location of max difference
    max_idx = torch.argmax(diff)
    max_location = torch.unravel_index(max_idx, diff.shape)
    triton_val = O2[max_location]
    baseline_val = baseline_O2[max_location]

    if torch.allclose(O2, baseline_O2, rtol=1e-2, atol=1e-3):
        print("\n✓ Results match!")
    else:
        print("\n✗ Results do not match!")
        print(f"Maximum difference: {max_diff}")
        print(f"  Location: {max_location}")
        print(f"  Triton value: {triton_val}")
        print(f"  Baseline value: {baseline_val}")
        if abs(baseline_val) > 1e-10:
            rel_err = max_diff / abs(baseline_val)
            print(f"  Relative error: {rel_err:.2%}")
        print(f"Mean difference: {mean_diff}")

    # match rms:
    #     case "vanilla":
    #         from ref_rms import Vanilla, TensorRT_Vanilla, FlashInfer_Vanilla
    #         trt = TensorRT_Vanilla(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
    #         ti = Vanilla(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #         fi = FlashInfer_Vanilla(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #     case "prenorm":
    #         from ref_rms import PreNorm, TensorRT_PreNorm, FlashInfer_PreNorm
    #         trt = TensorRT_PreNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
    #         ti = PreNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #         fi = FlashInfer_PreNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #     case "keyformer":
    #         if pre:
    #             from ref_rms import NormKeyFormer, TensorRT_NormKeyFormer, FlashInfer_NormKeyFormer
    #             trt = TensorRT_NormKeyFormer(M, N, D, H, K_cache.clone(), V_cache.clone(), P, noise, WQ, WK, WV)
    #             ti = NormKeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #             fi = FlashInfer_NormKeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #         else:
    #             from ref_rms import KeyFormer, TensorRT_KeyFormer, FlashInfer_KeyFormer
    #             trt = TensorRT_KeyFormer(M, N, D, H, K_cache.clone(), V_cache.clone(), P, noise, WQ, WK, WV)
    #             ti = KeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #             fi = FlashInfer_KeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #     case "qknorm":
    #         if pre:
    #             from ref_rms import NormQKNorm, TensorRT_NormQKNorm, FlashInfer_NormQKNorm
    #             trt = TensorRT_NormQKNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
    #             ti = NormQKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #             fi = FlashInfer_NormQKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #         else:
    #             from ref_rms import QKNorm, TensorRT_QKNorm, FlashInfer_QKNorm
    #             trt = TensorRT_QKNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
    #             ti = QKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #             fi = FlashInfer_QKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #     case "roco":
    #         if pre:
    #             from ref_rms import NormRoCo, TensorRT_NormRoCo, FlashInfer_NormRoCo
    #             trt = TensorRT_NormRoCo(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
    #             ti = NormRoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #             fi = FlashInfer_NormRoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #         else:
    #             from ref_rms import RoCo, TensorRT_RoCo, FlashInfer_RoCo
    #             trt = TensorRT_RoCo(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
    #             ti = RoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    #             fi = FlashInfer_RoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)

    # print("=" * 50)
    # print("Starting ref kernel execution...")
    # print(f"\nTesting {rms.upper()} Flash Infer...")
    # print("\nTesting Correct Flash Infer...")
    # # ------------------- Flash Infer -------------------
    # fi.half()
    # with torch.no_grad():
    #     for _ in range(10):
    #         out = fi(X)
    #     torch.cuda.synchronize()

    #     start_fi = torch.cuda.Event(enable_timing=True)
    #     end_fi = torch.cuda.Event(enable_timing=True)

    #     start_fi.record()
    #     for _ in range(ITER):
    #         out = fi(X)
    #     end_fi.record()
    #     torch.cuda.synchronize()
    #     fi_time = start_fi.elapsed_time(end_fi) / ITER
    #     print(f"FI: {fi_time}ms")
    # print(out)


    # print("=" * 50)
    # print(f"\nTesting {rms.upper()} TensorRT...")
    # print("\nTesting Correct TensorRT...")
    # # ------------------- Tensor RT -------------------
    
    # trt.half()
    # with torch.no_grad():
    #     for _ in range(10):
    #         out = trt(X)
    #     torch.cuda.synchronize()

    #     start_rt = torch.cuda.Event(enable_timing=True)
    #     end_rt = torch.cuda.Event(enable_timing=True)

    #     start_rt.record()
    #     for _ in range(ITER):
    #         out = trt(X)
    #     end_rt.record()
    #     torch.cuda.synchronize()
    #     rt_time = start_rt.elapsed_time(end_rt) / ITER
    #     print(f"TRT: {rt_time}ms")
    # print(out)

    # # ------------------- Torch Inductor -------------------
    # print("\nTesting Torch Inductor Implementation...")

    # benchmark_rms(ti.eval(), X)
    
    # print("\nComparing results...")
    # if torch.allclose(O2, out, rtol=1e-3, atol=1e-4):
    #     print("✓ Results match!")
    # else:
    #     print("✗ Results do not match!")
    #     max_diff = torch.abs(O2 - out).max()
    #     print(f"Maximum difference: {max_diff}")
    # print("=" * 50)

print(f"[Case{num}]")

# Setup directories and initialize configurations
setup_directories(seq)
(case_file, output_file, module_name, ir_code, constants,
 tensor_shapes, M, N, D, H, P) = initialize_paths_and_configs(
    num, seq, model, rms, pre
)

# Execute based on option
if option == 0:
    start_conversion(ir_code, output_file, tensor_shapes, constants)
elif option == 1:
    start_test(output_file, module_name, M, N, D, H, P)
else:
    start_conversion(ir_code, output_file, tensor_shapes, constants)
    start_test(output_file, module_name, M, N, D, H, P)