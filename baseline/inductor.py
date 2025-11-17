import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch._dynamo import config
import torch._inductor.config as inductor_config

from baseline.ref_rms import SimpleAttention
from baseline.ffn_ref import SimpleFFN

# Fix for PyTorch 2.7.0 and Triton 3.4.0 compatibility issue
try:
    import triton
    if hasattr(triton, 'compiler') and hasattr(triton.compiler, 'CompiledKernel'):
        if not hasattr(triton.compiler.CompiledKernel, 'launch_enter_hook'):
            triton.compiler.CompiledKernel.launch_enter_hook = lambda *args, **kwargs: None
except:
    pass

# Configure inductor for better performance
inductor_config.triton.unique_kernel_names = True
inductor_config.coordinate_descent_tuning = True
inductor_config.triton.cudagraphs = True  # Disable CUDA graphs to avoid launch_enter_hook issue

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
dtype = torch.float16

def benchmark_rms(model, input_data, num_warmup=10, num_iterations=100):
    """Benchmark model with and without compilation"""
    
    print("=" * 60)
    print("Starting Benchmark")
    print("=" * 60)
    
    # 1. Eager mode (no compilation) benchmark
    print("\n1. Eager Mode (No Compilation)")
    print("-" * 40)
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_data)
    
    torch.cuda.synchronize()
    
    # Measure eager mode
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    s.record()
    for _ in range(num_iterations):
        with torch.no_grad():
            out = model(input_data)
    e.record()
    torch.cuda.synchronize()
    avg_eager = s.elapsed_time(e) / num_iterations
    
    print(f"Eager mode - Mean: {avg_eager:.4f} ms")
    print(out)

    # 2. Compile with torch.compile (inductor backend)
    print("\n2. Compiled Mode (Torch Inductor)")
    print("-" * 40)
    print("Compiling model...")
    
    modes = ["default", "reduce-overhead", "max-autotune"]

    for mode in modes:
        compiled_model = torch.compile(
            model,
            backend="inductor",
            mode=mode,  # Options: "default", "reduce-overhead", "max-autotune"
            fullgraph=True
        )
        
        print("Compilation complete. Running warmup...")
        
        # Warmup compiled model
        for i in range(num_warmup):
            with torch.no_grad():
                _ = compiled_model(input_data)
            if i == 0:
                print("First compiled run complete (graph compilation)")
        
        torch.cuda.synchronize()
        
        print("Measuring compiled performance...")
        
        # Measure compiled mode
        
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            s.record()
            for _ in range(num_iterations):
                out_compile = compiled_model(input_data)
            e.record()
            torch.cuda.synchronize()
            avg_compile = s.elapsed_time(e) / num_iterations
        
        print(f"Compiled {mode} - Mean: {avg_compile:.4f} ms")
        print(out_compile)
    
    return

def benchmark_rms_inductor(M, N, D, P, H, X, WQ, WK, WV, cache_K, cache_V):
    # Create model
    model = SimpleAttention(M, N, D, P, cache_K, cache_V, WQ, WK, WV).to(device).to(dtype)
    model.eval()
    
    # Run benchmark
    benchmark_rms(
        model, 
        X,
        num_warmup=10,
        num_iterations=100
    )

def benchmark_ffn(model, X, O2, num_warmup=10, num_iterations=100):
    """Benchmark model with and without compilation"""
    
    print("=" * 60)
    print("Starting Benchmark")
    print("=" * 60)
    
    # 1. Eager mode (no compilation) benchmark
    print("\n1. Eager Mode (No Compilation)")
    print("-" * 40)
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(O2, X)
    
    torch.cuda.synchronize()
    
    # Measure eager mode
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    s.record()
    for _ in range(num_iterations):
        with torch.no_grad():
            out = model(O2, X)
    e.record()
    torch.cuda.synchronize()
    avg_eager = s.elapsed_time(e) / num_iterations
    
    print(f"Eager mode - Mean: {avg_eager:.4f} ms")
    print(out)

    # 2. Compile with torch.compile (inductor backend)
    print("\n2. Compiled Mode (Torch Inductor)")
    print("-" * 40)
    print("Compiling model...")
    
    modes = ["default", "reduce-overhead", "max-autotune"]

    for mode in modes:
        compiled_model = torch.compile(
            model,
            backend="inductor",
            mode=mode,  # Options: "default", "reduce-overhead", "max-autotune"
            fullgraph=True
        )
        
        print("Compilation complete. Running warmup...")
        
        # Warmup compiled model
        for i in range(num_warmup):
            with torch.no_grad():
                _ = compiled_model(O2, X)
            if i == 0:
                print("First compiled run complete (graph compilation)")
        
        torch.cuda.synchronize()
        
        print("Measuring compiled performance...")
        
        # Measure compiled mode
        
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            s.record()
            for _ in range(num_iterations):
                out_compile = compiled_model(O2, X)
            e.record()
            torch.cuda.synchronize()
            avg_compile = s.elapsed_time(e) / num_iterations
        
        print(f"Compiled {mode} - Mean: {avg_compile:.4f} ms")
        print(out_compile)
    
    return

def benchmark_ffn_inductor(M, N, P, X, O2, WO, WFF1a, WFF1b, WFF2):
    # Create model
    model = SimpleFFN(M, N, P, WO, WFF1a, WFF1b, WFF2)
    model.eval()
    
    # Run benchmark
    benchmark_ffn(
        model, 
        X,
        O2,
        num_warmup=10,
        num_iterations=100
    )
