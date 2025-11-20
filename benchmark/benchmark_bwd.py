import torch
import triton
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import tempfile
import importlib.util
import sys
import os
import shutil
import traceback
from dataclasses import dataclass
from tqdm import tqdm
import json
import argparse

from codegen.convert_module import convert_ir_to_triton
from utils.discord import send_discord_notification
from utils.shapes import TensorShapeBuilder
from utils.config import load_model_config

@dataclass
class BenchmarkResult:
    ir_id: int
    ir_expression: str
    execution_time: float
    tensor_config: Dict[str, int]
    error: Optional[str] = None


class BackwardBenchmark:
    def __init__(self, tensor_config: Dict[str, int], device_num: int = 0):
        """Initialize backward benchmark with given tensor configuration.

        Args:
            tensor_config: Dictionary containing M, N, D, H, P parameters
            device_num: CUDA device number (default: 0)
        """
        # Extract dimensions from config
        self.M = tensor_config['M']
        self.N = tensor_config['N']
        self.D = tensor_config['D']
        self.H = tensor_config['H']
        self.P = tensor_config['P']

        # Store the config
        self.tensor_config = tensor_config

        # Build shape definitions using utility module
        shape_builder = TensorShapeBuilder(self.M, self.N, self.D, self.H, self.P)
        self.tensor_shapes = shape_builder.get_backward_tensor_shapes()
        self.shape_dict = shape_builder.get_backward_shape_dict()
        self.const_dict = tensor_config.copy()

        # Setup device
        self.device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)
        print(f"GPU: {torch.cuda.get_device_name(self.device)}")

        # Create test tensors
        self.create_test_tensors()

        # Track temp files for cleanup
        self._temp_files = []

    def create_test_tensors(self):
        """Create random test tensors for benchmarking backward pass."""
        self.tensors = {}
        std = 0.01

        # Input tensors
        self.tensors['X'] = torch.randn(self.tensor_shapes['X'], dtype=torch.float32, device=self.device) * std
        self.tensors['X2'] = torch.randn(self.tensor_shapes['X2'], dtype=torch.float32, device=self.device)
        self.tensors['WQ'] = torch.randn(self.tensor_shapes['WQ'], dtype=torch.float32, device=self.device) * std
        self.tensors['WK'] = torch.randn(self.tensor_shapes['WK'], dtype=torch.float32, device=self.device) * std
        self.tensors['WV'] = torch.randn(self.tensor_shapes['WV'], dtype=torch.float32, device=self.device) * std
        self.tensors['K_cache'] = torch.randn(self.tensor_shapes['K_cache'], dtype=torch.float32, device=self.device) * std
        self.tensors['V_cache'] = torch.randn(self.tensor_shapes['V_cache'], dtype=torch.float32, device=self.device) * std
        self.tensors['noise'] = torch.randn(self.tensor_shapes['noise'], dtype=torch.float32, device=self.device)

        self.tensors['C_exp'] = torch.exp(torch.randn(self.tensor_shapes['C_exp'], dtype=torch.float32, device=self.device)*0.1)
        self.tensors['C_sum'] = torch.randn(self.tensor_shapes['C_sum'], dtype=torch.float32, device=self.device)

        # Gradient input
        self.tensors['dO2'] = torch.ones(self.tensor_shapes['dO2'], dtype=torch.float32, device=self.device)

        # Initialize output and intermediate tensors to zero
        zero_tensors = ['X_norm', 'Q', 'K', 'V', 'Q1', 'K1', 'V1', 'Q2', 'K2', 'V2',
                       'O', 'O1', 'O2', 'C', 'C_div',
                       'dWQ', 'dWK', 'dWV', 'dQ', 'dK', 'dV', 'dQ1', 'dK1', 'dV1',
                       'dO', 'dO_tmp', 'dC', 'dC_exp', 'dC_sum',
                       'C_perturb', 'C_exp_perturb', 'C_sum_perturb', 'C_div_perturb',
                       'C_out', 'C_out1', 'C_out2', 'Q_norm', 'K_norm']

        for name in zero_tensors:
            if name in self.tensor_shapes:
                self.tensors[name] = torch.zeros(self.tensor_shapes[name], dtype=torch.float32, device=self.device)

    def parse_ir_file(self, file_path: str) -> List[Tuple[int, str]]:
        """Parse the backward IR expressions file and extract all expressions."""
        expressions = []

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    # Extract IR ID and expression
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        id_part = parts[0]
                        try:
                            ir_id = int(id_part)
                            ir_expr = parts[1].strip()
                            if 'dummydata' not in ir_expr:
                                expressions.append((ir_id, ir_expr))
                        except:
                            continue

        return expressions

    def generate_kernel_code(self, ir_expr: str, constants: Dict[str, int] = None) -> Optional[str]:
        """Generate Triton kernel code from IR expression."""
        try:
            kernel_code = convert_ir_to_triton(ir_expr, self.shape_dict, self.const_dict)
            return kernel_code

        except Exception as e:
            print(f"Error generating kernel: {e}")
            traceback.print_exc()
            return None

    def compile_and_load_kernel(self, kernel_code: str, kernel_id: int) -> Optional[callable]:
        """Compile Triton kernel code and return callable function."""
        try:
            # Create temporary module file
            module_name = f"backward_kernel_{kernel_id}"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(kernel_code)
                temp_file = f.name

            # Load module dynamically
            spec = importlib.util.spec_from_file_location(module_name, temp_file)
            module = importlib.util.module_from_spec(spec)

            # Keep the file path in the module for Triton to find source
            module.__file__ = temp_file

            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Don't delete temp file immediately - Triton needs it
            # Store temp file path for later cleanup
            self._temp_files = getattr(self, '_temp_files', [])
            self._temp_files.append(temp_file)

            # Return the module instead of just the function
            if hasattr(module, 'forward'):
                return module
            else:
                print("Error: Cannot call the kernel")
                return None

        except Exception as e:
            print(f"Error compiling kernel: {e}")
            traceback.print_exc()
            if 'temp_file' in locals() and os.path.exists(temp_file):
                # Print the generated code for debugging
                with open(temp_file, 'r') as f:
                    print("Generated kernel code:")
                    print(f.read())
                os.unlink(temp_file)
            return None

    def benchmark_kernel(self, kernel_module, ir_id, warmup_runs: int = 10, benchmark_runs: int = 100) -> float:
        """Benchmark a single kernel and return execution time in milliseconds."""
        try:
            # Get metadata and forward function
            tensor_params = getattr(kernel_module, 'TENSOR_PARAMS', [])
            kernel_fn = kernel_module.forward

            # Reset output tensors to zero for each benchmark
            # NOTE: C_exp should NOT be reset as it's a forward pass output used in backward
            for name in ['dWQ', 'dWK', 'dWV', 'dQ', 'dK', 'dV', 'dQ1', 'dK1', 'dV1',
                        'dO', 'dO_tmp', 'dC', 'dC_exp', 'dC_sum',
                        'Q1', 'K1', 'V1', 'Q2', 'K2', 'V2', 'Q', 'K', 'V',
                        'C', 'C_div', 'O', 'O1', 'O2', 'X2', 'X_norm']:
                if name in self.tensors:
                    self.tensors[name].zero_()

            # Build argument list based on metadata
            args = []
            for param in tensor_params:
                if param in self.tensors:
                    args.append(self.tensors[param])
                else:
                    # Create zero tensor if not exists (for intermediate tensors)
                    if param in self.tensor_shapes:
                        args.append(torch.zeros(self.tensor_shapes[param], dtype=torch.float32, device=self.device))
                    else:
                        raise ValueError(f"Unknown tensor parameter: {param}")

            stream = torch.cuda.Stream(self.device)
            # First call to trigger autotune (not counted in warmup)
            kernel_fn(*args)
            torch.cuda.synchronize()

            # Triton Warmup - now using the best configuration from autotune
            with torch.cuda.stream(stream):
                for _ in range(warmup_runs):
                    kernel_fn(*args)
            stream.synchronize()

            # CUDA Graph Warmup
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    kernel_fn(*args)
            # Synchronize before timing
            stream.synchronize()

            # Benchmark runs
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            with torch.cuda.stream(stream):
                start_event.record()
                for _ in range(benchmark_runs):
                    graph.replay()
                end_event.record()
            stream.synchronize()

            # Check for NaN values in gradient tensors
            for name in ['dWQ', 'dWK', 'dWV']:
                if name in self.tensors:
                    has_nan = torch.isnan(self.tensors[name]).any().item()
                    if has_nan:
                        print(f"WARNING: NaN values detected in [{ir_id}] {name} tensor!")

            # Return average time in milliseconds
            avg_time = (start_event.elapsed_time(end_event)) / benchmark_runs
            return avg_time

        except Exception as e:
            print(f"Error benchmarking kernel: {e}")
            traceback.print_exc()
            # Re-raise to let the caller handle it
            raise

    def run_single_benchmark(self, ir_id: int, ir_expr: str) -> BenchmarkResult:
        """Run benchmark for a single IR expression."""
        try:
            # Generate kernel code
            kernel_code = self.generate_kernel_code(ir_expr)
            if kernel_code is None:
                return BenchmarkResult(ir_id, ir_expr, float('inf'), self.tensor_config, "Failed to generate kernel")

            # Compile kernel
            kernel_module = self.compile_and_load_kernel(kernel_code, ir_id)
            if kernel_module is None:
                return BenchmarkResult(ir_id, ir_expr, float('inf'), self.tensor_config, "Failed to compile kernel")

            # Benchmark kernel
            exec_time = self.benchmark_kernel(kernel_module=kernel_module, ir_id=ir_id)

            # Clean up GPU memory after each benchmark
            self.cleanup_gpu()

            # Also clean up the loaded module to prevent memory leaks
            module_name = f"backward_kernel_{ir_id}"
            if module_name in sys.modules:
                del sys.modules[module_name]

            return BenchmarkResult(ir_id, ir_expr, exec_time, self.tensor_config)

        except Exception as e:
            # Clean up GPU memory even on error
            self.cleanup_gpu()
            return BenchmarkResult(ir_id, ir_expr, float('inf'), self.tensor_config, str(e))

    def run_all_benchmarks(self, ir_file: str, min_expressions: Optional[int], num: Optional[int] = None) -> List[BenchmarkResult]:
        """Run benchmarks for all backward IR expressions in the file."""
        # Parse IR expressions
        expressions = self.parse_ir_file(ir_file)

        # Filter by ir_id if min_expressions is provided
        if min_expressions is not None:
            # Find expressions with ir_id >= min_expressions
            filtered_expressions = [(ir_id, expr) for ir_id, expr in expressions if ir_id >= min_expressions]

            # If num is specified, take only the first 'num' expressions
            if num:
                filtered_expressions = filtered_expressions[:num]

            expressions = filtered_expressions

        print(f"Found {len(expressions)} backward IR expressions to benchmark")

        results = []
        # tqdm progress bar with update every 10 items
        with tqdm(total=len(expressions), desc="Benchmarking", unit="IR") as pbar:
            for i, (ir_id, ir_expr) in enumerate(expressions):
                result = self.run_single_benchmark(ir_id, ir_expr)
                results.append(result)

                # Update progress bar
                pbar.update(1)

                # Update postfix with current status every 10 items
                if (i + 1) % 10 == 0:
                    valid_so_far = sum(1 for r in results if r.error is None)
                    pbar.set_postfix(valid=valid_so_far, errors=len(results)-valid_so_far)

        return results

    def find_best_kernels(self, results: List[BenchmarkResult], top_k: int = 10) -> List[BenchmarkResult]:
        """Find the top-k fastest kernels."""
        # Filter out failed kernels
        valid_results = [r for r in results if r.error is None and r.execution_time != float('inf')]

        # Sort by execution time
        valid_results.sort(key=lambda x: x.execution_time)

        return valid_results[:top_k]

    def save_results(self, results: List[BenchmarkResult], output_file: str):
        """Save benchmark results to a JSON file."""
        data = []
        for r in results:
            data.append({
                'ir_id': r.ir_id,
                'ir_expression': r.ir_expression,
                'execution_time_ms': r.execution_time,
                'tensor_config': r.tensor_config,
                'error': r.error
            })

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def cleanup_gpu(self):
        """Clean up GPU memory and reset CUDA context."""
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                # Force synchronization
                torch.cuda.synchronize()

                # Clear all allocated tensors
                self.tensors.clear()

                # Multiple empty_cache calls to ensure thorough cleanup
                for _ in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()

                # Small delay to ensure GPU cleanup
                time.sleep(0.1)

                # Recreate test tensors to ensure clean state
                self.create_test_tensors()

        except Exception as e:
            print(f"Warning: GPU cleanup failed: {e}")

    def clear_triton_cache(self):
        """Clear Triton's cache directory to free up disk space."""
        try:
            # Get Triton cache directory
            triton_cache_dir = os.path.expanduser("~/.triton/cache")

            if os.path.exists(triton_cache_dir):
                # Remove all files and subdirectories in the cache
                for item in os.listdir(triton_cache_dir):
                    item_path = os.path.join(triton_cache_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)

                print(f"  Successfully cleared Triton cache at {triton_cache_dir}")
            else:
                print(f"  Triton cache directory not found at {triton_cache_dir}")

        except Exception as e:
            print(f"  Warning: Failed to clear Triton cache: {e}")

    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        self._temp_files = []


def run_comprehensive_benchmark(tensor_configs, ir_file, start_expressions, num_expressions, top_k, output_file, device_num=0):
    """Run benchmarks for all tensor shape configurations.

    Args:
        tensor_configs: List of tensor configuration dictionaries
        ir_file: Path to IR expressions file
        start_expressions: Starting expression ID
        num_expressions: Number of expressions to benchmark
        top_k: Number of top results to return
        output_file: Path to save results
        device_num: CUDA device number (default: 0)
    """
    all_results = []
    benchmark_instances = []

    print(f"Running comprehensive backward benchmark with:")
    print(f"  - {len(tensor_configs)} tensor configurations")
    print(f"  - Device: cuda:{device_num}")
    print()

    # Initialize output file with empty list
    with open(output_file, 'w') as f:
        json.dump([], f)

    for tensor_idx, tensor_config in enumerate(tensor_configs):
        print(f"\nTensor Configuration {tensor_idx + 1}/{len(tensor_configs)}: M={tensor_config['M']}, N={tensor_config['N']}, D={tensor_config['D']}, H={tensor_config['H']}")

        # Initialize benchmark with this tensor configuration
        benchmark = BackwardBenchmark(tensor_config, device_num)
        benchmark_instances.append(benchmark)

        try:
            # Run benchmarks for this tensor configuration
            results = benchmark.run_all_benchmarks(ir_file, min_expressions=start_expressions, num=num_expressions)

            # Store results with configuration info
            config_results = {
                'tensor_config': tensor_config,
                'results': results
            }
            all_results.append(config_results)

            # Save results incrementally
            save_incremental_results(config_results, output_file)
            print(f"  Saved results for configuration {tensor_idx + 1}/{len(tensor_configs)}")

        except Exception as e:
            print(f"  Error in configuration: {str(e)}")
            # Save error information
            error_result = {
                'tensor_config': tensor_config,
                'error': str(e),
                'results': []
            }
            all_results.append(error_result)
            save_incremental_results(error_result, output_file)

        # Clean up after each tensor configuration
        benchmark.cleanup()

    return all_results, benchmark_instances


def save_incremental_results(config_results, output_file):
    """Append results from one configuration to the output file."""
    # Read existing data
    try:
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
    except:
        existing_data = []

    # Add new results
    if 'error' in config_results and config_results['error']:
        # Handle error case
        existing_data.append({
            'tensor_config': config_results['tensor_config'],
            'error': config_results['error'],
            'results': []
        })
    else:
        # Add all results from this configuration
        for result in config_results['results']:
            existing_data.append({
                'ir_id': result.ir_id,
                'ir_expression': result.ir_expression,
                'execution_time_ms': result.execution_time,
                'tensor_config': result.tensor_config,
                'error': result.error
            })

    # Write back to file
    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=2)


def save_comprehensive_results(all_results, output_file):
    """Save all benchmark results with configuration details."""
    data = []

    for config_result in all_results:
        if 'error' in config_result and config_result['error']:
            # Handle error case
            data.append({
                'tensor_config': config_result['tensor_config'],
                'error': config_result['error'],
                'results': []
            })
        else:
            for result in config_result['results']:
                data.append({
                    'ir_id': result.ir_id,
                    'ir_expression': result.ir_expression,
                    'execution_time_ms': result.execution_time,
                    'tensor_config': result.tensor_config,
                    'error': result.error
                })

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def print_comprehensive_report(all_results, top_k):
    """Print comprehensive report showing best kernels for each configuration."""
    print("\n" + "="*100)
    print("COMPREHENSIVE BACKWARD BENCHMARK REPORT")
    print("="*100)

    # Group results by configuration
    for config_idx, config_result in enumerate(all_results):
        tensor_config = config_result['tensor_config']
        results = config_result['results']

        print(f"\nConfiguration {config_idx + 1}:")
        print(f"  Tensor Shape: M={tensor_config['M']}, N={tensor_config['N']}, D={tensor_config['D']}, H={tensor_config['H']}")

        # Find best kernels for this configuration
        valid_results = [r for r in results if r.error is None and r.execution_time != float('inf')]
        valid_results.sort(key=lambda x: x.execution_time)
        best_kernels = valid_results[:top_k]

        if best_kernels:
            print(f"  Top {min(len(best_kernels), top_k)} kernels:")
            for i, result in enumerate(best_kernels):
                print(f"    {i+1}. IR {result.ir_id}: {result.execution_time:.4f} ms")
                if i == 0:  # Show expression for best kernel only
                    print(f"       Expression: {result.ir_expression[:80]}...")
        else:
            print("  No valid kernels found for this configuration")

    # Overall best across all configurations
    print("\n" + "="*100)
    print("OVERALL BEST KERNELS ACROSS ALL CONFIGURATIONS")
    print("="*100)

    # Flatten all results
    all_valid_results = []
    for config_result in all_results:
        valid_results = [r for r in config_result['results'] if r.error is None and r.execution_time != float('inf')]
        all_valid_results.extend(valid_results)

    all_valid_results.sort(key=lambda x: x.execution_time)
    overall_best = all_valid_results[:top_k]

    for i, result in enumerate(overall_best):
        print(f"\n{i+1}. IR {result.ir_id}: {result.execution_time:.4f} ms")
        print(f"   Tensor Config: M={result.tensor_config['M']}, N={result.tensor_config['N']}, D={result.tensor_config['D']}, H={result.tensor_config['H']}")
        print(f"   Expression: {result.ir_expression[:100]}...")

    return overall_best

def save_top_k_results(top_results, output_file):
    """Save top k results to a JSON file."""
    data = []
    for rank, result in enumerate(top_results, 1):
        data.append({
            'rank': rank,
            'ir_id': result.ir_id,
            'execution_time_ms': result.execution_time,
            'tensor_config': result.tensor_config,
            'ir_expression': result.ir_expression
        })

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nTop {len(data)} results saved to: {output_file}")

def main():
    """Main function to run backward IR benchmarks."""
    # Default configuration
    START_EXPRESSIONS = 0
    NUM_EXPRESSIONS = 10
    TOP_K = 5

    parser = argparse.ArgumentParser(description="Run comprehensive backward IR benchmarks")
    parser.add_argument('--ir', type=str, default=None, help="Path to the backward IR expressions file")
    parser.add_argument('--seq', type=int, help="Sequence Length")
    parser.add_argument('--case', type=int, required=True, help="Choose BWD list case")
    parser.add_argument('--model', '-m', type=str, default='falcon', help="Model type (falcon, llama, etc.)")
    parser.add_argument('--device', '-d', type=int, default=0, help="CUDA device number (default: 0)")
    parser.add_argument('--output', type=str, default=None, help="Path to save benchmark results")
    parser.add_argument('--start', type=int, default=START_EXPRESSIONS, help="Start from test case ID")
    parser.add_argument('--num', type=int, default=NUM_EXPRESSIONS, help="Number of expressions to benchmark")
    parser.add_argument('--end', action='store_true', help="Run from start ID to the last test case")
    parser.add_argument('--topk', type=int, default=TOP_K, help="Number of top kernels to report")
    parser.add_argument('--all', action='store_true', help="Run all configurations comprehensively")
    parser.add_argument('--webhook', type=str, default=os.getenv('DISCORD_WEBHOOK'), help="Discord webhook URL for notifications")

    args = parser.parse_args()

    # Load model configuration from JSON
    try:
        tensor_config = load_model_config(args.model)
        TENSOR_CONFIGS = [tensor_config]  # Wrap in list for compatibility
    except Exception as e:
        print(f"Error loading model configuration: {e}")
        return

    # Set default paths based on case number if not provided
    if args.ir is None:
        args.ir = f"/home/chani227/Trinity/TileIR/expressions/data/bwd/seq{args.seq}_bwd{args.case}_list.txt"
    if args.output is None:
        args.output = f"/home/chani227/Trinity/Training/data/bwd_json/seq{args.seq}_bwd{args.case}.json"

    # Validate conflicting arguments
    if args.end and args.num != NUM_EXPRESSIONS:
        print("Error: Cannot use --end and --num together. Use either --num or --end, not both.")
        return

    total_expressions = 0
    if args.all:
        with open(args.ir, 'r') as f:
            total_expressions = len(f.readlines())
    elif args.end:
        total_expressions = None  # None means no limit
    else:
        total_expressions = args.num

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA device not available. Triton requires CUDA.")
        return

    # Run comprehensive benchmarks
    print("Starting comprehensive backward benchmarks...")
    all_results, benchmark_instances = run_comprehensive_benchmark(
        TENSOR_CONFIGS,
        args.ir,
        args.start,
        total_expressions,
        args.topk,
        args.output,
        args.device
    )

    print(f"\nAll results saved to: {args.output}")

    # Print comprehensive report and get top k results
    top_results = print_comprehensive_report(all_results, args.topk)

    # Save top k results to a separate file
    topk_output = args.output.replace('.json', '_topk.json')
    save_top_k_results(top_results, topk_output)

    # Send Discord notification if webhook URL is provided
    if args.webhook:
        # Calculate total results and errors
        total_count = sum(len(config_result['results']) for config_result in all_results)
        error_count = sum(1 for config_result in all_results
                         for result in config_result['results']
                         if result.error is not None)

        send_discord_notification(
            args.webhook,
            args.case,
            top_results,
            args.topk,
            total_count,
            error_count
        )

if __name__ == "__main__":
    main()
