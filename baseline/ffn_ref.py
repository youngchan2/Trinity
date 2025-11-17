import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt
import tempfile
import os

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
dtype = torch.float16

class SimpleFFN(nn.Module):
    def __init__(self, M, N, P, WO=None, WFF1a=None, WFF1b=None, WFF2=None):
        super().__init__()
        # Initialize weight tensors - use provided tensors or create new ones
        self.M = M
        self.N = N
        self.P = P
        self.WO = WO
        self.WFF1a = WFF1a
        self.WFF1b = WFF1b
        self.WFF2 = WFF2
        
    def forward(self, O2, X):
        # attn_O1 = O2 * WO
        attn_O1 = torch.matmul(O2, self.WO)
        
        # attn_O2 = attn_O1 + X
        attn_O2 = attn_O1 + X
        
        # attn_O3 = reduce_sum(sqr(attn_O2), 1)
        attn_O3 = torch.sum(attn_O2 ** 2, dim=1, keepdim=True)
        
        # attn_O_norm = attn_O2 / bcast(sqrt(attn_O3 / 4544), 1)
        attn_O_norm = attn_O2 / torch.sqrt(attn_O3 / self.N)
        
        # FF1a = attn_O_norm * WFF1a
        FF1a = torch.matmul(attn_O_norm, self.WFF1a)
        
        # FF1b = attn_O_norm * WFF1b
        FF1b = torch.matmul(attn_O_norm, self.WFF1b)
        
        # FF1b_silu = FF1b x sigmoid(FF1b)
        FF1b_silu = FF1b * torch.sigmoid(FF1b)
        
        # FF1 = FF1a x FF1b_silu (element-wise multiplication)
        FF1 = FF1a * FF1b_silu
        
        # FF2 = FF1 * WFF2
        FF2 = torch.matmul(FF1, self.WFF2)
        
        return FF2


class TRT_Falcon_Ffn(nn.Module):
    def __init__(self, M, N, WO=None, WFF1a=None, WFF1b=None, WFF2=None):
        super().__init__()
        # Initialize weight tensors - use provided tensors or create new ones
        self.M = M
        self.N = N
        
        self.WO = WO
        self.WFF1a = WFF1a
        self.WFF1b = WFF1b
        self.WFF2 = WFF2
        
        # Pre-allocate output tensor
        self.output = torch.empty((self.M, self.N), dtype=dtype, device=device)
        
        self.engine = None
        self.context = None
        self.build_engine()
        
    def build_engine(self):
        # Create a model that performs the tensor operations
        class TensorOpsModel(nn.Module):
            def __init__(self, N, WO, WFF1a, WFF1b, WFF2):
                super().__init__()
                self.N = N
                
                self.WO = WO
                self.WFF1a = WFF1a
                self.WFF1b = WFF1b
                self.WFF2 = WFF2
                
            def forward(self, O2, X):
                # attn_O1 = O2 * WO
                attn_O1 = torch.matmul(O2, self.WO)
                
                # attn_O2 = attn_O1 + X
                attn_O2 = attn_O1 + X
                
                # attn_O3 = reduce_sum(sqr(attn_O2), 1)
                attn_O3 = torch.sum(attn_O2 ** 2, dim=1, keepdim=True)
                
                # attn_O_norm = attn_O2 / bcast(sqrt(attn_O3 / 4544), 1)
                attn_O_norm = attn_O2 / torch.sqrt(attn_O3 / self.N)
                
                # FF1a = attn_O_norm * WFF1a
                FF1a = torch.matmul(attn_O_norm, self.WFF1a)
                
                # FF1b = attn_O_norm * WFF1b
                FF1b = torch.matmul(attn_O_norm, self.WFF1b)
                
                # FF1b_silu = FF1b x sigmoid(FF1b)
                FF1b_silu = FF1b * torch.sigmoid(FF1b)
                
                # FF1 = FF1a x FF1b_silu
                FF1 = FF1a * FF1b_silu
                
                # FF2 = FF1 * WFF2
                FF2 = torch.matmul(FF1, self.WFF2)
                
                # O_FF = FF2 + attn_O_norm
                O_FF = FF2 + attn_O_norm
                
                # O_FF1 = reduce_sum(sqr(O_FF), 1)
                O_FF1 = torch.sum(O_FF ** 2, dim=1, keepdim=True)
                
                # O_FF_norm = O_FF / bcast(sqrt(O_FF1 / 4544), 1)
                O_FF_norm = O_FF / torch.sqrt(O_FF1 / 4544)

                return O_FF_norm
        
        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()
        
        try:
            # Create model with weights
            model = TensorOpsModel(self.N, self.WO, self.WFF1a, self.WFF1b, self.WFF2)
            
            # Dummy inputs for export
            dummy_O2 = torch.randn(self.M, self.N, dtype=dtype, device=device)
            dummy_X = torch.randn(self.M, self.N, dtype=dtype, device=device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_O2, dummy_X),
                onnx_path,
                input_names=['O2', 'X'],
                output_names=['O_FF_norm'],
                opset_version=13,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
    
    def forward(self, O2, X):
        # Create bindings using pre-allocated output tensor
        bindings = [
            O2.data_ptr(),
            X.data_ptr(),
            self.output.data_ptr()
        ]
        
        # Execute
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return self.output

class TRT_Llama_Ffn(nn.Module):
    def __init__(self, M, N, P, WO=None, WFF1a=None, WFF1b=None, WFF2=None):
        super().__init__()
        # Initialize weight tensors - use provided tensors or create new ones
        self.M = M
        self.N = N
        self.P = P
        
        self.WO = WO
        self.WFF1a = WFF1a
        self.WFF1b = WFF1b
        self.WFF2 = WFF2
        
        # Pre-allocate output tensor
        self.output = torch.empty((self.M, self.N), dtype=dtype, device=device)
        
        self.engine = None
        self.context = None
        self.build_engine()
        
    def build_engine(self):
        # Create a model that performs the tensor operations
        class TensorOpsModel(nn.Module):
            def __init__(self, N, WO, WFF1a, WFF1b, WFF2):
                super().__init__()
                self.N = N
                
                self.WO = WO
                self.WFF1a = WFF1a
                self.WFF1b = WFF1b
                self.WFF2 = WFF2
                
            def forward(self, O2, X):
                # attn_O1 = O2 * WO
                attn_O1 = torch.matmul(O2, self.WO)
                
                # attn_O2 = attn_O1 + X
                attn_O2 = attn_O1 + X
                
                # attn_O3 = reduce_sum(sqr(attn_O2), 1)
                attn_O3 = torch.sum(attn_O2 ** 2, dim=1, keepdim=True)
                
                # attn_O_norm = attn_O2 / bcast(sqrt(attn_O3 / 4544), 1)
                attn_O_norm = attn_O2 / torch.sqrt(attn_O3 / self.N)
                
                # FF1a = attn_O_norm * WFF1a
                FF1a = torch.matmul(attn_O_norm, self.WFF1a)
                
                # FF1b = attn_O_norm * WFF1b
                FF1b = torch.matmul(attn_O_norm, self.WFF1b)
                
                # FF1b_silu = FF1b x sigmoid(FF1b)
                FF1b_silu = FF1b * torch.sigmoid(FF1b)
                
                # FF1 = FF1a x FF1b_silu
                FF1 = FF1a * FF1b_silu
                
                # FF2 = FF1 * WFF2
                FF2 = torch.matmul(FF1, self.WFF2)

                return FF2
        
        onnx_file = tempfile.NamedTemporaryFile(suffix='.onnx', delete=False)
        onnx_path = onnx_file.name
        onnx_file.close()
        
        try:
            # Create model with weights
            model = TensorOpsModel(self.N, self.WO, self.WFF1a, self.WFF1b, self.WFF2)
            
            # Dummy inputs for export
            dummy_O2 = torch.randn(self.M, self.N, dtype=dtype, device=device)
            dummy_X = torch.randn(self.M, self.N, dtype=dtype, device=device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                (dummy_O2, dummy_X),
                onnx_path,
                input_names=['O2', 'X'],
                output_names=['FF2'],
                opset_version=13,
                do_constant_folding=True
            )
            
            # Build TensorRT engine
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    raise RuntimeError('Failed to parse ONNX file')
            
            # Configure builder
            config = builder.create_builder_config()
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
            
            # Deserialize the engine
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")
                
            self.context = self.engine.create_execution_context()
            
        finally:
            # Clean up ONNX file
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
    
    def forward(self, O2, X):
        # Create bindings using pre-allocated output tensor
        bindings = [
            O2.data_ptr(),
            X.data_ptr(),
            self.output.data_ptr()
        ]
        
        # Execute
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("TensorRT execution failed")
        
        return self.output

# def benchmark_tensor_program():
#     """Benchmark both PyTorch and TensorRT implementations"""
#     ITER = 100
    
#     # Create input tensors
#     O2 = torch.randn((16, 4544), device=device, dtype=dtype).clamp(-1, 1)*0.01
#     X = torch.randn((16, 4544), device=device, dtype=dtype).clamp(-1, 1)*0.01
#     attn_O_norm = torch.randn((16, 4544), device=device, dtype=dtype).clamp(-1, 1)*0.01
#     FF1 = torch.randn((16, 4544), device=device, dtype=dtype).clamp(-1, 1)*0.01

#     # PyTorch implementation
#     # print("Benchmarking FFN sub implementation...")
#     # pytorch_model = PyTorchTensorProgram().cuda()
#     # pytorch_model.half()
    
#     # with torch.no_grad():
#         # Warmup
#     #     for _ in range(10):
#     #         out = pytorch_model(O2, X)
#     #     torch.cuda.synchronize()
        
#     #     # Benchmark
#     #     start_event = torch.cuda.Event(enable_timing=True)
#     #     end_event = torch.cuda.Event(enable_timing=True)
        
#     #     start_event.record()
#     #     for _ in range(ITER):
#     #         out = pytorch_model(O2, X)
#     #     end_event.record()
#     #     torch.cuda.synchronize()
        
#     #     elapsed_time = start_event.elapsed_time(end_event)
#     # print(f"PyTorch: {elapsed_time/ITER:.3f}ms per iteration")
    
#     # TensorRT implementation
#     print("\nBenchmarking MLP implementation...")
#     trt_model = TensorRTTensorProgram().to(device=device)
#     trt_model.half()
    
#     with torch.no_grad():
#         # Warmup
#         for _ in range(10):
#             out = trt_model(O2, X)
#         torch.cuda.synchronize()
        
#         # Benchmark
#         start_event = torch.cuda.Event(enable_timing=True)
#         end_event = torch.cuda.Event(enable_timing=True)
        
#         start_event.record()
#         for _ in range(ITER):
#             out = trt_model(O2, X)
#         end_event.record()
#         torch.cuda.synchronize()
        
#         elapsed_time = start_event.elapsed_time(end_event)
#     print(f"TensorRT: {elapsed_time/ITER:.3f}ms per iteration")
    
#     print("\nBenchmarking FFN part1 implementation...")
#     trt_model = TensorRTFFN_part1Program().to(device=device)
#     trt_model.half()
    
#     with torch.no_grad():
#         # Warmup
#         for _ in range(10):
#             out = trt_model(X, O2)
#         torch.cuda.synchronize()
        
#         # Benchmark
#         start_event = torch.cuda.Event(enable_timing=True)
#         end_event = torch.cuda.Event(enable_timing=True)
        
#         start_event.record()
#         for _ in range(ITER):
#             out = trt_model(X, O2)
#         end_event.record()
#         torch.cuda.synchronize()
        
#         elapsed_time = start_event.elapsed_time(end_event)
#     print(f"FFN part1: {elapsed_time/ITER:.3f}ms per iteration")

#     print("\nBenchmarking FFN part2 implementation...")
#     trt_model = TensorRTFFN_part2Program().to(device=device)
#     trt_model.half()
    
#     with torch.no_grad():
#         # Warmup
#         for _ in range(10):
#             out = trt_model(FF1, attn_O_norm)
#         torch.cuda.synchronize()
        
#         # Benchmark
#         start_event = torch.cuda.Event(enable_timing=True)
#         end_event = torch.cuda.Event(enable_timing=True)
        
#         start_event.record()
#         for _ in range(ITER):
#             out = trt_model(FF1, attn_O_norm)
#         end_event.record()
#         torch.cuda.synchronize()
        
#         elapsed_time = start_event.elapsed_time(end_event)
#     print(f"FFN part2: {elapsed_time/ITER:.3f}ms per iteration")

#     # # Verify outputs match
#     # print("\nVerifying outputs...")
#     # with torch.no_grad():
#     #     pytorch_out = pytorch_model(O2, X)
#     #     trt_out = trt_model(attn_O_norm)
        
#     #     # Check if outputs are close (allowing for some numerical differences)
#     #     max_diff = torch.max(torch.abs(pytorch_out - trt_out)).item()
#     #     print(f"Max difference between PyTorch and TensorRT outputs: {max_diff}")
        
#     #     if max_diff < 0.01:
#     #         print("✓ Outputs match within tolerance")
#     #     else:
#     #         print("✗ Outputs differ significantly")


# if __name__ == "__main__":
#     print(torch.cuda.get_device_name(device))
#     benchmark_tensor_program()