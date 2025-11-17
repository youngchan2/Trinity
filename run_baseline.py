from baseline.inductor import benchmark_rms
import argparse
import torch
import torch._inductor.config as inductor_config

parser = argparse.ArgumentParser(description="Convert IR to Triton kernel")
parser.add_argument("--m", type=str, default="llama", help="Input model type")
parser.add_argument("--t", type=str, default="vanilla", help="RMS Type")
parser.add_argument("--b", type=str, help="Baseline Type")
parser.add_argument("--pre", action="store_true", help="Whether to use prenorm or not")
args = parser.parse_args()

model = args.m
rms = args.t
base = args.b
pre = args.pre
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
dtype = torch.float16

'''
 1. QKV Projection (Line 1808-1810)

  q = torch.matmul(X, self.W_q).view(self.M, self.H, self.D)
  k = torch.matmul(X, self.W_k).view(self.M, self.H, self.D)
  v = torch.matmul(X, self.W_v).view(self.M, self.H, self.D)
  커널: ampere_fp16_s16816gemm_fp16_64x64...
  - SM Throughput: ~29-30%
  - 횟수: 3번 (Q, K, V 각각)
  - 중요도: 높음 (compute-bound 연산)

  2. K, V Cache Concatenation (Line 1816-1817)

  k_cache = torch.cat([self.cache_K[:, :self.P, :], k], dim=1)
  v_cache = torch.cat([self.cache_V[:, :self.P, :], v], dim=1)
  커널: CatArrayBatchedCopy
  - SM Throughput: ~54-56%
  - 횟수: 2번 (K, V 각각)
  - 중요도: 중간 (memory-bound 연산)

  3. FlashInfer Attention (Line 1819-1826)

  output = flashinfer.single_prefill_with_kv_cache(...)
  커널:
  - SinglePrefillWithKVCacheKernel (메인 attention 연산)
    - SM Throughput: ~8.89-8.90% ⚠️
  - MergeStatesKernel (멀티 헤드 merge)
    - SM Throughput: ~2.46-2.49% ⚠️

  중요도: 매우 높음 (핵심 연산이지만 활용률 낮음)

  주목해야 할 커널

  가장 중요하게 봐야 할 커널:

  1. SinglePrefillWithKVCacheKernel (8.9% utilization)
    - Attention의 핵심 연산
    - 활용률이 매우 낮아 최적화 여지 큼
  2. ampere_fp16_s16816gemm (29-30% utilization)
    - 전체 시간의 상당 부분 차지
    - 3번 호출되므로 누적 시간이 큼
  3. MergeStatesKernel (2.5% utilization)
    - 활용률이 가장 낮음
    - FlashInfer의 병목 지점일 가능성
'''

'''
 NCU 프로파일링 결과 매칭

  | 연산                         | 커널                                          | GPU 활용도       | 비고
      |
  |----------------------------|---------------------------------------------|---------------|---------------|
  | 입력 데이터 준비                  | distribution_elementwise_grid... (랜덤 초기화)   | 69%, 60%, 34% |
  테스트용 더미 입력 생성 |
  | QKV Projection (3x matmul) | ampere_fp16_s16816gemm_fp16_64x64...        | 29-30%        | ⚠️ 매우 낮음      |
  | KV Cache Concat            | CatArrayBatchedCopy                         | 54-55%        | 중간            |
  | Attention Scores (matmul)  | cutlass_80_wmma_tensorop_f16_s161616gemm... | 19%           | ⚠️ 매우 낮음      |
  | Softmax                    | softmax_warp_forward                        | 30%           | 낮음            |
  | Output (matmul)            | cutlass_80_wmma_tensorop_f16_s161616gemm... | 13%           | ⚠️ 매우 낮음      |
  | TensorRT 내부 변환             | copyVectorizedKernel                        | 8%            | ⚠️ 병목
   |
  | 최종 TensorRT GEMM           | sm80_xmma_gemm_f32f32_f32f32...             | 46%           | 중간            |
'''

if model == "falcon":
    M = 16
    D = 64
    N = 4544
    P = 1024
    H = 71
    
    constants = {
        'M': 16,
        'D': 64,
        'N': 4544,
        'P': 1024,
        'H': 71
    }
elif model == "llama":
    M = 16
    D = 128
    N = 4096
    H = 32
    P = 1024
    
    constants = {
        'M': 16,
        'D': 128,
        'N': 4096,
        'P': 1024,
        'H': 32
    }

tensor_shapes = {
            'X': ('M', 'N'),
            'X2': ('M', ),
            'X_norm': ('M', 'N'),
            
            'WQ': ('N', 'N'),
            'WK': ('N', 'N'),
            'WV': ('N', 'N'),

            'Q': ('H', 'M', 'D'),
            'K': ('H', 'M', 'D'),
            'V': ('H', 'M', 'D'),

            'Q1': ('M', 'N'),
            'K1': ('M', 'N'),
            'V1': ('M', 'N'),

            'Q2': ('M', 'H', 'D'),
            'K2': ('M', 'H', 'D'),
            'V2': ('M', 'H', 'D'),

            'K_cache': ('H', 'P+M', 'D'),
            'V_cache': ('H', 'P+M', 'D'),

            'K': ('H', 'M', 'D'),
            'V': ('H', 'M', 'D'),

            'O': ('H', 'M', 'D'),
            'O1': ('M', 'H', 'D'),
            'O2': ('M', 'N'),

            'C': ('H', 'M', 'P'),
            'C_exp': ('H', 'M', 'P+M'),
            'C_div': ('H', 'M', 'P+M'),
            'C_sum': ('H', 'M'),
            'noise': ('H', 'M', 'P+M'),
            'C_perturb': ('H', 'M', 'P+M'),
            'C_exp_perturb': ('H', 'M', 'P+M'),
            'C_sum_perturb': ('H', 'M', 'P+M'),
            'C_div_perturb': ('H', 'M', 'P+M'),
            'C_out': ('H', 'P+M'),
            'C_out1': ('H', 'P+M'),
            'C_out2': ('H', 'P+M'),

            'Q_norm': ('H', 'M', 'D'),
            'K_norm': ('H', 'M', 'D')
        }

def main():

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Use much smaller values to avoid numerical overflow
    std = 0.01
    X = torch.randn((M, N), device=device, dtype=dtype) * std
    WQ = torch.randn((N, N), device=device, dtype=dtype) * std
    WK = torch.randn((N, N), device=device, dtype=dtype) * std
    WV = torch.randn((N, N), device=device, dtype=dtype) * std
    K_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std
    V_cache = torch.randn((H, P+M, D), device=device, dtype=dtype) * std
    O2 = torch.zeros((M, N), device=device, dtype=dtype)
    # Key Former
    noise = torch.randn((H, M, P+M), device=device, dtype=dtype)

    out = O2.clone()

    match rms:
        case "vanilla":
            if base == "trt":
                from baseline.ref_rms import TensorRT_Vanilla
                trt = TensorRT_Vanilla(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
            elif base == "inductor":
                from baseline.ref_rms import Vanilla
                ti = Vanilla(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            elif base == "flashinfer":
                from baseline.ref_rms import FlashInfer_Vanilla
                fi = FlashInfer_Vanilla(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
        case "prenorm":
            from baseline.ref_rms import PreNorm, TensorRT_PreNorm, FlashInfer_PreNorm
            trt = TensorRT_PreNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
            ti = PreNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            fi = FlashInfer_PreNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
        case "keyformer":
            if pre:
                from baseline.ref_rms import NormKeyFormer, TensorRT_NormKeyFormer, FlashInfer_NormKeyFormer
                trt = TensorRT_NormKeyFormer(M, N, D, H, K_cache.clone(), V_cache.clone(), P, noise, WQ, WK, WV)
                ti = NormKeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
                fi = FlashInfer_NormKeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            else:
                from baseline.ref_rms import KeyFormer, TensorRT_KeyFormer, FlashInfer_KeyFormer
                trt = TensorRT_KeyFormer(M, N, D, H, K_cache.clone(), V_cache.clone(), P, noise, WQ, WK, WV)
                ti = KeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
                fi = FlashInfer_KeyFormer(M, N, D, P, noise, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
        case "qknorm":
            if pre:
                from baseline.ref_rms import NormQKNorm, TensorRT_NormQKNorm, FlashInfer_NormQKNorm
                trt = TensorRT_NormQKNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
                ti = NormQKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
                fi = FlashInfer_NormQKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            else:
                from baseline.ref_rms import QKNorm, TensorRT_QKNorm, FlashInfer_QKNorm
                trt = TensorRT_QKNorm(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
                ti = QKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
                fi = FlashInfer_QKNorm(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
        case "roco":
            if pre:
                from baseline.ref_rms import NormRoCo, TensorRT_NormRoCo, FlashInfer_NormRoCo
                trt = TensorRT_NormRoCo(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
                ti = NormRoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
                fi = FlashInfer_NormRoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
            else:
                from baseline.ref_rms import RoCo, TensorRT_RoCo, FlashInfer_RoCo
                trt = TensorRT_RoCo(M, N, D, H, K_cache.clone(), V_cache.clone(), P, WQ, WK, WV)
                ti = RoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
                fi = FlashInfer_RoCo(M, N, D, P, K_cache.clone(), V_cache.clone(), WQ, WK, WV)
    ITER = 15
    # print("=" * 50)
    # print("Starting ref kernel execution...")
    # print(f"\nTesting {rms.upper()} Flash Infer...")
    # print("\nTesting Correct Flash Infer...")
    # ------------------- Flash Infer -------------------
    if base == "flashinfer":
        fi.half()
        with torch.no_grad():
            for _ in range(10):
                out = fi(X)
            torch.cuda.synchronize()

            start_fi = torch.cuda.Event(enable_timing=True)
            end_fi = torch.cuda.Event(enable_timing=True)

            start_fi.record()
            for _ in range(ITER):
                out = fi(X)
            end_fi.record()
            torch.cuda.synchronize()
            fi_time = start_fi.elapsed_time(end_fi) / ITER
        #     print(f"FI: {fi_time}ms")
        # print(out)


    # print("=" * 50)
    # print(f"\nTesting {rms.upper()} TensorRT...")
    # print("\nTesting Correct TensorRT...")
    # # ------------------- Tensor RT -------------------
    elif base == "trt":
        trt.half()
        with torch.no_grad():
            for _ in range(10):
                out = trt(X)
            torch.cuda.synchronize()

            start_rt = torch.cuda.Event(enable_timing=True)
            end_rt = torch.cuda.Event(enable_timing=True)

            start_rt.record()
            for _ in range(ITER):
                out = trt(X)
            end_rt.record()
            torch.cuda.synchronize()
            rt_time = start_rt.elapsed_time(end_rt) / ITER
        #     print(f"TRT: {rt_time}ms")
        # print(out)

    # # ------------------- Torch Inductor -------------------
    # print("\nTesting Torch Inductor Implementation...")
    elif base == "inductor":
        # NCU 프로파일링 호환성을 위한 설정
        inductor_config.triton.cudagraphs = False  # CUDA graphs 비활성화

        # torch.compile with max-autotune
        ti = torch.compile(ti.eval(), backend="inductor", mode="default", fullgraph=False)

        # 충분한 워밍업으로 JIT 컴파일 완료 보장
        with torch.no_grad():
            for _ in range(20):
                _ = ti(X)

        torch.cuda.synchronize()

        # NCU 프로파일링을 위한 실행
        with torch.no_grad():
            for _ in range(ITER):
                _ = ti(X)

        torch.cuda.synchronize()
        
        # print("\nComparing results...")
        # if torch.allclose(O2, out, rtol=1e-3, atol=1e-4):
        #     print("✓ Results match!")
        # else:
        #     print("✗ Results do not match!")
        #     max_diff = torch.abs(O2 - out).max()
        #     print(f"Maximum difference: {max_diff}")
        # print("=" * 50)

if __name__ == "__main__":
    main()
