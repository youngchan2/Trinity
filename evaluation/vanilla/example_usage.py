"""
GPU Utilization 측정 예시

vanilla_falcon_benchmark0.py 에서 gpu_utils를 사용하는 방법들
"""

import sys
sys.path.append('/home/chani227/Triton')

import torch
from vanilla_falcon_benchmark0 import forward, kernel_0
from gpu_utils import (
    GPUMonitor,
    get_gpu_info,
    monitor_kernel_execution,
    print_gpu_stats,
    calculate_memory_bandwidth_util
)

# 입력 텐서 준비
def prepare_inputs():
    M, N, D = 16, 4544, 64
    O2 = torch.zeros((M, N), dtype=torch.float16, device='cuda')
    WK = torch.randn((N, N), dtype=torch.float16, device='cuda')
    WQ = torch.randn((N, N), dtype=torch.float16, device='cuda')
    WV = torch.randn((N, N), dtype=torch.float16, device='cuda')
    X = torch.randn((M, N), dtype=torch.float16, device='cuda')
    return O2, WK, WQ, WV, X


# ===== 방법 1: Context Manager 사용 =====
def method1_context_manager():
    print("\n" + "="*70)
    print("방법 1: Context Manager로 GPU Utilization 측정")
    print("="*70)

    O2, WK, WQ, WV, X = prepare_inputs()

    # Warmup
    for _ in range(10):
        forward(O2, WK, WQ, WV, X)
    torch.cuda.synchronize()

    # 모니터링하면서 커널 실행
    with GPUMonitor(device_id=0, interval=0.001) as monitor:
        # 시간 측정
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(100):
            forward(O2, WK, WQ, WV, X)
        end.record()
        torch.cuda.synchronize()

    # 결과 가져오기
    stats = monitor.stop()
    stats['avg_time_ms'] = start.elapsed_time(end) / 100

    print_gpu_stats(stats)


# ===== 방법 2: 수동으로 start/stop =====
def method2_manual():
    print("\n" + "="*70)
    print("방법 2: 수동으로 start/stop 제어")
    print("="*70)

    O2, WK, WQ, WV, X = prepare_inputs()

    monitor = GPUMonitor(device_id=0, interval=0.001)

    # Warmup
    for _ in range(10):
        forward(O2, WK, WQ, WV, X)
    torch.cuda.synchronize()

    # 모니터링 시작
    monitor.start()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(100):
        forward(O2, WK, WQ, WV, X)

    end.record()
    torch.cuda.synchronize()

    # 모니터링 중지
    stats = monitor.stop()
    stats['avg_time_ms'] = start.elapsed_time(end) / 100

    print_gpu_stats(stats, verbose=False)


# ===== 방법 3: 래퍼 함수 사용 =====
def method3_wrapper():
    print("\n" + "="*70)
    print("방법 3: monitor_kernel_execution 래퍼 함수 사용")
    print("="*70)

    O2, WK, WQ, WV, X = prepare_inputs()

    # 자동으로 warmup과 반복 실행
    stats = monitor_kernel_execution(
        forward,
        O2, WK, WQ, WV, X,
        warmup=10,
        iterations=100,
        device_id=0
    )

    print_gpu_stats(stats)


# ===== 방법 4: Memory Bandwidth 계산 =====
def method4_bandwidth():
    print("\n" + "="*70)
    print("방법 4: Memory Bandwidth Utilization 계산")
    print("="*70)

    O2, WK, WQ, WV, X = prepare_inputs()

    # Warmup
    for _ in range(10):
        forward(O2, WK, WQ, WV, X)
    torch.cuda.synchronize()

    # 시간 측정
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    forward(O2, WK, WQ, WV, X)
    end.record()
    torch.cuda.synchronize()

    time_ms = start.elapsed_time(end)

    # 메모리 전송량 계산
    # 입력: X (16x4544), WQ, WK, WV (각 4544x4544)
    # 출력: O2 (16x4544)
    bytes_per_element = 2  # float16

    bytes_read = (
        16 * 4544 +          # X
        3 * 4544 * 4544      # WQ, WK, WV
    ) * bytes_per_element

    bytes_write = 16 * 4544 * bytes_per_element  # O2

    total_bytes = bytes_read + bytes_write

    achieved, peak, util = calculate_memory_bandwidth_util(
        total_bytes,
        time_ms,
        device_id=0
    )

    print(f"Kernel Execution Time:     {time_ms:.4f} ms")
    print(f"Total Data Transfer:       {total_bytes / 1e9:.4f} GB")
    print(f"Achieved Bandwidth:        {achieved:.2f} GB/s")
    print(f"Peak Bandwidth:            {peak:.2f} GB/s")
    print(f"Memory BW Utilization:     {util:.2f}%")
    print("="*70)


# ===== GPU 정보 확인 =====
def show_gpu_info():
    print("\n" + "="*70)
    print("현재 GPU 정보")
    print("="*70)

    info = get_gpu_info(device_id=0)
    for key, value in info.items():
        if isinstance(value, float):
            print(f"{key:.<30} {value:.2f}")
        else:
            print(f"{key:.<30} {value}")
    print("="*70)


if __name__ == "__main__":
    # GPU 정보 먼저 확인
    show_gpu_info()

    # 각 방법 실행
    method1_context_manager()
    method2_manual()
    method3_wrapper()
    method4_bandwidth()

    print("\n추천: 간단한 측정은 method3 (wrapper), 상세한 제어는 method1 (context manager)")
