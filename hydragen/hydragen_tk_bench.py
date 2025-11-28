import torch
import sys, os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydragen.attention import hydragen_attention_tk
from hydragen.attention_tk import can_use_tk

def benchmark_tk():
    print(f"{'='*60}")
    print(f"BENCHMARK: Hydragen w/ ThunderKittens (H100 Optimized)")
    print(f"{'='*60}")
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # 1. Setup Data for a Prefill Scenario
    # Batch=16 to saturate the GPU
    # Seq=768 (Multiple of 192 for TK)
    # HeadDim=128 (TK requirement)
    B, S, H, D = 16, 768, 16, 128
    
    # Ensure this configuration actually uses TK
    if not can_use_tk(S, S, D):
        print("ERROR: Config will NOT trigger TK. Aborting benchmark.")
        return

    # Create Tensors
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)
    
    # Dummy shared lists (we test the 'unique' path which is self-attention)
    # This is equivalent to prefilling the prompt.
    
    # 2. Warmup
    print("Warming up...")
    for _ in range(10):
        _ = hydragen_attention_tk(
            q, k, v, 
            shared_ks=[], shared_vs=[], 
            shared_cu_seq_lens=[], shared_max_seq_lens=[], use_varlens=[], seq_lens=None
        )
    torch.cuda.synchronize()
    
    # 3. Benchmark
    iters = 100
    print(f"Running {iters} iterations...")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        _ = hydragen_attention_tk(
            q, k, v, 
            shared_ks=[], shared_vs=[], 
            shared_cu_seq_lens=[], shared_max_seq_lens=[], use_varlens=[], seq_lens=None
        )
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / iters
    
    # 4. Calculate TFLOPS
    # FLOPs for Causal Attention = 2 * B * H * S^2 * D (approx) for Fwd
    # (4 for fwd+bwd, but we only run fwd here)
    # Standard formula for Attention FWD is 4 * B * H * S^2 * D if counting MACs as 2 ops
    # Let's use: 4 * B * H * S * S * D
    
    flops = 4 * B * H * (S**2) * D
    tflops = (flops / (avg_ms / 1000)) / 1e12
    
    print(f"\nResults:")
    print(f"  Avg Latency: {avg_ms:.4f} ms")
    print(f"  Est. Throughput: {tflops:.2f} TFLOPS")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    benchmark_tk()