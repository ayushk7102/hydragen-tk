import torch
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydragen.attention import hydragen_attention, hydragen_attention_tk
from hydragen.attention_tk import can_use_tk

def run_comparison_test(test_name, seq_len_shared, seq_len_unique, expect_tk_usage):
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"Shared Len: {seq_len_shared} | Unique Len: {seq_len_unique}")
    print(f"Expect TK Kernel Usage: {expect_tk_usage}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16  # Use BF16 to match TK precision

    # Configuration
    B = 2           # Total Batch Size
    NQ = 1          # Queries per sequence (Decoding step)
    H = 16          # Heads
    D = 128         # Head Dim (Must be 128 or 64 for TK)
    
    # Shared Prefix Configuration
    # We share 1 prefix across the whole batch (sbatch=1)
    shared_bs = 1
    
    # 1. Create Inputs
    q = torch.randn(B, NQ, H, D, device=device, dtype=dtype)
    k_unique = torch.randn(B, seq_len_unique, H, D, device=device, dtype=dtype)
    v_unique = torch.randn(B, seq_len_unique, H, D, device=device, dtype=dtype)

    # Shared KVs
    shared_k = torch.randn(shared_bs, seq_len_shared, H, D, device=device, dtype=dtype)
    shared_v = torch.randn(shared_bs, seq_len_shared, H, D, device=device, dtype=dtype)

    # 2. Run Baseline (Pure FlashAttention)
    print("Running Baseline (hydragen_attention)...")
    out_baseline = hydragen_attention(
        q, k_unique, v_unique,
        shared_ks=[shared_k],
        shared_vs=[shared_v],
        shared_cu_seq_lens=[None],
        shared_max_seq_lens=[None],
        use_varlens=[False],
        seq_lens=None
    )

    # 3. Run Optimized (ThunderKittens Integration)
    print("Running Optimized (hydragen_attention_tk)...")
    
    # Check if logic works as expected
    will_use_tk = can_use_tk(seq_len_shared, D)
    if will_use_tk != expect_tk_usage:
        print(f"WARNING: can_use_tk returned {will_use_tk}, expected {expect_tk_usage}")

    out_tk = hydragen_attention_tk(
        q, k_unique, v_unique,
        shared_ks=[shared_k],
        shared_vs=[shared_v],
        shared_cu_seq_lens=[None],
        shared_max_seq_lens=[None],
        use_varlens=[False],
        seq_lens=None
    )

    # 4. Compare
    diff = (out_baseline - out_tk).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    print(f"\nResults:")
    print(f"  Mean Difference: {mean_diff:.6f}")
    print(f"  Max Difference:  {max_diff:.6f}")

    # Threshold: slightly loose because FlashAttn and TK accumulate differently
    threshold = 0.05 
    
    if mean_diff < threshold:
        print(">> SUCCESS: Outputs match.")
    else:
        print(">> FAILURE: Significant mismatch detected.")

if __name__ == "__main__":
    # Test 1: Long sequence (Multiple of 192) -> Should use TK
    run_comparison_test(
        test_name="Long Prefix (Optimized Path)",
        seq_len_shared=768,  # 192 * 4
        seq_len_unique=32,
        expect_tk_usage=True
    )

    # Test 2: Short sequence -> Should fallback to FlashAttn
    # The outputs should be IDENTICAL (diff=0.0) because both paths use FlashAttn
    run_comparison_test(
        test_name="Short Prefix (Fallback Path)",
        seq_len_shared=100,
        seq_len_unique=32,
        expect_tk_usage=False
    )