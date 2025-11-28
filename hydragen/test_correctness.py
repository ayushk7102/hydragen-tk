import torch
import sys
import os

# Allow imports from current directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydragen.attention import hydragen_attention_tk
from hydragen.attention_tk import can_use_tk

def verify_tk_usage(q_len, k_len, d):
    """Ensure our test parameters actually trigger the TK kernel"""
    if not can_use_tk(q_len, k_len, d):
        raise ValueError(
            f"Test Configuration Error: Parameters Q={q_len}, K={k_len}, D={d} "
            "will NOT trigger ThunderKittens. The test would fall back to FlashAttention (which is missing)."
        )

def test_unique_prefill():
    """
    Test 1: Standard Prefill (Unique Path)
    Simulates a standard long prompt where Q and K are the same length.
    """
    print("\n" + "="*60)
    print("TEST 1: Unique Prefill (Self-Attention)")
    print("="*60)
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # Configuration matches TK constraints (L % 192 == 0)
    B, H, D = 2, 16, 128
    S = 768 
    
    verify_tk_usage(S, S, D)

    # Inputs
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)

    print(f"Running Hydragen (TK Path) with Shape [B={B}, S={S}, H={H}, D={D}]...")
    
    # Pass empty shared lists to force execution into the "Unique" path
    out_tk = hydragen_attention_tk(
        q=q, k=k, v=v,
        shared_ks=[], shared_vs=[],
        shared_cu_seq_lens=[], shared_max_seq_lens=[], use_varlens=[],
        seq_lens=None
    )

    # Reference (PyTorch SDPA)
    # Transpose for PyTorch: [B, S, H, D] -> [B, H, S, D]
    q_ref = q.transpose(1, 2)
    k_ref = k.transpose(1, 2)
    v_ref = v.transpose(1, 2)
    
    # Note: Hydragen's Unique path uses causal=True
    out_ref = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=True
    )
    out_ref = out_ref.transpose(1, 2) # Back to [B, S, H, D]

    diff = (out_tk - out_ref).abs()
    print(f"Mean Diff: {diff.mean().item():.6f}")
    print(f"Max Diff:  {diff.max().item():.6f}")
    
    if diff.mean().item() < 0.05:
        print(">> SUCCESS: TK Kernel matches Ground Truth.")
    else:
        print(">> FAILURE: Significant mismatch.")

def test_shared_prefill():
    """
    Test 2: Shared Prefix Prefill
    Simulates calculating attention for the Shared Prefix itself.
    """
    print("\n" + "="*60)
    print("TEST 2: Shared Prefix Logic")
    print("="*60)
    
    device = "cuda"
    dtype = torch.bfloat16
    
    B, H, D = 2, 16, 128
    S_SHARED = 768
    
    verify_tk_usage(S_SHARED, S_SHARED, D)

    # Shared KVs (Batch=1, broadcasted)
    shared_k = torch.randn(1, S_SHARED, H, D, device=device, dtype=dtype)
    shared_v = torch.randn(1, S_SHARED, H, D, device=device, dtype=dtype)
    
    # Q must match Shared length for this test to be valid Self-Attention
    # We construct Q such that when Hydragen batches it, it matches shared_k length
    # Hydragen broadcasts: if shared_k has batch 1, and we have batch B=2,
    # it treats it as 2 sequences sharing the prefix.
    # To test pure shared path, we provide a Q that corresponds to the shared tokens.
    q = torch.randn(B, S_SHARED, H, D, device=device, dtype=dtype)
    
    # Mock Unique KVs (Empty/Zero length to ignore them)
    k_empty = torch.randn(B, 0, H, D, device=device, dtype=dtype)
    v_empty = torch.randn(B, 0, H, D, device=device, dtype=dtype)

    print(f"Running Hydragen (TK Shared Path) with Shared Len={S_SHARED}...")
    
    out_tk = hydragen_attention_tk(
        q=q, k=k_empty, v=v_empty,
        shared_ks=[shared_k],
        shared_vs=[shared_v],
        shared_cu_seq_lens=[None],
        shared_max_seq_lens=[None],
        use_varlens=[False],
        seq_lens=None
    )

    # Reference Logic
    # Hydragen broadcasts the shared_k to match Q's batch size
    # So effectively, we are doing SDPA(q, shared_k_expanded, shared_v_expanded)
    
    shared_k_exp = shared_k.expand(B, -1, -1, -1) # [2, 768, 16, 128]
    shared_v_exp = shared_v.expand(B, -1, -1, -1)
    
    q_ref = q.transpose(1, 2)
    k_ref = shared_k_exp.transpose(1, 2)
    v_ref = shared_v_exp.transpose(1, 2)
    
    # Note: Hydragen's Shared path (prefix) is usually NON-causal
    out_ref = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=False
    )
    out_ref = out_ref.transpose(1, 2)

    diff = (out_tk - out_ref).abs()
    print(f"Mean Diff: {diff.mean().item():.6f}")
    
    if diff.mean().item() < 0.05:
        print(">> SUCCESS: TK Kernel matches Ground Truth.")
    else:
        print(">> FAILURE: Significant mismatch.")

if __name__ == "__main__":
    try:
        test_unique_prefill()
        test_shared_prefill()
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")