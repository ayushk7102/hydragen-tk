import torch
import thunderkittens as tk
from attention_tk import tk_simple_attention
def run_diagnostics():
    device = "cuda"
    torch.manual_seed(42)
    
    # Use a safe sequence length
    B, S, H, D = 1, 192 * 4, 16, 128  # S = 768
    dtype = torch.bfloat16
    
    print(f"DEBUG: Testing with B={B}, S={S}, H={H}, D={D}")

    # --- TEST 1: Zeros/Ones Check ---
    print("\n[TEST 1] Constant Inputs (All 1.0)")
    q = torch.ones(B, S, H, D, device=device, dtype=dtype)
    k = torch.ones(B, S, H, D, device=device, dtype=dtype)
    v = torch.ones(B, S, H, D, device=device, dtype=dtype)

    # Transform for TK
    q_tk = q.transpose(1, 2).contiguous()
    k_tk = k.transpose(1, 2).contiguous()
    v_tk = v.transpose(1, 2).contiguous()

    # Run
    o_tk, _ = tk.mha_forward(q_tk, k_tk, v_tk, False)
    out = o_tk.transpose(1, 2)

    # Check
    print(f"  Output Mean: {out.mean().item():.4f} (Expected ~1.0)")
    print(f"  Output Min:  {out.min().item():.4f}")
    print(f"  Output Max:  {out.max().item():.4f}")
    
    if out.abs().sum() == 0:
        print("  FATAL: Output is all ZEROS. The kernel is not writing to memory.")
        return

    # --- TEST 2: Indexing / Layout ---
    print("\n[TEST 2] Value Pass-Through")
    # Make Attention Matrix uniform (Q, K = 0) -> Softmax is uniform
    # Make V = Index of sequence
    q = torch.zeros(B, S, H, D, device=device, dtype=dtype)
    k = torch.zeros(B, S, H, D, device=device, dtype=dtype)
    
    # V has a distinct value for every row: 1, 2, 3...
    # If Attention is uniform, Output should be Mean(1..S) approx S/2
    v_seq = torch.arange(S, device=device, dtype=torch.float32).view(1, S, 1, 1)
    v = v_seq.expand(B, S, H, D).to(dtype)

    q_tk = q.transpose(1, 2).contiguous()
    k_tk = k.transpose(1, 2).contiguous()
    v_tk = v.transpose(1, 2).contiguous()

    o_tk, _ = tk.mha_forward(q_tk, k_tk, v_tk, False)
    out = o_tk.transpose(1, 2)

    expected_val = (S - 1) / 2.0
    print(f"  Input V Range: [0, {S-1}]")
    print(f"  Output Mean:   {out.mean().item():.2f} (Expected approx {expected_val:.2f})")
    print(f"  Output Slice [0,0,0,:5]: {out[0,0,0,:5].tolist()}")
    
    # --- TEST 3: Random with Tolerance ---
    print("\n[TEST 3] Random Data (BF16)")
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)

    # 1. Run ThunderKittens (Adapter handles transpose internally)
    tk_out, _ = tk_simple_attention(q, k, v, is_causal=False)
    
    # 2. Run PyTorch Reference
    # FIX: Transpose inputs to [B, H, S, D] so PyTorch understands them correctly
    q_ref = q.transpose(1, 2)
    k_ref = k.transpose(1, 2)
    v_ref = v.transpose(1, 2)
    
    torch_out_ref = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=False
    )
    
    # Transpose result back to [B, S, H, D] for comparison
    torch_out = torch_out_ref.transpose(1, 2)
    
    diff = (tk_out - torch_out).abs()
    print(f"  Mean Diff: {diff.mean().item():.4f}")
    print(f"  Max Diff:  {diff.max().item():.4f}")
    
    if diff.mean().item() < 0.05:
        print(">> SUCCESS: Kernel matches PyTorch!")
    else:
        print(">> WARNING: Still mismatching.")

if __name__ == "__main__":
    run_diagnostics()