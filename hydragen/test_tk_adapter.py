import torch
import thunderkittens as tk
from attention_tk import tk_simple_attention

def run_test():
    torch.manual_seed(42)
    device = "cuda"
    
    # CONSTRAINT: Seq length must be multiple of 192 for this specific kernel
    B, S, H, D = 2, 2304, 16, 128 
    
    print(f"Testing with B={B}, S={S}, H={H}, D={D} (BF16 Mode)")
    
    # Use BFLOAT16 for inputs to match Kernel precision exactly
    q = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)

    # 1. Run ThunderKittens
    tk_out, _ = tk_simple_attention(q, k, v)
    
    # 2. Run PyTorch Reference (in BF16)
    q_ref = q.transpose(1, 2)
    k_ref = k.transpose(1, 2)
    v_ref = v.transpose(1, 2)

    ref_out = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, is_causal=False)
    # Transpose back to [B, S, H, D] for comparison
    torch_out = ref_out.transpose(1, 2)
    
    # 3. Compare
    diff = (tk_out - torch_out).abs().mean().item()
    max_diff = (tk_out - torch_out).abs().max().item()
    
    print("-" * 40)
    print(f"Mean Difference: {diff:.6f}")
    print(f"Max Difference:  {max_diff:.6f}")
    
    if diff < 0.05:
        print(">> SUCCESS: Values match closely!")
    else:
        print(">> WARNING: Mismatch detected.")

if __name__ == "__main__":
    run_test()