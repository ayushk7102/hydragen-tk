import torch
import thunderkittens as tk

def tk_simple_attention(q, k, v, is_causal=False):
    """
    Adapter to make ThunderKittens H100 kernel compatible with Hydragen.
    
    Hydragen expects inputs: [Batch, Seq, Heads, Dim]
    TK H100 expects inputs:  [Batch, Heads, Seq, Dim] (and bfloat16)
    
    Returns:
        out: [Batch, Seq, Heads, Dim]
        lse: [Batch, Seq, Heads]
    """
    # 1. Capture original dtype to cast output back later (Hydragen uses float16)
    orig_dtype = q.dtype

    # 2. Prepare inputs for ThunderKittens
    # - Cast to bfloat16 (Required by h100.cu)
    # - Transpose from (B, N, H, D) -> (B, H, N, D)
    # - Ensure contiguous memory (Crucial for C++ kernels)
    q_tk = q.to(torch.bfloat16).transpose(1, 2).contiguous()
    k_tk = k.to(torch.bfloat16).transpose(1, 2).contiguous()
    v_tk = v.to(torch.bfloat16).transpose(1, 2).contiguous()

    # 3. Call the kernel
    # causal=False because Hydragen handles causality via prefix/suffix decomposition
    # Returns: o (B, H, N, D), l_vec (B, H, N, 1)
    o_tk, l_vec_tk = tk.mha_forward(q_tk, k_tk, v_tk, is_causal)

    # 4. Process Output
    # Transpose back: (B, H, N, D) -> (B, N, H, D)
    out = o_tk.transpose(1, 2).to(orig_dtype)

    # 5. Process LSE (Log-Sum-Exp)
    # TK returns (B, H, N, 1) -> We need (B, N, H)
    # Squeeze the last dim -> (B, H, N) -> Transpose -> (B, N, H)
    lse = l_vec_tk.squeeze(-1).transpose(1, 2).contiguous()

    return out, lse