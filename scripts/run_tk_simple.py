import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from hydragen.llama import HydragenLlamaForCausalLM

# Map string dtypes to torch dtypes
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

def main():
    parser = argparse.ArgumentParser(description="Run Hydragen with ThunderKittens optimization")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model name (Must have HeadDim 64 or 128)")
    parser.add_argument("--prompt", type=str, default="Once upon a time in a land of tensors,", help="Input prompt")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading model: {args.model}...")
    
    # 1. Load Model & Tokenizer
    torch_dtype = DTYPE_MAP[args.dtype]
    model = HydragenLlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Prepare Inputs
    print(f"Tokenizing prompt...")
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)
    
    # 3. ALIGNMENT LOGIC (Crucial for TK H100)
    # The kernel requires seq_len % 192 == 0. We pad the input to force this.
    seq_len = input_ids.shape[1]
    BLOCK_SIZE = 192
    
    if seq_len < BLOCK_SIZE or seq_len % BLOCK_SIZE != 0:
        # Calculate next multiple of 192
        target_len = ((seq_len // BLOCK_SIZE) + 1) * BLOCK_SIZE
        pad_amount = target_len - seq_len
        
        print(f"Original Length: {seq_len}")
        print(f"Padding with {pad_amount} tokens to reach {target_len} (Alignment for TK Kernel)...")
        
        # Pad left with pad_token (or space) so the meaningful text is at the end
        input_ids = F.pad(input_ids, (pad_amount, 0), value=tokenizer.pad_token_id)
    
    aligned_seq_len = input_ids.shape[1]
    
    # 4. Setup Hydragen Caches
    # We treat the input_ids as a "Shared Prefix" to trigger the prefill attention optimization
    model.setup_caches(
        max_unique_batch_size=1,            # 1 sequence generated
        max_unique_seq_length=16,           # Generate 16 new tokens
        max_shared_batch_sizes=[1],         # Batch size of the prompt
        max_shared_seq_lengths=[aligned_seq_len] # Length of the prompt
    )
    
    # 5. Generate
    print("\nStarting Generation (Watch for TK Print Statement below)...")
    print("-" * 50)
    
    output_ids = model.generate(
        input_ids=[input_ids],  # Passed as a shared prefix list
        seq_lens=[torch.tensor([aligned_seq_len], device=args.device)],
        num_return_sequences=1,
        max_new_tokens=16,
        temperature=0.7,
    )
    
    print("-" * 50)
    print("Generation Finished.")
    print("\nOutput Text:")
    # Decode only the generated part (or whole thing)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()