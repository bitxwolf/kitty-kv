# ANTIGRAVITY MASTER PROMPT: IMPLEMENTING THE KITTY KV CACHE QUANTIZATION FRAMEWORK

You are a World-Class AI Kernel Engineer and Systems Architect. Your objective is to write a complete, production-ready, modular PyTorch and OpenAI Triton implementation of the **"Kitty" 2-bit KV Cache Quantization** framework from scratch. 

The implementation must run locally on an NVIDIA RTX 5050 GPU (Laptop/Desktop GPU with high Tensor Core performance but highly constrained VRAM and memory bandwidth). It must be fully compatible with PyTorch and easily integrable into a vLLM-style PagedAttention pipeline.

---

## SECTION 1: DESIGN OBJECTIVES & CORE CONSTRAINTS

1. **Maximum KV Cache Compression**: Key Cache is compressed to 2-bit baseline (per-channel) with a dynamic channel-wise boost to 4-bit (INT4) for the most sensitive channels. Value Cache is compressed to 2-bit per-token.
2. **Zero Scattered HBM Reads**: To prevent performance degradation on RTX 5050 hardware, you must use **Dense-Sparse Decomposition Layouts**. Avoid non-coalesced or scattered global memory reads during key and value cache lookups.
3. **On-Chip Bit Reconstitution (SRAM)**: The dequantization and reconstruction of 4-bit/2-bit mixed-precision data must occur strictly on-chip inside Triton SRAM during the attention computation. No intermediate full-precision (FP16/BF16) quantized matrices should be written back to global memory (HBM).
4. **Attention Parity**: Maintain near-lossless generation quality compared to the FP16/BF16 baseline by integrating an **Attention Sinks** buffer and a sliding **Value Cache Local Buffer**.

---

## SECTION 2: MATHEMATICAL & ARCHITECTURAL SPECIFICATION

### 1. Key Cache Dimensions & Activation Vectors
Let the Key activation vector for a single attention head be denoted as:
$$\mathbf{K} \in \mathbb{R}^{B \times H \times L \times D}$$
where:
*   $B$ = Batch size
*   $H$ = Number of KV attention heads
*   $L$ = Sequence length
*   $D$ = Head dimension (typically 128)

### 2. Channel-wise Importance Metric
The offline or warmup sensitivity is determined by calculating the channel importance score $s_i$ over a sequence of $T$ profiling tokens:
$$s_i = \frac{1}{T} \sum_{t=1}^{T} |x_{i, t}| \quad \text{for } i \in [1, D]$$
where $x_{i, t}$ is the full-precision Key projection activation of head channel $i$ at token step $t$.

### 3. Dynamic Channel-wise Precision Boost (Top-K Masking)
Let $r$ be the dynamic channel boost rate (typically $12.5\%$ for Kitty, or $25.0\%$ for Kitty-Pro). 
*   Define the number of boosted channels: $D_{\text{boost}} = \text{round}(r \times D)$.
*   Sort the channels in descending order of their importance score $s_i$.
*   Generate a static head-specific boolean mask $\mathbf{M} \in \{0, 1\}^D$ where the top $D_{\text{boost}}$ channels are marked as `True` (4-bit), and the rest are `False` (2-bit baseline).
*   Generate an index lookup mapping array `Boost_IDX_uint8` of shape $(D,)$ where:
    $$\text{Boost\_IDX}[i] = \begin{cases} 
    \text{rank of channel } i \text{ in the boosted subset} & \text{if } \mathbf{M}[i] = \text{True} \\
    D_{\text{boost}} & \text{if } \mathbf{M}[i] = \text{False (sentinel value)}
    \end{cases}$$

### 4. Mixed-Precision Layout via Dense-Sparse Decomposition
To ensure memory coalescing, a physical page of size $G = 128$ tokens is stored using three distinct, flat, contiguous tensors:
1.  **Baseline 2-bit Tensor (`Tensor_2bits`)**: Shape $(D, G / 4)$ packed as `torch.uint8` (4 channels of 2-bit per byte).
    *   For non-boosted channels: Contains the standard 2-bit quantized value.
    *   For boosted channels: Contains the lower 2 bits of the 4-bit quantized value.
2.  **Sparse High 2-bit Tensor (`Tensor_High_2bits`)**: Shape $(D_{\text{boost}}, G / 4)$ packed as `torch.uint8` (4 channels of 2-bit per byte).
    *   Contains strictly the higher 2 bits of the 4-bit quantized values for the boosted channels.
3.  **Metadata Scale/Zero-point Tensor**: Shape $(2, D)$ stored as FP16/BF16, representing per-channel scale and zero-point values.

### 5. Bitwise In-SRAM Dequantization Math
In Triton SRAM, the final combined quantized value $X_{\text{quant}} \in [0, 15]$ for a boosted channel is reconstructed by reading the lower 2 bits from `Tensor_2bits` ($X_{\text{low}}$) and the higher 2 bits from `Tensor_High_2bits` ($X_{\text{high}}$):
$$X_{\text{quant}} = X_{\text{low}} \mid (X_{\text{high}} \ll 2)$$
Dequantize back to full precision using:
$$X_{\text{fp16}} = (X_{\text{quant}} - \text{ZeroPoint}) \times \text{Scale}$$

---

## SECTION 3: STEP-BY-STEP IMPLEMENTATION ROADMAP

### PHASE 1: THE OFFLINE SENSITIVITY RANKING ENGINE
Write a PyTorch profiling class `KittySensitivityProfiler` that:
*   Hooks into the Key Projection linear layer of target transformer layers (e.g., LLaMA or Qwen).
*   Feeds a calibration dataset (e.g., Wikitext-2 or GSM8K) through the model to record the absolute magnitudes of activations.
*   Calculates the average score $s_i$ for each head and channel.
*   Constructs the boolean mask tensor of shape `(num_layers, num_kv_heads, D)` and the corresponding `Boost_IDX_uint8` mapping. Saves these configurations to disk.

### PHASE 2: DUAL-TENSOR LAYOUT MANAGEMENT
Implement `PageCentricKVLayoutManager` to manage pre-allocated global memory pools:
*   Initialize two contiguous physical memory block pools:
    *   `key_baseline_pool`: Shape `(max_pages, num_kv_heads, D, G // 4)` of type `torch.uint8`.
    *   `key_boost_pool`: Shape `(max_pages, num_kv_heads, D_boost, G // 4)` of type `torch.uint8`.
    *   `key_metadata_pool`: Shape `(max_pages, num_kv_heads, 2, D)` of type `torch.float16`.
*   Ensure value caches are handled in a separate 2-bit pool with scales per token to prevent overhead.
*   Write indexing lookup wrappers that retrieve physical offset addresses from logical page indices.

### PHASE 3: CUSTOM TRITON KERNELS
Create fully optimized OpenAI Triton JIT kernels:
1.  **`quantize_key_page_kernel`**:
    *   Read FP16/BF16 key matrices for a sequence block.
    *   Calculate per-channel min/max to compute scale and zero-points.
    *   Perform quantization. Pack the lower 2 bits.
    *   Query `Boost_IDX_uint8`; if the channel is boosted, save the higher 2 bits into the coalesced high-boost pool contiguous rows.
2.  **`dequantize_key_page_kernel`**:
    *   Load low-bit values from `Tensor_2bits` and extract 2 bits via masking/shifting.
    *   Lookup the physical offset via `Boost_IDX_uint8`.
    *   If below $D_{\text{boost}}$, load the corresponding high 2 bits, shift them left by 2 bits, and bitwise OR them with the low-bit data.
    *   Multiply with scales and add zero-points on-chip inside SRAM.
3.  **`compositional_paged_attention_kernel`**:
    *   A fused query-key-value (QK/SV) kernel that loads query vector $Q$.
    *   Iterates through pages in the logical sequence map.
    *   Loads and dequantizes Key values in-SRAM on the fly.
    *   Computes attention scores, performs softmax, and loads and dequantizes Value cache blocks on the fly to yield final attention output.

### PHASE 4: INFERENCE ENGINE HOOK & RUNTIME PIPELINE
Write the coordination module `KittyInferencePipeline`:
*   **Attention Sinks Buffer**: Reserve the first $S = 32$ tokens of the sequence in full FP16. Do not quantize these.
*   **Q-Buffer**: Retain the latest $G = 128$ tokens in FP16. Once the Q-Buffer reaches capacity ($G$), trigger the quantization kernel to compress the oldest page and append it to the `PageCentricKVLayoutManager`.
*   **Value Cache Local Window**: Keep the latest $R = 128$ tokens for the Value Cache in FP16, only quantizing them as they drift out of the window.
*   **Pipeline Wrapper**: Implement `KittyPagedAttention(torch.nn.Module)` which wraps the above runtime buffer logic and overrides the standard attention forwarding function.

---

## SECTION 4: HARDWARE PROFILING & VALIDATION HOOKS

Provide a benchmark validation script `benchmark_kitty_hardware.py`:
*   Integrate PyTorch's `torch.profiler.profile` with GPU activity recording.
*   Measure Kernel latency, global memory load efficiency (bandwidth saturation), and layout stall metrics on your local RTX 5050.
*   Output an automated comparison sheet mapping:
    *   Baseline FP16 vs. Kitty 2-bit / 4-bit Boost.
    *   VRAM footprint reduction factor (expecting up to ~8x reduction for long contexts).
    *   Prefetch and decoding throughput (tokens per second).

---

Write this entire codebase in highly clean, documented, and structurally complete Python, PyTorch, and Triton code. Ensure all CUDA synchronization points are correctly specified, memory is pre-allocated to avoid runtime overheads, and the code contains zero placeholder functions. Generate the entire implementation now!
