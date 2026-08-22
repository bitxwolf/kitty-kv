# TECHNICAL SPECIFICATION: KITTY KV CACHE QUANTIZATION
## SYSTEM-ALGORITHM CO-DESIGN & ACCURATE 2-BIT INFERENCE REFERENCE GUIDE

This reference specification details the full algorithm, system architecture, mathematical formulas, and memory layout of the **Kitty** quantization framework. It is formatted to provide an exhaustive source-of-truth for an AI coding environment (such as Antigravity) to implement the entire framework from scratch in PyTorch and OpenAI Triton, targetting local inference hardware like the NVIDIA RTX 5050 Laptop/Desktop GPU.

---

## 1. PARADIGM & SYSTEM ARCHITECTURE

The Kitty framework addresses the KV Cache memory bottleneck during long-context LLM inference (e.g., up to 32K or 128K context lengths) through an **Algorithm-System Co-Design** [1]. Standard 2-bit quantization schemes (like KIVI) introduce substantial accuracy drops on complex reasoning tasks (e.g., GPQA, GSM8K, AIME) due to extreme representation loss in the Key Cache [1, 13, 14]. 

Kitty closes this accuracy gap with near-zero loss compared to FP16 baselines while achieving nearly **8× reduction in KV cache memory footprint** and enabling up to **8× larger batches** and **2.1×–4.1× higher inference throughput** [1, 5].

### 1.1 The Heterogeneous Memory Architecture
Kitty divides the KV cache into four active partitions to balance precision, locality, and compression [7]:
1.  **Attention Sinks ($S = 32$)**: The first $S=32$ tokens of any sequence are permanently retained in full-precision (FP16 or BF16) for both Key and Value caches, as they act as attention anchors receiving high attention weights [7, 16].
2.  **Quantization Buffer / Q-Buffer ($G = 128$)**: A temporary FP16 buffer that stores newly generated key and value states [7]. Quantization is only triggered when the Q-Buffer is full (accumulating exactly $G=128$ tokens) [37, 39]. This design amortizes quantization overhead over $G$ steps, reducing execution overhead to a negligible fraction [39].
3.  **Value Cache Sliding Window ($R = 128$)**: The Value Cache maintains the $R=128$ most recent tokens in FP16 [7, 37]. When the sliding window overflows, the oldest token is evicted from the Local Buffer and pushed to the Q-Buffer [37].
4.  **Quantized Pages (HBM Block Pool)**: When the Q-Buffer fills to size $G$, its tokens are quantized and packed into compact page-centric layouts in HBM [30, 39]. Key pages utilize mixed precision (2-bit baseline with dynamically boosted 4-bit channels), while Value pages utilize uniform 2-bit per-token quantization [7, 30].

---

## 2. MATHEMATICAL SPECIFICATIONS

### 2.1 Key Cache Channel-wise Sensitivity & Precision Boost
The Key Cache has size $(B, H, L, D)$ where $B$ is batch size, $H$ is number of KV heads, $L$ is sequence length, and $D$ is the head dimension [9]. In Grouped-Query Attention (GQA), multiple query heads share the same Key head [23].

#### Observation 1: Channel Magnitude Imbalance
A small subset of Key Cache channels consistently exhibits high activation magnitudes across sequence lengths and layers [21, 24].

#### Observation 2: Channel-wise Quantization Sensitivity
Isolating and quantizing single channels to 2-bit shows that a minor fraction of channels causes high Mean Squared Error (MSE) in the resulting attention scores:
$$\text{Attention Score} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right)$$ [23]
Quantizing sensitive channels to 2-bit severely degrades the attention matrix, while non-sensitive channels can be aggressively quantized with negligible error [25].

#### Mathematical Selection Heuristic
Instead of calculating expensive MSE at runtime, Kitty approximates channel sensitivity via a magnitude-based heuristic calculated over $T$ profiled calibration tokens [26, 27]:
$$s_i = \frac{1}{T} \sum_{t=1}^{T} |x_{i, t}|$$ [27]
Where $x_{i,t}$ represents the activation of channel $i$ at token step $t$.

#### Precision Allocation (Kitty vs. Kitty-Pro)
Let $r$ be the channel boost rate (the fraction of channels promoted to 4-bit precision):
-   **Kitty (Standard)**: $r = 12.5\%$ (e.g., $16$ out of $128$ channels for a head dimension of $128$) [43, 50].
-   **Kitty-Pro**: $r = 25.0\%$ (e.g., $32$ out of $128$ channels) [43, 50].

The top-$K$ channels (where $K = r \times D$) with the highest score $s_i$ are marked as **Boosted (INT4)**, while the remaining $D - K$ channels are stored at baseline **INT2** precision [7, 26, 27].

### 2.2 Quantization & Dequantization Formulas

#### Key Cache Quantization (Per-Channel Grouped)
Quantization parameters (scale $S$ and zero-point $Z$) are calculated per-channel across a group of size $G=128$ tokens [10, 30]. Let $X$ represent a channel's FP16 values along the $G$ tokens:
$$S_c = \frac{\max(X) - \min(X)}{2^b - 1}$$
$$Z_c = \text{round}\left(\frac{-\min(X)}{S_c}\right)$$
$$\hat{X}_c = \text{clip}\left(\text{round}\left(\frac{X}{S_c}\right) + Z_c, 0, 2^b - 1\right)$$
Where:
-   For non-boosted channels: bit-depth $b = 2$, values are clipped to $[0, 3]$.
-   For boosted channels: bit-depth $b = 4$, values are clipped to $[0, 15]$.

#### Value Cache Quantization (Per-Token)
The Value Cache is quantized per-token across the head dimension $D$ to preserve temporal and semantic coherence [10, 30]. Scale $S_t$ and zero-point $Z_t$ are computed dynamically per-token:
$$S_t = \frac{\max(V_t) - \min(V_t)}{3}$$
$$Z_t = \text{round}\left(\frac{-\min(V_t)}{S_t}\right)$$
$$\hat{V}_t = \text{clip}\left(\text{round}\left(\frac{V_t}{S_t}\right) + Z_t, 0, 3\right)$$

---

## 3. PAGE-CENTRIC DENSE-SPARSE MEMORY LAYOUT

To avoid scattered global memory reads, pointer divergence, and uncoalesced memory accesses on GPUs when handling mixed INT2/INT4 channels, Kitty employs a **Dense-Sparse Decomposition Layout** for each Key page of size $(D, G)$ [1, 29, 31]:

```
      Mixed-Precision Logical Key Page (D, G)
┌─────────────────────────────────────────────────┐  ───
│ Channel 0 (Non-boosted):  INT2, INT2, INT2...   │   ▲
├─────────────────────────────────────────────────┤   │
│ Channel 1 (Boosted):      INT4, INT4, INT4...   │   │  D channels
├─────────────────────────────────────────────────┤   │
│ Channel 2 (Non-boosted):  INT2, INT2, INT2...   │   ▼
└─────────────────────────────────────────────────┘  ───
                         │
                         ▼ (Dense-Sparse Decomposition)
                         
      1. Dense Baseline Tensor         2. Sparse Boosted Tensor
         "Tensor_2bits"                   "Tensor_High_2bits"
┌──────────────────────────────┐       ┌──────────────────────────────┐
│ Ch 0: Low 2 bits (Packed)    │       │ Ch 1: High 2 bits (Packed)   │ ── D_boost
├──────────────────────────────┤       └──────────────────────────────┘ channels
│ Ch 1: Low 2 bits (Packed)    │
├──────────────────────────────┤       3. Index Mapping Vector
│ Ch 2: Low 2 bits (Packed)    │          "Boost_IDX_uint8"
└──────────────────────────────┘       ┌──────────────────────────────┐
      Shape: (D, G // 4 bytes)         │ 255 │  0  │ 255 │ ...        │ ── Shape: (D,)
                                       └──────────────────────────────┘
```

1.  **Dense Baseline Pool (`Tensor_2bits`)**:
    -   **Shape**: $(D, G // 4)$ bytes (stored as packed `uint8` elements, where each byte contains four consecutive 2-bit values along the $G$-dimension) [32, 33].
    -   **Content**: Baseline INT2 values for non-boosted channels, and the **lower 2 bits** of the INT4 values for boosted channels [32].
2.  **Sparse Boosted Pool (`Tensor_High_2bits`)**:
    -   **Shape**: $(D_{\text{boost}}, G // 4)$ bytes [32].
    -   **Content**: The **higher 2 bits** of only the $D_{\text{boost}}$ channels promoted to INT4 precision [32].
3.  **Index Pointer Array (`Boost_IDX_uint8`)**:
    -   **Shape**: $(D,)$ bytes [32].
    -   **Content**: Maps each logical head channel index $d \in [0, D-1]$ to its physical offset index $d_{\text{boost}} \in [0, D_{\text{boost}}-1]$ inside the sparse boosted pool [32]. If channel $d$ is not boosted, it is assigned a sentinel value (e.g., $D_{\text{boost}} + 1$, or $255$) [32].

This decomposition guarantees that both baseline and high-bit storage elements are layout-aligned, eliminating scattered, non-contiguous reads from HBM. The GPU loads continuous blocks of 2-bit dense values and selectively fetches the high-bit blocks only when the mapping index resolves to a valid offset [29, 34].

---

## 4. CUDA/TRITON BIT RECONSTITUTION KERNELS

When pages are loaded into SRAM during attention decoding steps, they are unpacked, reconstituted to INT4/INT2, and scaled back to full-precision FP16 on-chip [34].

### 4.1 Bitwise Shift and OR Extraction
The packing format packs four consecutive 2-bit values into one `uint8` byte.
Let `shifts = [0, 2, 4, 6]`. The extraction of elements from a byte `B` is computed as:
$$\text{val} = (B \gg \text{shift}) \;\&\; 0\text{x}3$$ [35]

The combined INT4 representation is reconstituted as:
$$X = X_{\text{low}} \;|\; (X_{\text{high}} \ll 2)$$ [35]

### 4.2 Key Cache Dequantization Kernel (Triton Pseudo-code)

```python
import triton
import triton.language as tl

@triton.jit
def dequantize_key_page_kernel(
    tensor_2bits_ptr,       # Pointer to [D, G // 4] packed low bits
    tensor_high_2bits_ptr,  # Pointer to [D_boost, G // 4] packed high bits
    boost_idx_ptr,          # Pointer to [D] channel mapping index
    scale_ptr,              # Pointer to [D] scales
    zero_ptr,               # Pointer to [D] zero points
    out_fp16_ptr,           # Output buffer in SRAM/HBM [D, G]
    D, G, D_boost,
    stride_2bits_d, stride_2bits_g,
    stride_high_d, stride_high_g,
    stride_out_d, stride_out_g
):
    # Program ID along the channel dimension
    pid_d = tl.program_id(0)
    # Program ID along the token group dimension (each block processes G tokens)
    pid_g_block = tl.program_id(1)

    if pid_d >= D:
        return

    # Load scales and zero points for this channel
    scale = tl.load(scale_ptr + pid_d)
    zero_point = tl.load(zero_ptr + pid_d)

    # Load the boost mapping index for this channel
    b_idx = tl.load(boost_idx_ptr + pid_d)
    is_boosted = b_idx < D_boost

    # Base pointers for the G dimension (packed bytes)
    # G // 4 packed bytes. Process them in parallel
    for i in range(0, G // 4):
        g_byte_idx = i
        
        # Load dense low bits
        low_addr = tensor_2bits_ptr + pid_d * stride_2bits_d + g_byte_idx * stride_2bits_g
        low_packed = tl.load(low_addr)

        # Conditionally load sparse high bits
        if is_boosted:
            high_addr = tensor_high_2bits_ptr + b_idx * stride_high_d + g_byte_idx * stride_high_g
            high_packed = tl.load(high_addr)
        else:
            high_packed = 0

        # Unpack 4 tokens from the loaded bytes
        for t in range(4):
            token_idx = g_byte_idx * 4 + t
            shift = t * 2
            
            x_low = (low_packed >> shift) & 0x03
            
            if is_boosted:
                x_high = (high_packed >> shift) & 0x03
            else:
                x_high = 0

            # Reconstitute 2-bit/4-bit integer
            x_quant = x_low | (x_high << 2)

            # Dequantize to FP16
            x_fp16 = (x_quant.to(tl.float32) - zero_point) * scale

            # Store to output
            out_addr = out_fp16_ptr + pid_d * stride_out_d + token_idx * stride_out_g
            tl.store(out_addr, x_fp16.to(tl.float16))
```

---

## 5. RUNTIME COORDINATION LAYER

```
            [ New K, V Token Generated (FP16) ]
                           │
                           ▼
             Is the Sink Buffer Filled?
             ├── YES ───────────────────────────┐
             │                                  │
             ▼                                  ▼
    [ Save K, V to Q-Buffer ]           [ Save K, V to Sink ]
             │                                  │
             ▼                                  ▼
    Is Q-Buffer Full (Size == G)?          (Decoding Attention step)
    ├── YES ───────────────────┐
    │                          │
    ▼                          ▼
[ Trigger Quantization ]   [ Forward to QK/SV Kernels ]
    │
    ├─► Quantize Key via Per-Channel Dynamic Boost
    ├─► Quantize Value via Per-Token INT2 Window
    └─► Store to Quantized Page Pools (HBM)
```

The runtime coordination layer schedules memory execution without thread block divergence across the local RTX 5050 GPU:

1.  **The Q-Buffer Insertion**:
    When a new token $t$ is decoded, its key $K_t$ and value $V_t$ are generated in FP16 [37]. If the static FP16 Sink Buffer is not full, they are stored directly in FP16 [37]. Once the Sink Buffer is full, $K_t$ is stored in the Key Q-Buffer [37]. For Value cache, $V_t$ is pushed to the FP16 Local Buffer [37]. The oldest token in the Local Buffer is evicted and written into the Value Q-Buffer [37].
2.  **Page Quantization Event**:
    As soon as $G=128$ tokens accumulate in the Q-Buffer, a background quantization kernel is dispatched:
    -   **Key Cache**: Calculates per-channel scales, quantizes elements, performs Dense-Sparse Decomposition, and writes the packed `Tensor_2bits` and `Tensor_High_2bits` to the page block pool [31, 32].
    -   **Value Cache**: Performs per-token uniform 2-bit quantization and writes to the Value page block pool [30].
    The Q-Buffer is then cleared (size reset to 0) [37].
3.  **Attention Execution (Triton SRAM-fused Attention)**:
    Rather than writing full-precision intermediate key-value vectors back to GPU memory, Kitty performs dequantization and attention directly inside GPU SRAM:
    -   **`qk_kernel`**: Loads the FP16 Query vector, loads and dequantizes Key pages on-chip in SRAM, and performs the dot-product $Q K^T / \sqrt{d}$ to output the logits [38].
    -   **Softmax**: Computes softmax over the attention logits in FP32 [38].
    -   **`sv_kernel`**: Loads the softmax scores, loads and dequantizes Value pages on-chip, and multiplies them to output the final attention context vector [38].

---

## 6. BENCHMARK & HARDWARE VALIDATION SUITE

To validate the implementation on the local NVIDIA RTX 5050 Laptop/Desktop GPU, the codebase must include a profiling suite that uses PyTorch Profiler or Nsight Compute wrappers to verify three key metrics:

1.  **Memory Compression Factor**: Verify that peak GPU memory consumption of the KV Cache is reduced by $\sim 8\times$ compared to FP16, confirming that the dual-tensor packing works correctly without memory leaks or storage gaps.
2.  **Global Memory Bandwidth**: Confirm that loading pages via the dual-tensor Dense-Sparse layout yields coalesced global memory reads. Uncoalesced or scattered reads will trigger profiling flags in Nsight Compute; the dual-tensor layout must keep L2 cache hit rates high and layout-divergence stalls at zero.
3.  **Latency Parity**: Demonstrate that dequantization in SRAM adds minimal overhead ($\le 5\%$), and that under long sequences, the reduced data-transfer time between HBM and SRAM yields overall latency speedups compared to high-precision pipelines.

---

### CITATIONS
*   **[1]** Section 1 (Abstract): Core co-design philosophy, memory reduction, and throughput gains.
*   **[5]** Section 1 (Contributions): 8x larger batch size capability and throughput achievements.
*   **[7]** Section 3 (Figure 1): Heterogeneous memory layout (Sink, Q-Buffer, Local, Quantized partitions).
*   **[9]** Section 2.1: Key/Value matrix dimensions and head structure.
*   **[10]** Section 2.1: Per-channel and per-token quantization conventions.
*   **[13]** Section 2.2: Accuracy drops of traditional low-bit (2-bit) quantizations.
*   **[14]** Section 2.2 (Table 1): Empirical evaluation details on Qwen3 and LLaMA3 models.
*   **[16]** Section 3.1: Mathematical role and importance of initial Attention Sinks.
*   **[21]** Section 3.2 (Observation 1): Channel magnitude patterns in LLM Key caches.
*   **[23]** Section 3.2 (Observation 2): GQA query-sharing and attention scoring sensitivity.
*   **[24]** Section 3.2 (Figure 2): Mean Squared Error patterns under low-bit perturbation.
*   **[25]** Section 3.2: Vulnerability of sensitive channels vs. resilience of non-sensitive channels.
*   **[26]** Section 3.2: Heuristic selection design to bypass expensive runtime MSE calculations.
*   **[27]** Section 3.2 (Formula 2): Average absolute magnitude heuristic score calculation.
*   **[29]** Section 4.1: Page-centric structure and memory layout requirements.
*   **[30]** Section 4.1: Value cache paging and layout design.
*   **[31]** Section 4.1: Dense-Sparse decomposition of mixed-precision Key pages.
*   **[32]** Section 4.1: Packed bit layouts (`Tensor_2bits`, `Tensor_High_2bits`, and `Boost_IDX_uint8`).
*   **[33]** Section 4.2: SRAM parallel reconstruction principles.
*   **[34]** Section 4.2: Address-coalesced parallel page dequantization.
*   **[35]** Section 4.2 (Algorithm 1): Reconstitution bit-shifts, OR-combinations, and scale multiplications.
*   **[37]** Section 4.3 (Step 1): Buffer insertion pathways and sliding-window Value cache local buffers.
*   **[38]** Section 4.3 (Step 2): Triton compositional attention kernels (`qk_kernel`, `sv_kernel`).
*   **[39]** Section 4.3 (Step 3): Amortized quantization trigger conditions for Q-Buffer.
*   **[43]** Section 5.2: Percentage allocations of boosted channels (12.5% Kitty, 25% Kitty-Pro).
*   **[50]** Section 5.2 (Table 3 notes): Definition of Kitty and Kitty-Pro boost ratios.
