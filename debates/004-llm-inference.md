# Debate 004 — What is vLLM, the KV cache, and attention?

**Status:** open — draft 0.1
**Opened:** 2026-10-23
**Editor:** Divyansh Tripathi
**Written by:** AI assistant
**Framing as asked:** "What is vLLM?" and "What is KV cache and attention?"
**Note:** this chapter merges the two original questions into one, because they are
one story. (It replaces the earlier separate vLLM and KV-cache chapters.)

---

## 0. Framing: one machine, three objects

Asked separately, these three look like three topics. They are one chain:

- **Attention** is the operation — a learned retrieval.
- The **KV cache** is what autoregressive generation forces attention to keep.
- **vLLM** is the system that manages that cache so the machine can actually be fed.

Each one creates the next. The operation creates the cache; the cache becomes the
bottleneck; the bottleneck demands a system. So the honest order is: attention → cache
→ system, and the three only make sense read together.

---

## 1. Attention, minimally

For each token, a self-attention layer projects its hidden state into three vectors: a
**query** `q = W_q · x`, a **key** `k = W_k · x`, and a **value** `v = W_v · x`. Then,
for each query, it compares against every key and takes a weighted average of the values:

    a_ij = softmax( q_i · k_j / sqrt(d) )
    o_i  = sum_j  a_ij · v_j

That is the whole operation. Read plainly, it is a **retrieval**: the query is a search
term, the keys are an index, and the output is a soft, weighted lookup into the values.
Every token asks "which earlier tokens are relevant to me?" and pulls a blend of them.

Two facts about its cost drive everything that follows:

- **Compute is quadratic** in sequence length — the score matrix is `N × N`.
- **It has two phases with opposite bottlenecks:**
  - **Prefill** (the prompt): all tokens known at once → a matrix–matrix product →
    **compute-bound**.
  - **Decode** (generation): one token at a time → a matrix–*vector* product →
    **memory-bound**.

That asymmetry is the hinge of the chapter.

---

> **Tangent —** Attention was invented for translation. Bahdanau et al. introduced it in 2014 to let a decoder <em>look back</em> at the encoder's words — and the 2017 paper that removed the recurrence entirely was titled, with some swagger, <em>Attention Is All You Need</em>.

## 2. Why the cache exists

Language models generate **autoregressively**: token `t` depends on all tokens before it.
To produce token `t+1`, the model must attend over the keys and values of tokens `1 … t`.
Without a cache, every step would recompute the keys and values of the *entire* prefix —
the same vectors, over and over. Across a generation of `n` tokens that is `O(n²)`
redundant work. The **KV cache** fixes it: store the key and value vectors as they are
produced, so each step computes only the *new* token's `k` and `v` and attends over the
cached ones.

Two properties deserve emphasis, because they are commonly misremembered:

- **The cache changes nothing about the output.** It is memoization, not approximation —
  generation with and without it produces *identical* tokens. It buys speed, never
  accuracy.
- **It moves the cost from compute to memory.** With the cache, each decode step is
  `O(n)` in attention (it must *read* the whole cache) and `O(1)` in weight matrix
  products. The redundant compute is gone; what remains is a large, growing tensor read
  on every step.

---

> **Tangent —** The KV cache changes nothing. Greedy generation with and without it produces <em>bit-identical</em> tokens. It is pure memoisation — the same answer, computed less wastefully.

## 3. The size — and why it is the bottleneck

The cache's size is simple to write down. For a model with `L` layers, `h` key/value
heads, head dimension `d_h`, context length `S`, in 16-bit:

    KV bytes  =  2 (keys and values) × 2 (bytes) × L × h × d_h × S  =  4 · L · h · d_h · S

It grows **linearly in context length and batch size** — and it is not a rounding error.
For **LLaMA-3-70B** (`L=80, h=64, d_h=128`):

| Context length | KV cache (MHA) | With GQA (8 KV heads) |
|---|---|---|
| 4,096 tokens | ≈ 10.7 GB | ≈ 1.3 GB |
| 32,768 tokens | ≈ 85.9 GB | ≈ 10.7 GB |

At 32k context the MHA cache alone exceeds the 80 GB of an H100. At scale it dwarfs the
weights: one analysis puts an 8B model at batch 128 and 8,192 tokens of context at
**280 GB** of MHA cache — about 30× the model's own weights (69 GB with GQA).

Now put that against the roofline. A kernel's **arithmetic intensity** — FLOPs per byte
moved — decides whether it is compute- or memory-bound. On an H100 the *ridge point* is
about **295 FLOP/byte**. Prefill sits well above it. Decode, at small batch, sits at
roughly **1 FLOP/byte** — two to three orders of magnitude below the ridge. The tensor
cores idle; measured GPU utilisation during decode can fall to around **7%**.

**The cache is not "large in the abstract."** Decoding must re-read it every step while
doing almost no math — so the cache's size, not the model's, becomes the binding
constraint on batch size, throughput and context length.

---

> **Tangent —** For a 70B model at 32k context, the KV cache alone is about 85.9 GB — larger than the entire 80 GB memory of an H100. The cache is bigger than the machine that has to hold it.

## 4. vLLM: the system that manages it

vLLM is **not** a model, an architecture, or a training method. It is an **inference
engine** — a serving system. Nothing about it changes what the model computes; it
changes how fast and how cheaply the model can be *run*.

**The problem.** Serving is memory-bound, and the memory is dominated by the KV cache.
To keep a request's cache *contiguous*, prior systems pre-allocated a chunk sized to the
request's **maximum** length. Three kinds of waste followed — slots reserved for future
tokens, internal fragmentation, and external fragmentation. The measured result: in
prior systems **only 20.4%–38.2% of KV-cache memory was actually used** to store real
token states.

**The trick — PagedAttention.** Borrow OS virtual-memory paging:

- split a request's KV cache into **fixed-size blocks**, each holding the keys and values
  for a fixed number of tokens;
- the blocks **need not be contiguous** in physical memory;
- a per-request **block table** maps logical blocks to physical ones — pages, tokens as
  bytes, requests as processes.

Consequences:

- **Near-zero waste.** On-demand, uniform-size blocks kill external fragmentation and
  confine internal waste to the last, partly-filled block — in practice **under 4%**,
  against the 60–80% of the naive scheme.
- **Sharing becomes possible.** Sequences sharing a prefix can point at the *same*
  physical blocks. Parallel sampling saved 6–10% of memory; beam search, **38–55%**, and
  up to 66% on real chat data.

On top sits **continuous batching** (iteration-level scheduling): finished requests are
dropped and new ones admitted **every decoding iteration** — no padding, no
head-of-line blocking.

**The result:** **2–4× higher throughput than FasterTransformer and Orca at the same
latency** (up to 22× the request rate of FasterTransformer), widening for longer
sequences and larger models — with model outputs bit-for-bit unchanged. This is pure
systems work.

---

> **Tangent —** PagedAttention is not a new idea so much as a borrowed one. It is operating-system virtual memory, transplanted: pages become blocks, bytes become tokens, processes become requests. The trick is roughly sixty years old — it had simply never been applied to a KV cache.

## 5. The debates

**D1. Architecture — share the heads.** The most fundamental lever: make fewer keys and
values to begin with. **MHA** (standard) has one key/value head per query head. **MQA**
(Shazeer, 2019) uses a *single* shared key/value head — the cache shrinks by a factor of
`h`, but quality suffers, and in distributed inference the single head must be replicated
on every device, eroding the saving. **GQA** (Ainslie et al., 2023) is the compromise —
queries are grouped, each group sharing one key/value head; it recovers most of the
quality and is now the default in Llama 2 70B / Llama 3, Mistral and Qwen. **MLA**
(DeepSeek) compresses the hidden state into a low-rank latent and caches *that*. The
trade-off is explicit: **quality vs. KV budget vs. parallelizability.**

**D2. Management — page it.** PagedAttention does not shrink the cache; it stops
*wasting* it. But is paging *necessary*? **vAttention** (2024) argues you can manage the
cache dynamically using CUDA's own virtual-memory APIs, without a custom block
abstraction — which would make PagedAttention less "a new algorithm" than "the right fix
for a limitation the existing allocators had."

**D3. Precision — quantize it.** Store the cache in FP8 or INT4 and you halve or quarter
the bytes; FP8 gives a near-2× decode win in the memory-bound regime. But the cautionary
case is real: PyTorch found an INT4 GQA cache read **4× less data for no latency
improvement at all**, because the kernel was *less bandwidth-efficient* at INT4.
**Fewer bytes ≠ faster.**

**D4. Algorithm — don't materialize it (FlashAttention).** A widespread confusion lives
here: **FlashAttention does *not* reduce the KV cache.** It is *exact* attention that
avoids ever writing the `N × N` score matrix to slow HBM, by tiling the computation into
fast on-chip SRAM. On a GPT-2 attention block, HBM reads/writes fell from **40.3 GB to
4.4 GB** and runtime from **41.7 ms to 7.3 ms**. It slashes memory *traffic* and peak
memory — the cache itself is unchanged.

**D5. Approximation — the road not taken.** Sparse and low-rank attention reduce the
*FLOPs* to near-linear in sequence length, yet many "do not display wall-clock speedup
against standard attention, and have not gained wide adoption" — because they optimised
FLOPs and ignored memory I/O. FlashAttention's lesson was the inverse: **FLOPs are not
time; bytes moved are.**

**D6. The landscape.** vLLM is one engine among several: **TensorRT-LLM** (NVIDIA; often
fastest raw latency, less flexible), **TGI** (Hugging Face), **SGLang** (whose
RadixAttention attacks *prefix* reuse — a different axis), **LMDeploy**. There is no
universal winner; throughput, latency, hardware and features trade off per workload.

**D7. Is this research?** The vLLM/PagedAttention paper appeared at **SOSP**, an
operating-systems venue — not an ML conference. That is the debate in miniature: as
architecture matures, the bottleneck moves to *systems*, and the most impactful work
stops looking like ML.

A systems-level trend sits above all of these: **prefill/decode disaggregation** —
running the compute-bound prefill and the memory-bound decode on *different* hardware,
since they stress opposite resources.

---

> **Tangent —** GQA was designed to be <em>retrofitted</em>. You take an existing multi-head model, mean-pool its key and value heads into groups, and continue training for just 5% of the original compute. Most of the world's fast models were converted, not retrained.

## 6. What the evidence, read together, suggests

1. **The three objects are one chain.** Attention forces the cache; the cache is the
   bottleneck; vLLM is one answer to the bottleneck.
2. **The cache is exact.** It never changes the output — only the speed — and it moves
   cost from compute to memory.
3. **Decode is memory-bandwidth-bound**, and arithmetic intensity is the reason. This is
   why *batching* is the primary lever and why the cache is worth shrinking.
4. **The fixes live on different axes** — architecture (share heads), management (page),
   precision (quantize), algorithm (tile) — and they **compose**. GQA + PagedAttention +
   FP8 + FlashAttention is a coherent stack, not four rival ideas.
5. **FLOPs are not wall-clock.** The approximate-attention story and the INT4 story are
   the same cautionary tale from opposite directions.
6. **vLLM is a memory manager and a scheduler**, not a model. Keeping that boundary
   clear is most of the answer to "what is vLLM."

---

> **Tangent —** Decode runs at roughly 1 FLOP per byte on a chip whose ridge point is 295. The tensor cores sit idle for the overwhelming majority of every generated token — the GPU is waiting on memory, not thinking. And it is getting worse: GPUs have gained about 3× more FLOPs every two years, but only about 1.6× more memory bandwidth.

## 7. Open forks — your call

**F1. Depth of the attention section.** Keep the minimal "retrieval" treatment, or
develop it (multiple heads, positional encoding, causal masking, why `sqrt(d)`)?

**F2. Which axis to go deep on?** Architecture (MHA→GQA→MLA), management (paging),
precision (quantization), or algorithm (FlashAttention)? *They are separable; better to
do one properly than four shallowly.*

**F3. Do we run an experiment?** A small script computing KV-cache size and arithmetic
intensity across models, batches and contexts — and plotting the roofline. *Cheap, and it
makes the chapter quantitative.*

**F4. The paging debate.** Chase vAttention's "you don't need paging" claim?

**F5. The landscape.** Keep vLLM alone, or compare the serving engines as a family?

**F6. Deliverable shape.** Same as before?

**F7. Verdict policy.** Same — does a debate ever get marked *settled*?

---

## Cross-debate notes

- **Debate 001 (attention vs LSTM)** argued about the *architecture*; this chapter is
  about the *cost of running* it. The first-order variable moved from model choice to
  memory management — the same theme, one level down.
- **Debate 003 (CNN for time series)** found that a component's power is a *prior*, not a
  capability. Here the "component" is the allocator: paging's power is not extra compute,
  it is *not wasting* memory.

---

## Further down the rabbit hole

Everything below is real, and grouped so you can go as deep as you want.

**Foundations of attention**
- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762.
- Bahdanau, Cho, Bengio. *Neural Machine Translation by Jointly Learning to Align and
  Translate.* ICLR 2015. arXiv:1409.0473. (Attention before Transformers.)
- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness.* NeurIPS 2022. arXiv:2205.14135.
- Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work
  Partitioning.* 2023. arXiv:2307.08691.
- Shah et al. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low
  Precision.* 2024. arXiv:2407.08608.
- Milakov & Gimelshein. *Online normalizer calculation for softmax.* 2018.
  arXiv:1805.02867. (The trick that makes tiling exact.)

**The KV cache and the memory wall**
- Pope et al. *Efficiently Scaling Transformer Inference.* MLSys 2023. arXiv:2211.05102.
- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model.*
  CACM 2009. (Arithmetic intensity and the ridge point.)
- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019.
  arXiv:1911.02150. (MQA.)
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from
  Multi-Head Checkpoints.* EMNLP 2023. arXiv:2305.13245.
- DeepSeek-AI. *DeepSeek-V2.* 2024. arXiv:2405.04434. (Multi-head Latent Attention.)
- Shazeer. *Fast Transformer Decoding* and the follow-up literature on head sharing.

**Serving systems**
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with
  PagedAttention.* SOSP 2023. arXiv:2309.06180. (vLLM.)
- Yu et al. *Orca: A Distributed Serving System for Transformer-Based Generative
  Models.* OSDI 2022. (Iteration-level scheduling; the direct predecessor.)
- Prabhu et al. *vAttention: Dynamic Memory Management for Serving LLMs without
  PagedAttention.* 2024. arXiv:2405.04437.
- Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs.*
  NeurIPS 2024. arXiv:2312.07104. (RadixAttention / prefix caching.)
- Aminabadi et al. *DeepSpeed-Inference: Enabling Efficient Inference of Transformer
  Models at Unprecedented Scale.* 2022. arXiv:2207.00032.
- Sheng et al. *FlexGen: High-Throughput Generative Inference of Large Language Models
  with a Single GPU.* ICML 2023. arXiv:2303.06865. (Offloading.)
- Agrawal et al. *Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked
  Prefills.* 2023. arXiv:2308.16369.
- Zhong et al. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized
  Large Language Model Serving.* OSDI 2024. arXiv:2401.09670.
- Patel et al. *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.*
  ISCA 2024. arXiv:2311.18677.

**Squeezing the cache (beyond paging)**
- Hooper et al. *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
  Quantization.* NeurIPS 2024. arXiv:2401.18079.
- Liu et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024.
  arXiv:2402.02750.
- Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large
  Language Models.* NeurIPS 2023. arXiv:2306.14048. (Evicting unimportant KV entries.)
- Xiao et al. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024.
  arXiv:2309.17453. (StreamingLLM.)
- Nawrot et al. *Dynamic Memory Compression* / *Scissorhands* — adaptive KV allocation.
- Hooper et al. and the vLLM/quantization documentation for FP8 KV cache in practice.

**Alternative sequence models (the same fight, different ring)**
- Gu, Goel, Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.*
  ICLR 2022. arXiv:2111.00396. (S4.)
- Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023.
  arXiv:2312.00752.
- Sun et al. *Retentive Network: A Successor to Transformer for Large Language Models.*
  2023. arXiv:2307.08621. (RetNet.)
- Peng et al. *RWKV: Reinventing RNNs for the Transformer Era.* 2023. arXiv:2305.13048.
- Katharopoulos et al. *Transformers are RNNs: Fast Autoregressive Transformers with
  Linear Attention.* ICML 2020. arXiv:2006.16236.

**Approximate attention — and why it stalled**
- Child et al. *Generating Long Sequences with Sparse Transformers.* 2019.
  arXiv:1904.10509.
- Kitaev, Kaiser, Levskaya. *Reformer: The Efficient Transformer.* ICLR 2020.
  arXiv:2001.04451.
- Wang et al. *Linformer: Self-Attention with Linear Complexity.* 2020. arXiv:2006.04768.
- Beltagy, Peters, Cohan. *Longformer: The Long-Document Transformer.* 2020.
  arXiv:2004.05150.
- Tay et al. *Efficient Transformers: A Survey.* ACM Computing Surveys 2022.
  arXiv:2009.06732.

**Speed beyond the cache**
- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative
  Decoding.* ICML 2023. arXiv:2211.17192.
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.*
  2023. arXiv:2302.01318.
- Kwon et al. and the vLLM docs for continuous batching in practice; the Anyscale
  *continuous batching* write-up for the systems framing.

**Blogs, docs and code**
- vLLM project blog: *vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention.*
- Hugging Face *Text Generation Inference* (TGI) documentation.
- NVIDIA TensorRT-LLM documentation and examples.
- PyTorch blog: *INT4 Decoding GQA CUDA Optimizations for LLM Inference.* 2024.
- Raschka. *Grouped-Query Attention (GQA) in Modern LLMs.* 2025.
- Dao-AILab *flash-attention* repository; vLLM repository; SGLang repository.
