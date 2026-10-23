# Debate 005 — What is the KV cache, and what is attention?

**Status:** open — draft 0.1
**Opened:** 2026-10-23
**Editor:** Divyansh Tripathi
**Written by:** AI assistant
**Framing as asked:** "What is KVCache and Attention?"

---

## 0. Framing: two questions, one object

The two are usually asked as if they were separate topics. They are not. **The KV cache
is not a separate invention — it is a consequence.** It falls straight out of how
attention works and how generation works, and it becomes, in the process, the single
most important object in the economics of running an LLM.

So the honest order is: **attention first, then the cache as the thing it forces.**

1. **Attention** is a retrieval operation — a learned lookup.
2. **The KV cache** is the memoization that *autoregressive* decoding makes necessary.
3. Together they explain why inference is **memory-bound**, why long context is
   expensive, and why an entire field (GQA, PagedAttention, FlashAttention, KV
   quantization) exists to shrink or manage one tensor.

---

## 1. Attention, minimally

For each token, a self-attention layer projects its hidden state into three vectors:

- a **query** `q = W_q · x`, a **key** `k = W_k · x`, and a **value** `v = W_v · x`.

Attention then, for each query, compares it against every key and takes a weighted
average of the values:

    a_ij = softmax( q_i · k_j / sqrt(d) )
    o_i  = sum_j  a_ij · v_j

That is the whole operation. Read plainly, it is a **retrieval**: the query is a search
term, the keys are an index, and the output is a soft, weighted lookup into the values.
Every token asks "which earlier tokens are relevant to me?" and pulls a blend of them.

Two facts about its cost matter for everything that follows:

- **Compute is quadratic** in sequence length: the score matrix is `N × N`.
- **It has two phases with opposite bottlenecks:**
  - **Prefill** (the prompt): all tokens are known at once, so the work is a big
    matrix–matrix product — **compute-bound**.
  - **Decode** (generation): one token at a time, so the work is a matrix–*vector*
    product — **memory-bound**.

That last asymmetry is the hinge of the whole chapter.

---

## 2. Why the cache exists

Language models generate **autoregressively**: token `t` depends on all tokens before
it. To produce token `t+1`, the model must attend over the keys and values of tokens
`1 … t`.

Without a cache, every generation step would recompute the keys and values of the
*entire* prefix — the same vectors, over and over. Across a generation of `n` tokens
that is `O(n²)` redundant work. The **KV cache** fixes it: store the key and value
vectors as they are produced, so each step computes only the *new* token's `k` and `v`
and attends over the cached ones.

Two properties deserve emphasis, because they are commonly misremembered:

- **The cache changes nothing about the output.** It is memoization, not approximation —
  generation with and without the cache produces *identical* tokens. It buys speed,
  never accuracy.
- **It moves the cost from compute to memory.** With the cache, each decode step is
  `O(n)` in attention (it must *read* the whole cache) and `O(1)` in weight matrix
  products. The redundant compute is gone; what remains is a large, growing tensor that
  must be read on every single step.

---

## 3. The size — and why it is the bottleneck

The cache's size is simple to write down. For a model with `L` layers, `h` key/value
heads, head dimension `d_h`, context length `S`, in 16-bit:

    KV bytes  =  2 (keys and values) × 2 (bytes) × L × h × d_h × S  =  4 · L · h · d_h · S

It grows **linearly in context length and in batch size** — and it is not a rounding
error. For **LLaMA-3-70B** (`L=80, h=64, d_h=128`):

- at 4,096 tokens: **≈ 10.7 GB** per sequence;
- at 32,768 tokens: **≈ 85.9 GB** — *larger than the 80 GB of an H100.*

At scale it dwarfs the weights. One analysis of an 8B model at batch 128 and 8,192
tokens of context puts the MHA KV cache at **280 GB — about 30× the model's own
weights**. (With GQA it drops to 69 GB. More on that below.)

Now put that against the roofline. A kernel's **arithmetic intensity** — FLOPs per byte
moved — decides whether it is compute- or memory-bound. On an H100 the *ridge point* is
about **295 FLOP/byte**. Prefill sits well above it. Decode, at small batch, sits at
roughly **1 FLOP/byte** — two to three orders of magnitude below the ridge. The tensor
cores idle; measured GPU utilisation during decode can fall to around **7%**.

**This is the answer to "why is the KV cache such a big deal":** it is not that the
cache is large in the abstract, it is that decoding must *re-read it every step* while
doing almost no math — so the cache's size, not the model's, becomes the binding
constraint on batch size, throughput, and context length.

---

## 4. The debates: five ways to fight the cache

Everything in modern inference efficiency is, roughly, an attack on this one tensor —
but the attacks are on **different axes**, which is why they compose.

**D1. Architecture — share the heads.** The most fundamental lever: make fewer keys and
values to begin with.

- **MHA** (standard): one key/value head per query head.
- **MQA** (Shazeer, 2019): a *single* shared key/value head for all queries — the cache
  shrinks by a factor of `h`, but model quality suffers, and in distributed inference the
  single head must be replicated on every device, eroding the saving.
- **GQA** (Ainslie et al., 2023): the compromise — queries are grouped, each group
  sharing one key/value head. This sits between MHA and MQA and recovers most of the
  quality; it is now the default in Llama 2 70B / Llama 3, Mistral, and Qwen.
- **MLA** (DeepSeek): compress the hidden state into a low-rank latent and cache *that*,
  then up-project — high arithmetic intensity, at the cost of some parallelizability.

  The trade-off is explicit: **quality vs. KV budget vs. parallelizability.** The GQA
  paper's finding — that intermediate grouping approaches MHA quality while keeping most
  of MQA's benefit — is why GQA, not MQA, won.

**D2. Management — page it.** PagedAttention (**Debate 004**) does not shrink the cache
at all; it stops *wasting* it. Blocking the cache and mapping it through a table takes
memory waste from 60–80% down to under 4% and enables block-level sharing. This is the
orthogonal axis: same tensor, better allocator.

**D3. Precision — quantize it.** Store the cache in FP8 or INT4 instead of BF16 and you
halve or quarter the bytes. FP8 KV cache gives a near-2× decode throughput win in the
memory-bound regime. But the cautionary case is real: PyTorch found that an INT4 GQA
cache read **4× less data for no latency improvement at all**, because the kernel was
*less bandwidth-efficient* at INT4. **Fewer bytes ≠ faster**, unless the kernel can
actually move them.

**D4. Algorithm — don't materialize it (FlashAttention).** This is where a widespread
confusion lives. **FlashAttention does *not* reduce the KV cache.** It is *exact*
attention that avoids ever writing the `N × N` score matrix to slow HBM, by tiling the
computation into fast on-chip SRAM. The measured effect on a GPT-2 attention block:
HBM reads/writes fell from **40.3 GB to 4.4 GB**, and runtime from **41.7 ms to
7.3 ms**. So it slashes memory *traffic* and peak memory (making it linear in sequence
length) — but the KV cache itself is unchanged. Different problem, different axis.

**D5. Approximation — the road not taken.** Sparse and low-rank attention reduce the
*FLOPs* to linear or near-linear in sequence length. Yet many "do not display
wall-clock speedup against standard attention, and have not gained wide adoption" —
precisely because they optimised FLOPs and ignored memory I/O. FlashAttention's lesson
was the inverse: **FLOPs are not time; bytes moved are.**

A sixth, systems-level trend sits above all of these: **prefill/decode disaggregation** —
running the compute-bound prefill and the memory-bound decode on *different* hardware,
since they stress opposite resources.

---

## 5. What the evidence, read together, suggests

1. **Attention is retrieval; the KV cache is its memoization.** The cache is exact —
   it never changes the output, only the speed.
2. **The cache is the binding constraint.** It grows with batch × context and, at long
   context, exceeds the model weights many times over.
3. **Decode is memory-bandwidth-bound**, and the arithmetic-intensity argument is the
   reason: too few FLOPs per byte read, so the hardware waits on memory. This is why
   *batching* is the primary lever and why the cache is worth shrinking.
4. **The fixes live on different axes** — architecture (share heads), management (page),
   precision (quantize), algorithm (tile) — and they **compose**. GQA + PagedAttention +
   FP8 + FlashAttention is a coherent stack, not four competing ideas.
5. **FLOPs are not wall-clock.** The approximate-attention story and the INT4 story are
   the same cautionary tale from opposite directions.

---

## 6. Open forks — your call

**F1. Depth of the attention section.** Keep the minimal "retrieval" treatment, or
develop it (multiple heads, positional encoding, causal masking, why `sqrt(d)`)?

**F2. Which axis to go deep on?** Architecture (MHA→GQA→MLA), management (paging),
precision (quantization), or algorithm (FlashAttention)? *They are separable; we could
do one properly rather than all four shallowly.*

**F3. Do we run an experiment?** A small script that computes KV-cache size and
arithmetic intensity across models, batch sizes, and context lengths — and plots the
roofline — would make the whole chapter quantitative. *Cheap and decisive.*

**F4. Reconcile with Debate 004?** The two chapters overlap deliberately (paging appears
in both). Keep them separate with cross-links, or merge into one "LLM inference" chapter?

**F5. Deliverable shape.** Same as before?

**F6. Verdict policy.** Same — does a debate ever get marked *settled*?

---

## Cross-debate notes

- **Debate 004 (vLLM)** is the systems answer to the problem this chapter defines:
  the KV cache is the bottleneck, and paging is one of the ways to attack it.
- **Debate 001 (attention vs LSTM)** argued about the *architecture*; this chapter is
  about the *cost of running* it. The first-order variable moved from model choice to
  memory management — the same theme, one level down.

---

## Sources (plain text)

- Vaswani et al. *Attention Is All You Need.* 2017. (Original attention.)
- Pope et al. *Efficiently Scaling Transformer Inference.* 2022. (KV cache size;
  memory-bandwidth bottleneck.)
- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019. (MQA.)
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from
  Multi-Head Checkpoints.* 2023. (GQA.)
- DeepSeek-AI. *DeepSeek-V2* (Multi-head Latent Attention), 2024.
- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness.* NeurIPS 2022. arXiv:2205.14135.
- Kwon et al. *Efficient Memory Management for LLM Serving with PagedAttention.*
  SOSP 2023. arXiv:2309.06180.
- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model.*
  2009. (Arithmetic intensity and the ridge point.)
- PyTorch blog. *INT4 Decoding GQA CUDA Optimizations for LLM Inference.* 2024.
- Raschka. *Grouped-Query Attention (GQA) in Modern LLMs.* 2025.
