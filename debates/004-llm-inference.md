# Debate 004 — What is vLLM, the KV cache, and attention?

**Status:** open — draft 0.2
**Opened:** 2026-10-23
**Editor:** Divyansh Tripathi
**Written by:** AI assistant
**Framing as asked:** "What is vLLM?" and "What is KV cache and attention?"
**Note:** this chapter merges the two original questions into one, because they are one story. It replaces the earlier separate vLLM and KV-cache chapters.

---

## 0. Framing: one machine, three objects

Asked separately, these three look like three topics. They are one chain. Attention is the operation, a learned retrieval. The KV cache is what autoregressive generation forces attention to keep. vLLM is the system that manages that cache so the machine can actually be fed. Each one creates the next: the operation creates the cache, the cache becomes the bottleneck, and the bottleneck demands a system. The honest order is therefore attention, then the cache, then the system, and the three only make sense read together.

---

## 1. Attention, minimally

For each token, a self-attention layer projects its hidden state into three vectors: a query q = W_q · x, a key k = W_k · x, and a value v = W_v · x. Then, for each query, it compares against every key and takes a weighted average of the values:

    a_ij = softmax( q_i · k_j / sqrt(d) )
    o_i  = sum_j  a_ij · v_j

That is the whole operation, and read plainly it is a retrieval: the query is a search term, the keys are an index, and the output is a soft, weighted lookup into the values. Every token asks which earlier tokens are relevant to it and pulls a blend of them.

Two facts about its cost drive everything that follows. Compute is quadratic in sequence length, since the score matrix is N by N. And it has two phases with opposite bottlenecks. Prefill, the prompt, knows all tokens at once and is therefore a matrix-matrix product, which is compute-bound. Decode, generation, works one token at a time and is therefore a matrix-vector product, which is memory-bound. That asymmetry is the hinge of the chapter.

---

## 2. Why the cache exists

Language models generate autoregressively, which means that token t depends on all the tokens before it. To produce token t+1, the model must attend over the keys and values of tokens 1 to t. Without a cache, every step would recompute the keys and values of the entire prefix, the same vectors over and over, and across a generation of n tokens that is O(n²) redundant work. The KV cache fixes it by storing the key and value vectors as they are produced, so that each step computes only the new token's key and value and attends over the cached ones.

Two properties deserve emphasis, because they are commonly misremembered. The first is that the cache changes nothing about the output. It is memoisation, not approximation, and generation with and without it produces identical tokens; it buys speed and never accuracy. The second is that it moves the cost from compute to memory. With the cache, each decode step is O(n) in attention, since it must read the whole cache, and O(1) in weight matrix products. The redundant compute is gone, and what remains is a large and growing tensor that must be read on every step.

---

## 3. The size, and why it is the bottleneck

The cache's size is simple to write down. For a model with L layers, h key/value heads, head dimension d_h, context length S, in sixteen-bit precision, the number of bytes is 2 for keys and values, times 2 bytes, times L, times h, times d_h, times S, which is 4·L·h·d_h·S. It grows linearly in context length and in batch size, and it is not a rounding error. For LLaMA-3-70B, with L of 80, h of 64 and d_h of 128, the cache is about 10.7 GB at 4096 tokens and about 85.9 GB at 32,768 tokens, which is larger than the 80 GB of an H100. With GQA using eight key/value heads, the same figures fall to about 1.3 GB and about 10.7 GB.

At scale it dwarfs the weights. One analysis of an 8B model at batch 128 and 8192 tokens of context puts the MHA cache at 280 GB, which is about thirty times the model's own weights, falling to 69 GB with GQA.

The reason this matters is the roofline. A kernel's arithmetic intensity, meaning FLOPs per byte moved, decides whether it is compute-bound or memory-bound. On an H100 the ridge point is about 295 FLOPs per byte. Prefill sits well above it, whereas decode at small batch sits at roughly 1 FLOP per byte, two to three orders of magnitude below the ridge. The tensor cores idle, and measured GPU utilisation during decode can fall to around 7 per cent. The cache is therefore not large in the abstract so much as re-read every step while doing almost no arithmetic, which makes the cache's size, rather than the model's, the binding constraint on batch size, throughput and context length.

---

> **Tangent —** For a 70B model at 32k context, the KV cache alone is about 85.9 GB, which is larger than the entire 80 GB memory of an H100. The cache is bigger than the machine that has to hold it.

## 4. vLLM: the system that manages it

vLLM is not a model, an architecture, or a training method. It is an inference engine, a serving system, and nothing about it changes what the model computes. It changes how fast and how cheaply the model can be run.

The problem it addresses is that serving is memory-bound and the memory is dominated by the KV cache. To keep a request's cache contiguous, prior systems pre-allocated a chunk sized to the request's maximum length, and three kinds of waste followed: slots reserved for future tokens, internal fragmentation, and external fragmentation. The measured result was that only 20.4 to 38.2 per cent of KV-cache memory was actually used to store real token states.

The trick is PagedAttention, which borrows operating-system virtual memory paging. A request's KV cache is split into fixed-size blocks, the blocks need not be contiguous in physical memory, and a per-request block table maps logical blocks to physical ones, with pages becoming blocks, bytes becoming tokens, and processes becoming requests. The consequences are the whole point. Waste becomes near zero, because blocks are allocated on demand and all have the same size, which kills external fragmentation and confines internal waste to the last partly-filled block, in practice under 4 per cent against the 60 to 80 per cent of the naive scheme. And sharing becomes possible, because sequences that share a prefix can point at the same physical blocks; parallel sampling saved 6 to 10 per cent of memory and beam search saved 38 to 55 per cent, up to 66 per cent on real chat data.

On top of the paging sits continuous batching, or iteration-level scheduling, in which finished requests are dropped and new ones admitted every decoding iteration, with no padding and no head-of-line blocking. Together these give a throughput of two to four times that of FasterTransformer and Orca at the same latency, up to twenty-two times the request rate of FasterTransformer, with the gap widening for longer sequences and larger models. Model outputs are bit-for-bit unchanged, and this is pure systems work.

---

## 5. The debates

The first is architecture, meaning sharing the heads. MHA, the standard, has one key/value head per query head. MQA, from Shazeer in 2019, uses a single shared key/value head, which shrinks the cache by a factor of h but costs quality and requires replicating the single head across devices in distributed inference. GQA, from Ainslie and colleagues in 2023, groups the queries so that each group shares one key/value head, recovering most of the quality, and it is now the default in Llama 2 70B and Llama 3, Mistral and Qwen. MLA, from DeepSeek, caches a low-rank latent instead. The trade-off is explicit: quality against KV budget against parallelisability.

The second is management, meaning paging it. PagedAttention does not shrink the cache; it stops wasting it. The open question is whether paging is necessary at all, since vAttention, in 2024, argues that you can manage the cache dynamically with CUDA's own virtual-memory APIs, without a custom block abstraction.

The third is precision, meaning quantising it. Storing the cache in FP8 or INT4 halves or quarters the bytes, and FP8 gives a near-doubling of decode throughput in the memory-bound regime. The cautionary case is real, however: PyTorch found that an INT4 GQA cache read four times less data for no latency improvement at all, because the kernel was less bandwidth-efficient at INT4. Fewer bytes does not automatically mean faster.

The fourth is algorithmic, meaning not materialising it. FlashAttention is the standard example and also the site of a widespread confusion, because it does not reduce the KV cache. It is exact attention that avoids writing the N by N score matrix to slow HBM by tiling the computation into fast on-chip SRAM. On a GPT-2 attention block, HBM traffic fell from 40.3 GB to 4.4 GB and runtime from 41.7 milliseconds to 7.3, but the cache itself is unchanged. It slashes memory traffic and peak memory, which is a different problem on a different axis.

The fifth is approximation, the road not taken. Sparse and low-rank attention reduce FLOPs to near-linear in sequence length, yet many of them do not display wall-clock speedup against standard attention and have not gained wide adoption, precisely because they optimised FLOPs and ignored memory input and output. FlashAttention's lesson was the inverse: FLOPs are not time, and bytes moved are.

The sixth is the landscape. vLLM is one engine among several, alongside TensorRT-LLM from NVIDIA, which is often the fastest in raw latency but less flexible; TGI from Hugging Face; SGLang, whose RadixAttention attacks prefix reuse, which is a different axis; and LMDeploy. There is no universal winner, since throughput, latency, hardware support and features trade off differently per workload.

The seventh is whether this is research at all. The vLLM and PagedAttention paper appeared at SOSP, an operating-systems venue rather than an ML conference, which is the debate in miniature: as architecture matures the bottleneck moves to systems, and the most impactful work stops looking like ML. A systems-level trend sits above all of these, which is prefill and decode disaggregation, running the compute-bound prefill and the memory-bound decode on different hardware because they stress opposite resources.

---

> **Tangent —** PagedAttention is not so much a new idea as a borrowed one. It is operating-system virtual memory, transplanted: pages become blocks, bytes become tokens, processes become requests. The trick is roughly sixty years old, and it had simply never been applied to a KV cache.

## 6. What the evidence, read together, suggests

This is a reading, not a verdict. The editor decides what stands.

The three objects are one chain, in that attention forces the cache, the cache is the bottleneck, and vLLM is one answer to it. The cache is exact, never changing the output and only moving cost from compute to memory. Decode is memory-bandwidth-bound, and arithmetic intensity is the reason, which is why batching is the primary lever and why the cache is worth shrinking. The fixes live on different axes, being architecture, management, precision and algorithm, and they compose: GQA with PagedAttention, FP8 and FlashAttention is a coherent stack rather than four rival ideas. FLOPs are not wall-clock, and the approximate-attention story and the INT4 story are the same cautionary tale from opposite directions. vLLM, finally, is a memory manager and a scheduler rather than a model, and keeping that boundary clear is most of the answer to the question of what vLLM is.

---

## 7. Open forks — your call

**F1. Depth of the attention section.** Keep the minimal treatment, or develop it with multiple heads, positional encoding, causal masking and the reason for the square root of d?

**F2. Which axis to go deep on?** Architecture, management, precision, or algorithm? They are separable, and it is better to do one properly than four shallowly.

**F3. Do we run an experiment?** A small script that computes KV-cache size and arithmetic intensity across models, batch sizes and context lengths, and plots the roofline. It would be cheap and it would make the chapter quantitative.

**F4. The paging debate.** Chase vAttention's claim that you do not need paging?

**F5. The landscape.** Keep vLLM alone, or compare the serving engines as a family?

**F6. Deliverable shape.** Same as before?

**F7. Verdict policy.** Same, in that a debate does eventually get marked settled, or not?

---

## Cross-debate notes

Debate 001 argued about the architecture, whereas this chapter is about the cost of running it; the first-order variable has moved from model choice to memory management, which is the same theme one level down. Debate 003 found that a component's power is a prior rather than a capability, and here the component is the allocator, whose power is not extra compute but the absence of waste.

---

## Further down the rabbit hole

Everything below is real, and grouped so you can go as deep as you want.

**Foundations of attention**
- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762.
- Bahdanau, Cho, Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015. arXiv:1409.0473. (Attention before Transformers.)
- Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. arXiv:2205.14135.
- Dao. *FlashAttention-2.* 2023. arXiv:2307.08691.
- Shah et al. *FlashAttention-3.* 2024. arXiv:2407.08608.
- Milakov & Gimelshein. *Online normalizer calculation for softmax.* 2018. arXiv:1805.02867. (The trick that makes tiling exact.)

**The KV cache and the memory wall**
- Pope et al. *Efficiently Scaling Transformer Inference.* MLSys 2023. arXiv:2211.05102.
- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model.* CACM 2009. (Arithmetic intensity and the ridge point.)
- Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019. arXiv:1911.02150. (MQA.)
- Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023. arXiv:2305.13245.
- DeepSeek-AI. *DeepSeek-V2.* 2024. arXiv:2405.04434. (Multi-head Latent Attention.)

**Serving systems**
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180. (vLLM.)
- Yu et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI 2022. (Iteration-level scheduling; the direct predecessor.)
- Prabhu et al. *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention.* 2024. arXiv:2405.04437.
- Zheng et al. *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS 2024. arXiv:2312.07104. (RadixAttention.)
- Aminabadi et al. *DeepSpeed-Inference.* 2022. arXiv:2207.00032.
- Sheng et al. *FlexGen: High-Throughput Generative Inference with a Single GPU.* ICML 2023. arXiv:2303.06865.
- Agrawal et al. *Sarathi: Piggybacking Decodes with Chunked Prefills.* 2023. arXiv:2308.16369.
- Zhong et al. *DistServe: Disaggregating Prefill and Decoding.* OSDI 2024. arXiv:2401.09670.
- Patel et al. *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.* ISCA 2024. arXiv:2311.18677.

**Squeezing the cache (beyond paging)**
- Hooper et al. *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.* NeurIPS 2024. arXiv:2401.18079.
- Liu et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024. arXiv:2402.02750.
- Zhang et al. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference.* NeurIPS 2023. arXiv:2306.14048. (Evicting unimportant KV entries.)
- Xiao et al. *Efficient Streaming Language Models with Attention Sinks.* ICLR 2024. arXiv:2309.17453. (StreamingLLM.)

**Alternative sequence models (same fight, different ring)**
- Gu, Goel, Ré. *Efficiently Modeling Long Sequences with Structured State Spaces.* ICLR 2022. arXiv:2111.00396. (S4.)
- Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* 2023. arXiv:2312.00752.
- Sun et al. *Retentive Network: A Successor to Transformer for LLMs.* 2023. arXiv:2307.08621. (RetNet.)
- Peng et al. *RWKV: Reinventing RNNs for the Transformer Era.* 2023. arXiv:2305.13048.
- Katharopoulos et al. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML 2020. arXiv:2006.16236.

**Approximate attention, and why it stalled**
- Child et al. *Generating Long Sequences with Sparse Transformers.* 2019. arXiv:1904.10509.
- Kitaev, Kaiser, Levskaya. *Reformer: The Efficient Transformer.* ICLR 2020. arXiv:2001.04451.
- Wang et al. *Linformer: Self-Attention with Linear Complexity.* 2020. arXiv:2006.04768.
- Beltagy, Peters, Cohan. *Longformer: The Long-Document Transformer.* 2020. arXiv:2004.05150.
- Tay et al. *Efficient Transformers: A Survey.* ACM Computing Surveys 2022. arXiv:2009.06732.

**Speed beyond the cache**
- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. arXiv:2211.17192.
- Chen et al. *Accelerating Large Language Model Decoding with Speculative Sampling.* 2023. arXiv:2302.01318.

**Blogs, docs and code**
- vLLM project blog: *vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention.*
- Hugging Face *Text Generation Inference* (TGI) documentation.
- NVIDIA TensorRT-LLM documentation and examples.
- PyTorch blog: *INT4 Decoding GQA CUDA Optimizations for LLM Inference.* 2024.
- Raschka. *Grouped-Query Attention (GQA) in Modern LLMs.* 2025.
- Repositories: Dao-AILab *flash-attention*; *vllm-project/vllm*; *sgl-project/sglang*.
