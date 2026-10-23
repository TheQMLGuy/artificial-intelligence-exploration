# Debate 004 — What is vLLM?

**Status:** open — draft 0.1
**Opened:** 2026-10-23
**Editor:** Divyansh Tripathi
**Written by:** AI assistant
**Framing as asked:** "What is vLLM?"

---

## 0. Framing: vLLM is not a model

The name invites the wrong picture. vLLM is **not** an architecture, not a model, and
not a training method. It is an **inference engine** — a serving system. Nothing about
it changes what the model computes; it changes how fast and how cheaply the model can
be *run*.

So "what is vLLM?" is really three questions stacked on each other:

1. **What problem does it solve?** Serving an LLM is a *memory* problem, not a compute
   problem — and memory is wasted badly by naive implementations.
2. **What is the trick?** A single idea — *paging the KV cache* — plus a scheduling
   change that the paging makes possible.
3. **How well, and against what?** A 2–4× throughput win over the previous
   state of the art, and a place in a crowded field of serving engines.

It helps to read this alongside **Debate 005 (the KV cache)**: vLLM's central trick is
a KV-cache technique, so this chapter is the *systems* answer to a problem that
chapter sets up.

---

## 1. The problem: serving an LLM is a memory problem

Autoregressive generation produces one token at a time. Each new token depends on all
previous ones, so the work is sequential — and at each step the model does a tiny
amount of math (a matrix–vector product) while moving an enormous amount of data. The
GPUs are not compute-starved; they are **memory-bandwidth-starved**.

The standard fix is **batching**: run many requests together so the cost of loading the
model weights is shared. But how many requests fit in a batch? That is decided by
memory — and the memory is dominated by the **KV cache**, the stored keys and values
of every token generated so far.

The numbers make it concrete. Serving a 13B-parameter model on an A100 with 40 GB of
memory, roughly **65% of the memory holds the weights** (static), and **over 30% holds
the KV cache** (dynamic, per request). For OPT-13B, the KV cache of a *single token*
is about 800 KB — and with sequences up to 2048 tokens, one request's cache can reach
**1.6 GB**. Only a few tens of requests fit even if every spare byte went to cache.

And the naive implementations waste most of that. To store a request's cache in
*contiguous* memory, existing systems pre-allocate a chunk sized to the request's
**maximum** possible length. The result is three kinds of waste — slots reserved for
future tokens, internal fragmentation (the request finished long before its maximum),
and external fragmentation from the allocator. The vLLM paper measured it: in prior
systems **only 20.4%–38.2% of the KV-cache memory was actually used** to store real
token states.

The trend makes this worse, not better: from A100 to H100, compute throughput more
than doubled, while memory capacity stayed at 80 GB. **Memory is the bottleneck, and
it is getting tighter.**

---

## 2. The trick: PagedAttention

The insight is an old one from operating systems. Programs want contiguous memory;
physical memory is fragmented. The OS solved this with **virtual memory and paging** —
fixed-size pages, mapped from logical to physical through a page table.

PagedAttention applies exactly that to the KV cache:

- A request's KV cache is split into **fixed-size blocks** (each holding the keys and
  values for a fixed number of tokens).
- The blocks **need not be contiguous** in physical memory.
- A **block table** per request maps its logical blocks to physical ones — pages,
  tokens as bytes, requests as processes.

The consequences are the whole point:

- **Near-zero waste.** Blocks are allocated on demand and all have the same size, so
  external fragmentation disappears and internal waste is confined to the last,
  partially-filled block — in practice **under 4% waste**, against the 60–80% of the
  naive scheme.
- **Sharing becomes possible.** Because sharing happens at block granularity, sequences
  that share a prefix can literally point at the same physical blocks. For parallel
  sampling this saved 6–10% of memory; for beam search, **38–55%**, and up to 66% on
  real chat data.

On top of the paging sits **continuous batching** (iteration-level scheduling): instead
of admitting a fixed batch and waiting for the longest sequence, the scheduler can drop
finished requests and admit new ones **every decoding iteration** — no padding, no
head-of-line blocking.

Together these give the headline result: **2–4× higher throughput than
FasterTransformer and Orca at the same latency**, with the gap widening for longer
sequences and larger models — and up to **22×** the request rate of FasterTransformer,
which had neither paging nor fine-grained scheduling. Model outputs are bit-for-bit
unchanged; this is pure systems work.

---

## 3. The debates

**D1. Memory or batching — which is the real lever?** Both, but not symmetrically. The
throughput win comes from *batching more requests*, and batching more requests is only
possible because the paging freed the memory. So the causal chain runs
paging → memory → batch size → throughput. Calling it "a batching trick" undersells
the memory work; calling it "a memory trick" hides why memory matters.

**D2. PagedAttention, or just manage the cache better?** A follow-up system (vAttention,
2024) argues that paging is not *necessary* — you can manage the KV cache dynamically
using CUDA's own virtual-memory APIs without introducing a custom block abstraction.
If that holds up, PagedAttention's win is less "a new algorithm" than "the right fix
for a limitation the existing allocators had." Worth tracking; it is the kind of claim
that ages one way or the other.

**D3. vLLM vs the field.** The serving landscape is crowded: **TensorRT-LLM** (NVIDIA;
often the fastest raw latency, at the cost of flexibility), **TGI** (Hugging Face),
**SGLang** (whose RadixAttention attacks *prefix* reuse, a different axis), and
**LMDeploy**. There is no universal winner — throughput, latency, hardware support,
and features trade off differently per workload. "vLLM is fastest" is a
workload-dependent claim, and should be quoted as one.

**D4. Is this research?** The vLLM/PagedAttention paper appeared at **SOSP**, an
operating-systems venue — not an ML conference. That is the debate in miniature: as
model architecture matures, the bottleneck moves to *systems*, and the most impactful
work stops looking like ML. Whether that is a healthy rebalancing or a sign that
architecture research is plateauing is a live question.

**D5. What vLLM is *not*.** It does not make the model smarter, change its output,
reduce its weights, or alter its math. It is a scheduler and a memory manager. Keeping
that boundary clear is most of the answer to "what is vLLM."

---

## 4. What the evidence, read together, suggests

1. **vLLM is an inference engine, not a model.** Its entire contribution is systems:
   memory management and scheduling.
2. **The problem it solves is memory-bound serving**, and specifically the waste and
   inflexibility of KV-cache allocation.
3. **PagedAttention is the core idea**: block the cache, map it through a table, share
   at block granularity — near-zero waste and real memory sharing.
4. **The measured win is 2–4× throughput** over the prior state of the art, larger for
   long sequences and complex decoding, with unchanged outputs.
5. **The claim is contested at the edges** — a follow-up questions whether paging is
   necessary at all, and the competitive landscape is workload-dependent.

---

## 5. Open forks — your call

**F1. Depth.** Stay at the level of "what is vLLM and why," or go into the mechanics —
block tables, the PagedAttention kernel, the scheduler, swapping/preemption?

**F2. Scope of the landscape.** Focus on vLLM alone, or compare the serving engines
(TensorRT-LLM, TGI, SGLang, LMDeploy) as a family?

**F3. Do we run an experiment?** A small local serving benchmark (throughput vs batch
size, with and without paging) would make the memory argument concrete. *This is now
the fourth debate to reach an experiment fork.*

**F4. Follow the follow-up.** Should we treat vAttention's "you don't need paging"
claim as a debate to chase down?

**F5. Deliverable shape.** Same as before?

**F6. Verdict policy.** Same — does a debate ever get marked *settled*?

---

## Cross-debate notes

- **Debate 005 (KV cache and attention)** sets up the problem this chapter solves:
  the KV cache is the bottleneck, and paging is one of several answers.
- **Debate 001** established that architecture is rarely the first-order variable.
  This chapter is the systems-level echo of that: the first-order variable for
  *serving* is memory management, not model choice.

---

## Sources (plain text)

- Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica. *Efficient Memory
  Management for Large Language Model Serving with PagedAttention.* SOSP 2023.
  arXiv:2309.06180. (vLLM.)
- vLLM project blog. *vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention.*
  2023.
- Anyscale. *Achieve 23x LLM Inference Throughput & Reduce p50 Latency* (continuous
  batching), 2023.
- Prabhu et al. *vAttention: Dynamic Memory Management for Serving LLMs without
  PagedAttention.* 2024. arXiv:2405.04437.
- Pope et al. *Efficiently Scaling Transformer Inference.* 2022. (KV cache analysis;
  memory-bandwidth bottleneck.)
- Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model.*
  2009. (Arithmetic intensity.)
