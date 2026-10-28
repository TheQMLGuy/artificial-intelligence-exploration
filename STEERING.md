# STEERING.md — decision log

The editor's decisions live here, so the direction of the project is auditable.
Anything not yet decided stays **open** until the editor rules on it.

## Ground rules (set at project start)

- The AI drafts; the editor decides.
- Artifacts are drafts until the editor marks them settled.
- Where a decision has real trade-offs, it is surfaced as an Open Fork rather than
  silently chosen.

## Decision log

| Date | Decision | Notes |
|------|----------|-------|
| 2026-10-22 | Project created as `artificial-intelligence-exploration`; written-artifact style, not code. | First debate: attention-augmented RNNs vs. LSTMs for time series. |
| 2026-10-22 | Debate 001 opened (draft 0.1). | Framing corrected: LSTM is itself an RNN, so the axis is three-way. |
| 2026-10-22 | Debate 001 bumped to draft 0.2: added the xLSTM case. | Axis widened to four-way (added "modern recurrence"). New fork F7 added. |
| 2026-10-22 | Debate 002 opened (draft 0.1). | "Why not x² / is nonlinearity all we need / why tanh vs trig." Six open forks. |
| 2026-10-22 | Site restructured: `docs/index.html` is now an index page; each debate has its own page. | Debate 001 moved to `docs/debate-001-attention-vs-lstm.html`. |
| 2026-10-22 | Debate 003 opened (draft 0.1): can a CNN do time series? | TCN / dilated causal convs / receptive field; ModernTCN + TimesNet; ROCKET + InceptionTime. Six open forks. |
| 2026-10-22 | Debate 002 bumped to draft 0.2: added the e^x case. | e^x passes the non-polynomial test and still fails on conditioning; ELU as the tamed exponential. |
| 2026-10-22 | Debate 002 bumped to draft 0.3: added the log x case. | log x fails on *domain* (undefined for x ≤ 0) — a third distinct failure mode. Taxonomy table added. |
| 2026-10-22 | Debate 002 bumped to draft 0.4: added section 04 ("But Fourier can approximate anything?"). | Fourier ≠ Taylor; complete basis ≠ good activation; spectral bias / frequency principle. Verdict on sin upgraded. |
| 2026-10-23 | Debate 004 opened (draft 0.1): what is vLLM? | Inference engine, not a model. PagedAttention; continuous batching; 2–4× throughput. Six open forks. |
| 2026-10-23 | Debate 005 opened (draft 0.1): what is the KV cache, and what is attention? | Attention as retrieval; the cache as its memoization; memory-bandwidth bottleneck; MHA/MQA/GQA; FlashAttention ≠ cache reduction. Six open forks. |
| 2026-10-23 | Debates 004 and 005 MERGED into one chapter (004: "What is vLLM, the KV cache, and attention?"). | The three topics are one chain: attention → cache → system. Old separate chapters removed from repo and Drive. |
| 2026-10-23 | Added "Further down the rabbit hole" reference sections to every chapter. | Categorized, annotated further-reading lists (papers, surveys, code) so each chapter can be read as a rabbit hole. |
| 2026-10-23 | Added "Tangent" asides throughout every chapter's *body* (editor request: rabbit-hole type overall, not just a reference section). | Verifiable surprise facts woven into the prose — LSTM had no forget gate until 2000; ReLU dates to 1969; GQA is mean-pooled + 5% uptraining; a 70B KV cache exceeds H100 memory; random-kernel ROCKET beats deep nets; etc. |
| 2026-10-23 | Full language pass on every chapter (editor: "the language is too AI-ish"). | Removed the AI tells: em-dashes, bold sentence-leads, punchy one-line paragraphs, "not X but Y" reveals, rhetorical questions. Rewritten as flowing prose. Facts, structure, tables and references unchanged. |
| 2026-10-28 | Site draft badges corrected to match the sources. | The 2026-10-23 language pass bumped every chapter without regenerating its page; 001 to 0.3, 002 to 0.5, 003 to 0.2, 004 to 0.2, both labels per page. |
| 2026-10-28 | Debate 005 opened (draft 0.1): what are deep belief networks? | The model, the training algorithm and the historical moment kept apart, because only one of the three still exists. Six open forks. |
| 2026-10-28 | Debate 006 opened (draft 0.1): what are neural radiance fields? | The representation, the renderer and the training signal kept apart. Seven open forks. |
| 2026-10-28 | Project renamed from `ml-debates` to `artificial intelligence exploration`. | Repository, site wordmark, running heads, README and the Brain entry renamed together; the old repository URL redirects. |
| 2026-10-28 | Debate 007 opened (draft 0.1): static vs contextual embeddings. | The unit (word type against token), the mechanism (recurrent against transformer) and frozenness kept apart, because they are three different questions. Seven open forks. |
| 2026-10-28 | Debate 008 opened (draft 0.1): what is deep learning? | The definition is one paragraph; the chapter is the causal argument between depth, scale and optimisation. Seven open forks. |

## Open decisions (awaiting the editor)

From Debate 008, Section 07:

- [ ] **D8-F1** — Definition, history, or causal argument?
- [ ] **D8-F2** — How much mathematics (linear-region count, scaling-law fits)?
- [ ] **D8-F3** — Run the grokking experiment on modular arithmetic?
- [ ] **D8-F4** — Do the scaling laws get their own chapter?
- [ ] **D8-F5** — How much of the theory gap (generalisation, loss landscapes, double descent)?
- [ ] **D8-F6** — Verdict policy when there is no accepted theory?
- [ ] **D8-F7** — Deliverable shape?

From Debate 007, Section 07:

- [ ] **D7-F1** — History, technical comparison, or decision guide?
- [ ] **D7-F2** — How much mathematics (negative sampling, GloVe loss, the linear reduction)?
- [ ] **D7-F3** — Run the word2vec / contextual / linear-projection experiment?
- [ ] **D7-F4** — Does the retrieval side (sentence embeddings, contrastive training) get its own treatment?
- [ ] **D7-F5** — How much of the anisotropy and rogue-dimension literature?
- [ ] **D7-F6** — Verdict policy when there is a clear practical winner?
- [ ] **D7-F7** — Deliverable shape?

From Debate 006, Section 06:

- [ ] **D6-F1** — Which is the chapter about: the model, the rendering formulation, or the task?
- [ ] **D6-F2** — How much of the mathematics (quadrature, transmittance recursion, sampling weights)?
- [ ] **D6-F3** — Run the tiny NeRF / encoding ablation?
- [ ] **D6-F4** — Does Gaussian splatting get its own chapter?
- [ ] **D6-F5** — How far into the applied territory (robotics, driving, street scale)?
- [ ] **D6-F6** — Verdict policy?
- [ ] **D6-F7** — Deliverable shape?

From Debate 005, Section 07:

- [ ] **D5-F1** — Which of the three things is the chapter about: model, algorithm, or moment?
- [ ] **D5-F2** — How much of the mathematics (partition function, log-likelihood gradient, CD bias)?
- [ ] **D5-F3** — Run the layer-wise pretraining ablation at small scale?
- [ ] **D5-F4** — Do the modern descendants (EBMs, diffusion) get their own treatment?
- [ ] **D5-F5** — Verdict policy for a largely historical chapter?
- [ ] **D5-F6** — Deliverable shape?

From Debate 004, Section 07 (merged vLLM + KV cache + attention):

- [ ] **D4-F1** — Depth of the attention section?
- [ ] **D4-F2** — Which axis to go deep on (architecture / management / precision / algorithm)?
- [ ] **D4-F3** — Run the KV-size + arithmetic-intensity experiment?
- [ ] **D4-F4** — Chase vAttention's "no paging needed" claim?
- [ ] **D4-F5** — vLLM alone, or the serving-engine family?
- [ ] **D4-F6** — Deliverable shape?
- [ ] **D4-F7** — Verdict policy?

From Debate 003, Section 06:

- [ ] **D3-F1** — Scope: forecasting only, or all five tasks?
- [ ] **D3-F2** — Which CNN: TCN only, or the modern branch (TimesNet / ModernTCN)?
- [ ] **D3-F3** — Run the experiment (TCN vs LSTM vs Transformer)?
- [ ] **D3-F4** — Make the receptive-field math a worked example?
- [ ] **D3-F5** — Deliverable shape: same as before?
- [ ] **D3-F6** — Verdict policy: same as before?

From Debate 002, Section 06:

- [ ] **D2-F1** — Scope: "why not x²" + "what makes an activation," or the full activation zoo?
- [ ] **D2-F2** — Mathematical depth: keep the one-line proof, or develop the Leshno theorem properly?
- [ ] **D2-F3** — Run the 1-D regression experiment (x² vs tanh vs ReLU vs sin)?
- [ ] **D2-F4** — Treat learned/adaptive activations as a separate camp?
- [ ] **D2-F5** — Deliverable shape: same as Debate 001, or different?
- [ ] **D2-F6** — Verdict policy: same as Debate 001?

From Debate 001, Section 06:

- [ ] **F1** — Scope of the axis: four-way vs. "attention-on-LSTM, worth it?"
- [ ] **F2** — Task scope: forecasting only, or also classification / anomaly detection?
- [ ] **F3** — Evidence standard: benchmark papers only, or include protocol critiques?
- [ ] **F4** — Run an experiment, or keep it reading-based?
- [ ] **F5** — Deliverable shape: single essay vs. living evidence table?
- [ ] **F6** — Verdict policy: does a debate ever get marked *settled*, and by what standard?
- [ ] **F7** — Weight of xLSTM: is recurrence *back* (a real fourth camp), or a well-researched special case with narrow wins?
