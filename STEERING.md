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
| 2026-10-22 | Project created as `ml-debates`; written-artifact style, not code. | First debate: attention-augmented RNNs vs. LSTMs for time series. |
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

## Open decisions (awaiting the editor)

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
