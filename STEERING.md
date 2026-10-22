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

## Open decisions (awaiting the editor)

From Debate 002, Section 05:

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
