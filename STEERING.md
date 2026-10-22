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

## Open decisions (awaiting the editor)

From Debate 001, Section 5:

- [ ] **F1** — Scope of the axis: three-way vs. "attention-on-LSTM, worth it?"
- [ ] **F2** — Task scope: forecasting only, or also classification / anomaly detection?
- [ ] **F3** — Evidence standard: benchmark papers only, or include protocol critiques?
- [ ] **F4** — Run an experiment, or keep it reading-based?
- [ ] **F5** — Deliverable shape: single essay vs. living evidence table?
- [ ] **F6** — Verdict policy: does a debate ever get marked *settled*, and by what standard?
