# ml-debates

A living archive of written debates on machine learning. The writing is done by an
AI assistant; the direction, scope, and verdicts are set by the human editor.

## How this works

This repo is a *written-artifact* project, not a code project. Each debate is a
document that:

1. **Frames the question precisely** — including fixing the question when the way
   it is posed hides a category error.
2. **Gives the strongest case for each side**, in that side's own terms.
3. **Reports what the evidence actually shows**, including the skeptical reading
   that argues the debate is measured on a flawed benchmark.
4. **Ends with Open Forks** — the concrete decisions the editor makes next.

The AI drafts. The editor decides. Nothing in here is treated as *settled* until
the editor says so; every artifact carries a status line and a draft number.

## The protocol

- **One debate per file** under `debates/`, numbered in order.
- **Draft, not decree.** Every artifact is a draft until the editor marks it
  otherwise. Claims are attributed to their sources, not asserted as fact.
- **Forks over guesses.** Where a decision has real trade-offs, the artifact
  surfaces it as an Open Fork rather than silently picking a direction.
- **The editor can overrule anything** — scope, framing, evidence standard,
  whether we run experiments, and the final verdict.
- **Decisions are logged** in `STEERING.md` so the direction of the project is
  auditable.

## Index

| # | Debate | Status |
|---|--------|--------|
| 001 | Attention-augmented RNNs vs. LSTMs for time series | open — draft 0.2 |
| 002 | Why not x²? Is nonlinearity all we need, and why tanh? | open — draft 0.1 |

## Files

- `debates/` — the artifacts themselves (the source text).
- `docs/` — the published site (GitHub Pages): an index page and one page per debate.
- `STEERING.md` — the decision log; what the editor has decided and what is still open.
