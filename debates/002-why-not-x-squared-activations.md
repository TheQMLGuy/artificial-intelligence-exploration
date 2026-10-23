# Debate 002 — Why not x²? Is nonlinearity all we need, and why tanh?

**Status:** open — draft 0.4
**Opened:** 2026-10-22
**Editor:** Divyansh Tripathi
**Written by:** AI assistant
**Framing as asked:** "Why don't we use y = x² as an activation function? Even though it's a nonlinear function — is nonlinearity everything we need to make a function an activation function, or are we missing something? And why only tanh? Why not other trigonometric ones?"
**Follow-up:** "...and why not e^x, since it's also nonlinear and non-polynomial?" ... "What about log x?"

---

## 0. Framing: the question carries three assumptions

The question bundles three claims, and all three are false or half-true. Naming
them is most of the work:

1. **"A function is an activation function iff it is nonlinear."** Nonlinearity is
   *necessary* but nowhere near *sufficient*.
2. **"Nonlinearity is everything we need."** No — the *shape* of the nonlinearity,
   and how it behaves under composition and gradient flow, matters more than the
   mere fact of being nonlinear.
3. **"tanh is the only choice, and trigonometry is unexplored."** Both false. There
   is a whole zoo of activations (sigmoid, ReLU, GELU, Swish, Mish, ...), and
   trigonometric activations are a real, published tool — just a *specialised* one.

So the debate is really: **what does an activation function actually have to do,
and why do x², e^x, log x, and most trigonometric functions fail that test while
tanh passes it?**

---

## 1. Why x², e^x and log x are not "just other nonlinearities"

### 1.1 It is the one case the theorem excludes

There is a clean, famous result. Leshno, Lin, Pinkus & Schocken (1993): a
single-hidden-layer feedforward network is a universal approximator **if and only
if the activation function is not a polynomial**. Polynomial activations are the
single class the universal approximation theorem rules out.

That is not a heuristic — it is an iff. x² sits exactly in the excluded class.
Hornik's earlier results had given sufficient conditions (bounded, continuous,
non-constant); Leshno et al. sharpened it to the minimal necessary-and-sufficient
condition, and "non-polynomial" is it.

### 1.2 The one-line proof

A single hidden layer with activation σ computes:

    y = Σⱼ vⱼ · σ(wⱼ·x + bⱼ)

Put σ(t) = t². Each term is vⱼ(wⱼ·x + bⱼ)², and the sum expands to:

    y = xᵀ A x + cᵀ x + d

— a **quadratic form**. So no matter how wide you make the layer, a one-hidden-
layer x² network is *just a quadratic polynomial in the input*. It cannot
represent x³, or sin(x), or anything that is not a degree-2 polynomial. Adding
width adds nothing; the model class is frozen at "parabolas."

This is why "it's nonlinear, so it should work" fails: nonlinearity is not the
currency. The currency is whether the family of functions you can build is dense
in the space of functions you care about.

### 1.3 Depth does not rescue it — it wrecks it

You might think: fine, stack layers. Composing L quadratic layers gives a
polynomial of degree 2^L. Polynomials *are* dense, so asymptotically this is
technically a universal approximator again. In practice it is a disaster:

- **Values explode.** |x|^(2^L) grows doubly-exponentially with depth. Forward
  activations leave the well-behaved range and overflow to NaN. This is the
  "escaping activation problem" documented in the *Sisyphus* study of polynomial
  activations: forward values inevitably grow without bound, causing NaNs or
  garbage approximations.
- **Gradients are ill-conditioned.** The derivative is 2x — unbounded, so the
  gradient magnitude depends on how big the activations happen to be. Near zero
  the gradient vanishes; far from zero it explodes. You get *both* classic
  pathologies depending on the region.

### 1.4 It is also lossy and badly shaped

- **Not injective.** x² maps x and −x to the same value. A single unit literally
  cannot tell a positive pre-activation from its negative — the sign is destroyed.
- **Not monotonic.** It goes down then up, so it is non-monotone and has a
  stationary point at 0.
- **Unbounded above, non-negative range [0, ∞).** It can never output a negative
  value, and it has no ceiling.

### 1.5 But it is not "forbidden" — it is a bad *default*

Honesty demands the caveat. x² and polynomial activations *are* used, where the
objective is different:

- **Privacy-preserving inference.** Replacing ReLU with x² removes the need for
  garbled circuits (x² is just a multiply under homomorphic encryption), cutting
  latency dramatically. The *Sisyphus* paper exists precisely because practitioners
  want x² to work — and it documents why it struggles past a few layers.
- **Recent polynomial/Hermite activations** (2025) can train GPT-2 and ConvNeXt,
  but only with a carefully designed variance-preserving initialisation, and the
  authors note they had to solve exactly the exploding/vanishing problem that
  makes polynomials hard.

So the accurate statement is: **x² is not illegal, it is a poor general-purpose
prior.** It needs special handling and buys you less expressivity per layer.

### The mirror case: why not e^x?

The follow-up question is sharper than it looks, because **e^x passes the very
test that x² fails.** It is nonlinear, and it is emphatically *non-polynomial* — so
the 1993 theorem's condition is satisfied, and an e^x network is, in principle, a
universal approximator. On **expressivity, e^x is fine.**

It also has several properties x² lacks: it is monotonic, injective, smooth, and
convex. So if "nonlinear", or "non-polynomial", or "monotonic" were the whole
story, e^x would be a perfectly good activation.

It fails on the *other* axis — **conditioning** — and it fails harder than x² does:

- **It is unbounded above.** e^x → ∞. Compose layers and you get an *exponential
  tower*: e^(W₂·e^(W₁·x)). Each layer exponentiates the last, so values overflow to
  `inf` after a handful of layers. This is the exploding-activation problem taken to
  its extreme.
- **The gradient *is* the activation.** d/dx e^x = e^x. So wherever the activation
  is large, the gradient is equally large — exploding gradients, by construction.
- **It vanishes on the other side.** As x → −∞, e^x → 0 and its gradient → 0. So e^x
  has a *vanishing* gradient for negative inputs and an *exploding* one for positive
  inputs — both classic pathologies in a single function.
- **It is not zero-centred.** Its range is (0, ∞): never negative. Same all-positive
  gradient problem as the logistic sigmoid — and indeed the sigmoid *is* 1/(1+e^−x),
  so e^x is the sigmoid's raw ingredient.
- **A curiosity: the bias does nothing.** Because e^(w·x + b) = e^b · e^(w·x), the
  threshold term is absorbed into a positive scaling. The bias that the 1993 theorem
  needs for functions like sin is redundant here — a hint that e^x's function class
  is more rigid than it first appears.

**So what did the field do?** It did not throw the exponential away — it *tamed*
it. Almost every bounded activation is e^x in a wrapper: sigmoid is 1/(1+e^−x),
tanh is (e^x − e^−x)/(e^x + e^−x), softplus is log(1+e^x), softmax *normalises* it,
and attention uses a normalised exponential as its kernel. The most direct answer is
**ELU** (Clevert, Unterthiner & Hochreiter, 2015): keep the exponential only on the
*negative* side, where it saturates gracefully, and use the identity on the positive
side, where the exponential would explode:

    ELU(x) = x            if x > 0
             α(e^x − 1)   if x ≤ 0

ELU is, quite literally, "e^x with the dangerous half removed."

The set makes this artifact's point sharper than any one case alone: **x² fails
expressivity; e^x passes expressivity and fails conditioning.** Neither nonlinearity
nor non-polynomiality is the currency on its own — conditioning under composition is
co-equal. And log x will fail in a third way entirely.

### The third case: why not log x?

Log x fails in a way neither of the others does — **it is not even defined on half
the real line.** An activation function must be a map from ℝ to ℝ: it has to accept
every real pre-activation — positive, zero, or negative. log x is undefined for
x ≤ 0, so any neuron with a non-positive input produces `NaN`. This is a **domain**
failure, and it is fatal before any question of conditioning even arises.

You can patch the domain, but you then inherit a conditioning problem:

- **Use log(1 + x)** (the usual fix): defined for x > −1, but its derivative is
  1/(1+x), which → 0 as x → ∞. **Vanishing gradients for large inputs** — it
  saturates, one-sided, like a tanh that only flattens on the right.
- **Use log(1 + |x|), or the symmetric log:** now defined on all of ℝ, but *even*
  (log|−x| = log|x|), so it destroys the sign — the same defect as x². Its
  derivative, 1/(1+|x|), → 0 at *both* ends: vanishing gradients everywhere except
  near zero.
- **Bare log near zero:** the derivative 1/x → ∞ as x → 0⁺. So the unpatched
  function has an *exploding* gradient at the boundary and a *vanishing* one at
  infinity.

It is also **unbounded**, so it fails the boundedness condition of the original
Cybenko universal-approximation theorem (it does satisfy Leshno's non-polynomial
condition, so expressivity is again not the problem).

There is a symmetry worth naming: **log x and e^x are inverses** — mirror images
across the line y = x. Where e^x explodes as x → +∞, log x explodes as x → 0⁺.
Where e^x vanishes as x → −∞, log x's gradient vanishes as x → +∞. They are the
*same failure, reflected*, and they are tamed by the same move: bound the domain and
the range.

**And again the field tamed it rather than discarding it.** Log is everywhere in
machine learning — but wrapped:

- **symlog** (also called the logarithmic activation function, LAF): the symmetric
  version, effectively sign(x)·log(1+|x|), used in practice to compress huge dynamic
  ranges — it is the reward/value transform in Dreamer-style RL agents.
- **softplus = log(1 + e^x)** — the most famous log-based activation, and itself a
  composition of the two functions we have been discussing.
- **log-sigmoid = log σ(x)** — used as an activation.
- **NLReLU = log(1 + max(0, x))** — log applied to ReLU's positive side only, keeping
  ReLU's sparsity while softening its growth.

The three candidates together form a small taxonomy of *why an activation fails* —
and none of the reasons is "because it isn't nonlinear":

| Candidate | Failure mode | Still used as |
|---|---|---|
| x² | **expressivity** — it is a polynomial | polynomial nets (private inference) |
| e^x | **conditioning** — unbounded, explosive | sigmoid, tanh, softplus, ELU |
| log x | **domain** — undefined for x ≤ 0 | symlog, softplus, log-sigmoid, NLReLU |

That table is the answer to the original question, stated exactly: nonlinearity is
the entry ticket, not the qualification.

---

## 2. So what does an activation function actually have to do?

Nonlinearity is necessary and insufficient. The properties that actually matter,
and how the usual suspects score:

| Property | Why it matters | x² | e^x | log x | tanh | ReLU | sin |
|---|---|---|---|---|---|---|---|
| Defined on all ℝ | an activation must accept every real input | yes | yes | **no** | yes | yes | yes |
| Nonlinear | breaks linear collapse; **necessary** | yes | yes | yes | yes | yes | yes |
| Non-polynomial | universal-approximation guarantee | **no** | yes | yes | yes | yes | yes |
| Monotonic | *helps* optimisation; **not required** | no | yes | yes | yes | yes | no |
| Zero-centred output | avoids zig-zag gradient updates | no | **no** | no | yes | no | yes |
| Bounded / controlled growth | numerical stability under depth | no | **no** | no | yes | linear | yes |
| Well-behaved gradient | no exploding / vanishing | **no** | **explodes** | **vanishes** | saturates | 0 or 1 | oscillates |
| Cheap to compute | speed at scale | cheap | exp | log | exp | cheapest | exp |
| Right inductive bias | matches the task | — | — | — | general | general | periodic |

Three corrections to the "nonlinearity is everything" theory fall straight out of
this table:

1. **Non-polynomiality is necessary but not sufficient.** It is the actual
   requirement for the classical guarantee — and it is the exact thing x² lacks.
   But e^x *has* it, and is still unusable (see section 1).
2. **Monotonicity is not required.** This is the part most people get wrong.
   GELU, Swish and Mish are all *non-monotonic* (they dip slightly negative near
   zero) and they are state of the art. Recent theory even establishes
   learnability for non-monotonic activations such as SiLU/Swish and GELU. So the
   folk rule "activations must be monotonic" is obsolete.
3. **Conditioning beats expressivity in practice.** A shallow polynomial network
   is too weak; a deep one is too unstable. The activation that wins is the one
   whose *derivative* keeps gradients in a sane range through many layers.

---

## 3. Why tanh — and why not (only) the other trigonometric functions?

### 3.1 Why tanh won, historically

- **A bounded, smooth, monotonic squash** with range (−1, 1): numerically safe,
  differentiable everywhere, and it satisfies the sigmoidal universal-approximation
  conditions.
- **Zero-centred**, unlike the logistic sigmoid. This is the real reason it beat
  sigmoid. If every activation feeding a neuron is positive, the gradient w.r.t.
  the weights is *always all-positive or all-negative* — so the weight vector can
  only move along one diagonal, and it has to "zig-zag" to reach the target. LeCun
  called this out in *Efficient BackProp*; tanh removes it.
- **Bigger gradient near the origin.** Sigmoid's derivative peaks at 0.25; tanh's
  reaches 1.0. Larger local gradients mean faster learning in the linear regime.
- **It is a rescaled logistic:** tanh(x) = 2·σ(2x) − 1. So it inherits sigmoid's
  guarantees for free.
- **It pairs naturally with gates.** In an LSTM, logistic sigmoids gate the flow
  and tanh squashes the cell state — a division of labour that has stuck.

### 3.2 But tanh has real weaknesses

It saturates: for large |x| the gradient goes to zero, giving the vanishing-gradient
problem in deep stacks. It also costs an exponentiation. Those two facts are
exactly why **ReLU displaced tanh** as the default in deep feedforward networks —
ReLU is monotone, piecewise-linear (so its gradient is 0 or 1, never tiny on the
positive side), and dirt cheap. "Why only tanh?" has a one-word answer: **it
isn't** — it is a legacy default that lost the deep-net crown.

### 3.3 The trigonometric question

It is not that trigonometry is forbidden. It is that **most trigonometric functions
violate the properties in the table**:

- **sin and cos are periodic.** They map inputs that are far apart to the same
  output — the function is not injective *everywhere*, and it has infinitely many
  local minima. For a general function approximator on an unbounded domain,
  periodicity is simply the wrong prior: it says "the world repeats," which is
  true for signals and false for most everything else.
- **A lovely subtlety.** In the 1993 theorem, the *threshold* (bias) is essential.
  The family {sin(w·x)} contains only *odd* functions, so without a bias term it
  cannot even represent cos(x) — an even function. The offset is not a detail; it
  is what makes the family dense.
- **tan is hopeless** as an activation: it has poles, is undefined at π/2, and is
  unbounded — the opposite of what you want.

And yet **sin is used** — where periodicity *is* the right prior:

- **SIREN** (Sitzmann et al., 2020) uses sin as the activation for implicit neural
  representations. It represents images, audio, wavefields and 3-D shapes *and
  their derivatives* far better than ReLU or tanh, and it has an elegant property:
  the derivative of a SIREN is itself a SIREN (d/dx sin = cos = a phase-shifted
  sin). It needs a carefully designed initialisation (ω₀ = 30 for the first layer),
  which is the price of the periodic prior.
- **Fourier / trigonometric activations** (2025) can train GPT-2 and ConvNeXt using
  a trigonometric basis with variance-preserving initialisation.

So the honest answer to "why not other trigonometric ones?" is: **sin is not
unused — it is specialised.** tanh is a general-purpose squash that happens to be
built from exponentials; sin is a periodic prior that wins exactly when the data
is periodic. The choice of activation is a choice of *inductive bias*, not a
search for "any nonlinear function."

---

## 4. But Fourier can approximate anything — doesn't that settle it?

The strongest form of the "why not sin?" objection runs like this: a Fourier series
can represent any reasonable function, just as a Taylor series can — and it is built
entirely out of sin and cos. So sin and cos are *the* universal building blocks. Why
not use them as activations?

It is a good argument, and following it carefully is instructive — because it turns
out to prove something slightly different from what it seems to.

**First, Fourier is not Taylor.** They are different bases with different domains of
validity. A Taylor series is *local*: it matches derivatives at a point, in the
monomial basis {1, x, x², …}, and converges only inside a radius set by the nearest
singularity — at a discontinuity the radius collapses to zero and the series cannot
even see the jump. A Fourier series is *global*: its coefficients are integrals over
the whole period, in the trigonometric basis, and it converges in L² for *any*
square-integrable function, discontinuities included. The analogy is loose, and the
difference matters.

**Second — and this is the twist — the Taylor half of the analogy is exactly why x²
fails.** Take the Taylor route to features and you get polynomials: monomials xⁿ. A
polynomial activation gives you precisely a truncated Taylor basis, and we showed in
section 1 that a single-hidden-layer x² network collapses to a quadratic form — a
Taylor expansion truncated at degree 2. So the Taylor analogy *diagnoses* the x²
failure. The Fourier analogy is the right one instead, because sin is not a
polynomial. Followed carefully, the analogy picks out the exact dividing line:
**polynomial basis → fails; non-polynomial basis → passes.**

**Third, "a complete basis" is not "a good activation."** These are two different
universality statements:

- *Representational universality.* The fixed family {1, sin(nx), cos(nx) : n ∈ ℕ} is
a complete orthonormal basis of L². But using it means summing *countably many fixed
terms* — for sharp features, very many of them.
- *Learnable universality.* The set {Σ vⱼ sin(wⱼx + bⱼ)} with *finitely many, learned*
frequencies is dense in C(K) — this is Leshno's theorem, and sin qualifies because it
is non-polynomial.

The completeness of the Fourier basis says nothing about activations. The activation's
job is to make a *finite, learned* network dense — a different theorem. And the reason
a network beats a fixed Fourier expansion is that it *learns which frequencies matter*:
it is an adaptive, sparse Fourier representation.

**Fourth — the part that actually matters for activations: spectral bias.** Here the
Fourier connection pays off. Networks trained by gradient descent fit their target
**from low frequencies to high** — a phenomenon called *spectral bias* (Rahaman et al.,
2019) or the *frequency principle* (Xu et al.). The mechanism is striking: **the
regularity of the activation function translates into the decay rate of the loss in
the frequency domain.** A smoother activation ⇒ a more strongly low-frequency-biased
network ⇒ high-frequency content is learned slowly, or not at all.

This reframes the trig question entirely. The sinusoid is not a curiosity — it is the
basis of the two main *cures* for spectral bias:

- **Fourier features as input encoding** (Tancik et al., 2020): mapping the input
through sin/cos at random frequencies turns the network's effective kernel into a
stationary one with a *tunable bandwidth*, letting it learn high frequencies.
- **sin as the activation** (SIREN): a periodic activation, with a deliberately
high-frequency first layer, represents high-frequency signals *and their derivatives*
far better than ReLU or tanh.

**Where this leaves sin.** It is, in fact, the best-conditioned of all our candidates.
It is defined on all of ℝ (unlike log x), bounded (unlike x² and e^x), smooth, with a
bounded derivative (|cos| ≤ 1) — and non-polynomial (unlike x²). It passes *every*
structural test. Its only weakness is the *inductive bias*: periodicity means aliasing
and infinitely many local minima. But that is a **soft** failure, and a *choosable*
prior — which is exactly why SIREN works: the right initialization turns the liability
into an advantage for periodic and high-frequency signals.

**One caveat, for honesty.** "Fourier can approximate anything" is qualified.
Convergence is in L², not uniform in general; at a jump discontinuity the partial sums
overshoot by about 9% forever (the Gibbs phenomenon); and some continuous functions
have Fourier series that diverge at a point. The universality is real but specific.

The net effect: the Fourier argument does not settle the question — but it *upgrades*
sin, from "a specialised prior that happens to work for periodic signals" to "the
candidate that passes every structural test and fails only on a soft, choosable
prior."

---

## 5. What the evidence, read together, suggests

1. **Nonlinearity is necessary but not sufficient.** An activation must also be
   non-polynomial (or otherwise escape the polynomial trap), numerically
   well-conditioned under depth, and carry the right inductive bias.
2. **x² fails on the sharpest test and the practical ones.** It is the exact class
   the universal approximation theorem excludes; it is lossy (sign-destroying),
   unbounded, and ill-conditioned.
3. **e^x shows non-polynomiality alone is not enough either.** It passes the theorem
   and is still unusable: unbounded, non-zero-centred, and with a gradient equal to
   its own value — exploding on one side, vanishing on the other. The field's
   response was to *tame* it (sigmoid, tanh, softplus, softmax, ELU), not to
   discard it.
4. **log x fails in a third way: it is not defined on half of ℝ.** A *domain*
   failure, not an expressivity or conditioning one — though patched (symlog,
   log(1+|x|)) it also inherits vanishing gradients. Log survives in machine
   learning wrapped: symlog, softplus = log(1+e^x), log-sigmoid, NLReLU.
5. **Monotonicity is a convenience, not a requirement.** The strongest modern
   activations are non-monotonic.
6. **tanh is a good general squash, not the only one.** It won for zero-centredness
   and a healthy gradient near zero, and it lost the deep-net default to ReLU.
7. **Trigonometric activations are a live tool, not a forbidden zone** — sin wins
   where the signal is periodic (SIREN), and the bias term is what makes any of it
   work.
8. **The Fourier argument upgrades sin rather than settling the question.** A complete
   Fourier basis is a *representational* universality statement, not a statement about
   activations; and the Taylor half of the analogy is exactly what condemns x². What
   the Fourier connection really delivers is *spectral bias* — the regularity of the
   activation sets the frequency decay of learning — which is why Fourier features and
   sin activations are the standard cures for it.

---

## 6. Open forks — your call

**F1. Scope.** Keep this to "why not x² / e^x / log x" plus "what makes an
activation," or open it into the full zoo (ReLU family, ELU, GELU, Swish, Mish,
parametric and learned activations)? *The zoo is where "why choose at all?" gets answered.*

**F2. Mathematical depth.** Keep the one-line quadratic-form proof, or develop the
Leshno theorem and the L²-density argument properly? *Deeper is more satisfying
but turns the artifact into a mini-course.*

**F3. Do we run an experiment?** Train a tiny MLP on 1-D regression — fit x³ (or
sin x) with x², tanh, ReLU, and sin as the activation, and watch x² fail to fit a
cubic while sin nails the sine. *Cheap, decisive, and it would make the point
unarguable.*

**F4. Learned activations as their own camp.** PReLU, the Pade Activation Unit,
learned polynomials, "learn your own nonlinearity" — this is the modern answer to
"why pick a fixed function at all." Treat as a separate axis?

**F5. Deliverable shape.** Same as Debate 001 (single essay + living evidence
table), or different?

**F6. Verdict policy.** Same as Debate 001 — does a debate ever get marked
*settled*?

---

## Sources (plain text)

- Leshno, Lin, Pinkus, Schocken. *Multilayer feedforward networks with a
  nonpolynomial activation function can approximate any function.* Neural Networks
  6(6), 1993.
- Hornik. *Approximation capabilities of multilayer feedforward networks.* Neural
  Networks, 1991.
- Sitzmann, Martel, Bergman, Lindell, Wetzstein. *Implicit Neural Representations
  with Periodic Activation Functions* (SIREN). NeurIPS 2020. arXiv:2006.09661.
- Garimella, Jha, Reagen. *Sisyphus: A Cautionary Tale of Using Low-Degree
  Polynomial Activations in Privacy-Preserving Deep Learning.* 2021. arXiv:2107.12342.
- Ismail et al. *Polynomial, trigonometric, and tropical activations.* 2025.
  arXiv:2502.01247.
- Hendrycks, Gimpel. *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415.
- Misra. *Mish: A Self Regularized Non-Monotonic Activation Function.* BMVC 2020.
  arXiv:1908.08681.
- Wu et al. *Learning a Single Neuron for Non-monotonic Activation Functions.*
  AISTATS 2022.
- LeCun, Bottou, Orr, Müller. *Efficient BackProp.* 1998.
- Clevert, Unterthiner, Hochreiter. *Fast and Accurate Deep Network Learning by
  Exponential Linear Units (ELUs).* ICLR 2016. arXiv:1511.07289.
- Rahman et al. *Natural-Logarithm-Rectified Activation Function in Convolutional
  Neural Networks* (NLReLU). 2019. arXiv:1908.03682.
- *A Comprehensive Survey of 400 Activation Functions.* 2024. (LAF / symlog,
  log-sigmoid, logish, NLReLU, symexp.)
- Rahaman et al. *On the Spectral Bias of Neural Networks.* ICML 2019.
- Xu et al. *Frequency Principle: Fourier Analysis Sheds Light on Deep Neural
  Networks.* arXiv:1901.06523.
- Tancik et al. *Fourier Features Let Networks Learn High Frequency Functions in Low
  Dimensional Domains.* NeurIPS 2020. arXiv:2006.10739.
- Chrysos, Moschoglou, Bouritsas, Deng, Panagakis. *Deep Polynomial Neural
  Networks.* IEEE TPAMI, 2022.
- Dubey, Singh, Chaudhuri. *Activation Functions in Deep Learning: A Comprehensive
  Survey and Benchmark.* arXiv:2109.14545.
