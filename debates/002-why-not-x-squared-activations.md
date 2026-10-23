# Debate 002 — Why not x², e^x — or log x? Is nonlinearity all we need, and why tanh?

**Status:** open — draft 0.5
**Opened:** 2026-10-22
**Last edited:** 2026-10-23
**Editor:** Divyansh Tripathi
**Written by:** AI assistant
**Framing as asked:** "Why don't we use y = x² as an activation function? Even though it's a nonlinear function — is nonlinearity everything we need to make a function an activation function, or are we missing something? And why only tanh? Why not other trigonometric ones?"
**Follow-up:** "...and why not e^x, since it's also nonlinear and non-polynomial?" ... "What about log x?"

---

## 0. Framing: the question carries three assumptions

The question bundles three claims together, and all three are false or only half true. Naming them is most of the work.

The first claim is that a function is an activation function if and only if it is nonlinear. Nonlinearity is necessary but nowhere near sufficient, and much of what follows is an unpacking of what else is required. The second is that nonlinearity is the whole story, which it is not, since the shape of the nonlinearity and its interaction with optimisation matter at least as much. The third is that tanh is the only sensible choice among the smooth bounded curves and that the trigonometric functions are unexplored territory. Both halves of that are false: there is a whole zoo of activations, and trigonometric activations are a real, published tool, just a specialised one.

So the real question is what an activation function actually has to do, and why x², e^x and log x all fail that test while tanh passes it. It is worth saying at the outset that the answer is not "because they are nonlinear," since two of the three are perfectly nonlinear.

---

## 1. Why x², e^x and log x are not just other nonlinearities

### The case the theorem excludes

There is a clean and famous result here. Leshno, Lin, Pinkus and Schocken showed in 1993 that a single-hidden-layer feedforward network is a universal approximator if and only if its activation function is not a polynomial. Polynomial activations are the single class that the universal approximation theorem rules out, and x² belongs to it.

The reason is easy to see directly. Put σ(t) = t² into the usual sum y = Σ vⱼ σ(wⱼ·x + bⱼ) and expand it. Each term becomes a quadratic form, and the whole sum reduces to xᵀAx + cᵀx + d, which is a single quadratic polynomial. However wide you make the hidden layer, a one-layer x² network is still just a quadratic model. It cannot represent a cubic, and adding units does not change that.

Depth does not rescue it cleanly either. Composing L quadratic layers produces a polynomial of degree 2^L, which is technically universal because polynomials are dense, but the values grow like |x| raised to 2^L and the gradients become unusable. There is a documented version of this failure: the Sisyphus paper, which exists precisely because practitioners wanted polynomial activations to work for privacy-preserving inference and found that they stop behaving after a few layers. So x² is doubly bad. Shallow, it is too weak to represent what you need; deep, it is numerically unstable.

There are two more problems that are less discussed but just as real. x² is not monotonic and not injective, so x and −x map to the same output and the sign of the input is lost. It is also unbounded above, and its derivative, 2x, is unbounded too, so the gradient depends heavily on the magnitude of the activation, which makes training ill-conditioned.

None of this makes x² illegal. Polynomial activations are used where the objective is different, such as privacy-preserving inference, where replacing ReLU with x² makes the operation a simple multiply under homomorphic encryption and cuts latency considerably. Recent work in 2025 also trains GPT-2 and ConvNeXt with polynomial and Hermite activations, though only with a carefully designed variance-preserving initialisation. The accurate statement is that x² is not illegal but is a poor general-purpose prior, needing special handling and buying less expressivity per layer.

### The mirror case: why not e^x

The follow-up question about e^x is sharper than it looks, because e^x passes the very test that x² fails. It is nonlinear, and it is emphatically non-polynomial, so the condition in the 1993 theorem is satisfied and an e^x network is in principle a universal approximator. On expressivity, e^x is fine. It also has several properties that x² lacks: it is monotonic, injective, smooth and convex. If nonlinearity, or non-polynomiality, or monotonicity were the whole story, e^x would be a perfectly good activation.

It fails on the other axis, which is conditioning, and it fails harder than x² does. It is unbounded above, so composing layers gives an exponential tower, e^(W₂·e^(W₁·x)), in which each layer exponentiates the last and values overflow to infinity after a handful of layers. The gradient is the activation itself, since the derivative of e^x is e^x, so wherever the activation is large the gradient is equally large and gradients explode by construction. On the other side, as x goes to negative infinity e^x tends to zero and so does its gradient, which means e^x has a vanishing gradient for negative inputs and an exploding one for positive inputs, both classic pathologies in a single function. It is also not zero-centred, since its range is (0, ∞) and it is never negative, which is the same all-positive gradient problem as the logistic sigmoid; indeed the sigmoid is 1/(1 + e^−x), so e^x is the sigmoid's raw ingredient. One curiosity is that the bias does nothing: because e^(w·x + b) equals e^b times e^(w·x), the threshold term is absorbed into a positive scaling, so the bias that the 1993 theorem needs for functions like sin is redundant here.

What the field did with this was not to discard the exponential but to tame it. Almost every bounded activation is e^x in a wrapper. Sigmoid is 1/(1 + e^−x), tanh is (e^x − e^−x)/(e^x + e^−x), softplus is log(1 + e^x), softmax normalises it, and attention uses a normalised exponential as its kernel. The most direct answer is the exponential linear unit, or ELU, proposed by Clevert, Unterthiner and Hochreiter in 2015, which keeps the exponential only on the negative side where it saturates gracefully and uses the identity on the positive side where it would explode. ELU is, quite literally, e^x with the dangerous half removed.

### The third case: why not log x

Log x fails in a way that neither of the others does: it is not defined on half the real line. An activation function has to be a map from the reals to the reals, accepting every real pre-activation whether positive, zero or negative. Log x is undefined for x ≤ 0, so any neuron with a non-positive input produces NaN. This is a domain failure, and it is fatal before any question of conditioning arises.

You can patch the domain, but then you inherit a conditioning problem. The usual fix, log(1 + x), is defined for x > −1, but its derivative is 1/(1 + x), which tends to zero as x grows, so you get vanishing gradients for large inputs and the function saturates one-sidedly, like a tanh that only flattens on the right. The symmetric version, log(1 + |x|), is defined on all of the reals but is even, since log|−x| equals log|x|, so it destroys the sign, which is the same defect as x²; its derivative, 1/(1 + |x|), tends to zero at both ends. The bare function, near zero, has a derivative of 1/x that tends to infinity as x approaches zero from above, so unpatched it has an exploding gradient at the boundary and a vanishing one at infinity. It is also unbounded, so it fails the boundedness condition of the original Cybenko theorem, although it does satisfy Leshno's non-polynomial condition, which means expressivity is again not the problem.

There is a symmetry worth naming here. Log x and e^x are inverses, mirror images across the line y = x. Where e^x explodes as x goes to positive infinity, log x explodes as x approaches zero from above; where e^x vanishes as x goes to negative infinity, the gradient of log x vanishes as x goes to positive infinity. They are the same failure, reflected, and they are tamed by the same move of bounding the domain and the range.

The field tamed log as well rather than discarding it, and log is everywhere in machine learning, though always wrapped. There is symlog, also called the logarithmic activation function, which is the symmetric version and is used in practice to compress huge dynamic ranges; it is the reward and value transform in Dreamer-style reinforcement learning agents. There is softplus, log(1 + e^x), which is the most famous log-based activation and is itself a composition of the two functions we have been discussing. There is log-sigmoid, log σ(x), which is used as an activation, and there is NLReLU, log(1 + max(0, x)), which applies log to the positive side of ReLU only, keeping ReLU's sparsity while softening its growth.

The three candidates together form a small taxonomy of why an activation fails, and none of the reasons is that it is nonlinear. x² fails on expressivity, because it is a polynomial. e^x fails on conditioning, because it is unbounded and explosive. log x fails on domain, because it is undefined for x ≤ 0. That table is the answer to the original question, stated exactly: nonlinearity is the entry ticket, not the qualification.

---

## 2. So what does an activation function actually have to do?

Nonlinearity is necessary and insufficient. The properties that actually matter, and how the usual suspects score on them, are set out below.

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

Three corrections to the "nonlinearity is everything" theory fall out of this table. The first is that non-polynomiality is necessary but not sufficient, since it is the exact thing x² lacks but e^x has, and e^x is still unusable. The second is that monotonicity is not required, which is the part most people get wrong: GELU, Swish and Mish are all non-monotonic, dipping slightly negative near zero, and they are state of the art, and recent theory even establishes learnability for non-monotonic activations. The folk rule that activations must be monotonic is obsolete. The third is that conditioning beats expressivity in practice, because a shallow polynomial network is too weak and a deep one is too unstable, and the activation that wins is the one whose derivative keeps gradients in a sane range through many layers.

---

## 3. Why tanh — and why not (only) the other trigonometric functions?

Tanh won historically for reasons that are easy to state. It is a squashed, bounded, monotonic and smooth curve with range (−1, 1), so it is numerically safe. It is zero-centred, unlike the sigmoid, which avoids the gradient zig-zag that comes from all-positive activations. Its derivative is up to 1 at the origin, whereas the sigmoid's maximum derivative is only 0.25, so the signal flows more strongly. It is a rescaled logistic, since tanh(x) equals 2σ(2x) − 1, and shares the sigmoidal universal-approximation guarantee. In an LSTM it pairs naturally with the logistic gates, squashing the cell state while the gates are gated with sigmoids.

It also has real weaknesses, which is why it is no longer the default. It saturates, so gradients vanish at the tails; it is expensive because of the exponentiation; and ReLU displaced it in deep feedforward networks for exactly those reasons.

On the trigonometric question, the point is not that trigonometry is forbidden but that most trigonometric functions violate the properties above. Sine and cosine are periodic, so they map far-apart inputs to the same value, which is aliasing; they have infinitely many local minima and they oscillate rather than monotonise. For a general function approximator on an unbounded domain, periodicity is simply the wrong prior. There is a nice subtlety from the 1993 theorem as well: the family {sin(w·x)} contains only odd functions, so without a threshold it cannot even represent cos, which is why the bias term is essential.

Sine is nevertheless used where periodicity is the right prior. SIREN, published by Sitzmann and colleagues in 2020, uses sine as the activation for implicit neural representations, and represents images, audio, wavefields and 3-D shapes, along with their derivatives, far better than ReLU or tanh. It has an elegant property, in that the derivative of a SIREN is itself a SIREN, since the derivative of sine is cosine, which is a phase-shifted sine. It needs a carefully designed initialisation, with ω₀ set to 30 for the first layer, and that is the price of the periodic prior. Work in 2025 also trains GPT-2 and ConvNeXt with a trigonometric basis.

So the honest answer to the question about other trigonometric functions is that sine is not unused but specialised. Tanh is a general-purpose squash built from exponentials, whereas sine is a periodic prior that wins when the data is periodic. Choosing an activation is choosing an inductive bias, not searching for any nonlinear function.

---

## 4. But Fourier can approximate anything — doesn't that settle it?

The strongest form of the argument for sine runs like this. A Fourier series can represent any reasonable function, just as a Taylor series can, and it is built entirely out of sine and cosine. So sine and cosine are the universal building blocks. Why not use them as activations?

It is a good argument, and following it carefully is instructive, because it turns out to prove something slightly different from what it appears to.

First, Fourier is not Taylor. They are different bases with different domains of validity. A Taylor series is local: it matches derivatives at a point, in the monomial basis {1, x, x², …}, and converges only inside a radius set by the nearest singularity, so at a discontinuity the radius collapses to zero and the series cannot even see the jump. A Fourier series is global: its coefficients are integrals over the whole period, in the trigonometric basis, and it converges in L² for any square-integrable function, discontinuities included. The analogy is loose, and the difference matters.

Second, and this is the twist, the Taylor half of the analogy is exactly why x² fails. Take the Taylor route to features and you get polynomials, the monomials xⁿ. A polynomial activation gives you precisely a truncated Taylor basis, and we showed in section 1 that a single-hidden-layer x² network collapses to a quadratic form, which is a Taylor expansion truncated at degree two. The Taylor analogy diagnoses the x² failure. The Fourier analogy is the right one instead, because sine is not a polynomial, and followed carefully the analogy picks out the exact dividing line: a polynomial basis fails, a non-polynomial basis passes.

Third, a complete basis is not the same as a good activation, because these are two different universality statements. Representational universality concerns the fixed family {1, sin(nx), cos(nx)} for integer n, which is a complete orthonormal basis of L², but using it means summing countably many fixed terms, and for sharp features very many of them. Learnable universality concerns the set of finite sums Σ vⱼ sin(wⱼx + bⱼ) with learned frequencies, which is dense in the space of continuous functions by Leshno's theorem, and sine qualifies because it is non-polynomial. The completeness of the Fourier basis says nothing about activations, since the activation's job is to make a finite, learned network dense, which is a different theorem. The reason a network beats a fixed Fourier expansion is that it learns which frequencies matter, which makes it an adaptive and sparse Fourier representation.

Fourth, and this is the part that actually matters for activations, there is spectral bias. Networks trained by gradient descent fit their target from low frequencies to high, a phenomenon called spectral bias by Rahaman and colleagues in 2019 and the frequency principle by Xu and colleagues. The mechanism is striking: the regularity of the activation function translates into the decay rate of the loss in the frequency domain, so a smoother activation gives a more strongly low-frequency-biased network and high-frequency content is learned slowly or not at all. This reframes the trigonometric question entirely, because the sinusoid is not a curiosity but the basis of the two main cures for spectral bias. One is Fourier features as an input encoding, from Tancik and colleagues in 2020, where mapping the input through sine and cosine at random frequencies turns the network's effective kernel into a stationary one with a tunable bandwidth, letting it learn high frequencies. The other is sine as the activation, in SIREN, which with a deliberately high-frequency first layer represents high-frequency signals and their derivatives far better than ReLU or tanh.

Where this leaves sine is better than where the earlier discussion put it. It is, in fact, the best-conditioned of all the candidates. It is defined on all of the reals, unlike log x; it is bounded, unlike x² and e^x; it is smooth, with a bounded derivative; and it is non-polynomial, unlike x². It passes every structural test. Its only weakness is the inductive bias, since periodicity means aliasing and infinitely many local minima, and that is a soft failure and a choosable prior, which is exactly why SIREN works.

One caveat, for honesty: "Fourier can approximate anything" is qualified. Convergence is in L², not uniform in general; at a jump discontinuity the partial sums overshoot by about 9 per cent forever, which is the Gibbs phenomenon; and some continuous functions have Fourier series that diverge at a point. The universality is real but specific.

The net effect is that the Fourier argument does not settle the question, but it upgrades sine, moving it from a specialised prior that happens to work for periodic signals to the candidate that passes every structural test and fails only on a soft, choosable prior.

---

## 5. What the evidence, read together, suggests

This is a reading, not a verdict. The editor decides what stands.

Nonlinearity is necessary but not sufficient, and an activation must also be non-polynomial, or otherwise escape the polynomial trap, numerically well-conditioned under depth, and carry the right inductive bias. x² fails on the sharpest test and on the practical ones, being the exact class the universal approximation theorem excludes, lossy in that it destroys the sign, unbounded and ill-conditioned. e^x shows that non-polynomiality alone is not enough either, since it passes the theorem and is still unusable, and the field's response was to tame it rather than discard it. log x fails in a third way, by not being defined on half the reals, and it too survives in machine learning only in wrapped form. Monotonicity is a convenience rather than a requirement, since the strongest modern activations are non-monotonic. tanh is a good general squash but not the only one, and it lost the deep-network default to ReLU. Trigonometric activations are a live tool rather than a forbidden zone, with sine winning where the signal is periodic and the bias term making any of it work. The Fourier argument, finally, upgrades sine rather than settling the question, since a complete Fourier basis is a representational universality statement and not a statement about activations, and what the Fourier connection really delivers is spectral bias, the regularity of the activation setting the frequency decay of learning.

---

## 6. Open forks — your call

**F1. Scope.** Keep this to "why not x², e^x or log x" plus "what makes an activation," or open it into the full zoo of the ReLU family, ELU, GELU, Swish, Mish and the parametric and learned activations? The zoo is where the question of why to choose at all gets answered.

**F2. Mathematical depth.** Keep the one-line quadratic-form proof, or develop the Leshno theorem and the L²-density argument properly? Deeper is more satisfying but turns the artifact into a mini-course.

**F3. Do we run an experiment?** Train a tiny MLP on 1-D regression, fitting x³ or sin x with x², tanh, ReLU and sin as the activation, and watch x² fail to fit a cubic while sin nails the sine. It would be cheap and decisive, and would make the point unarguable.

**F4. Learned activations as their own camp.** PReLU, the Padé Activation Unit and learned polynomials are the modern answer to why pick a fixed function at all. Treat as a separate axis?

**F5. Deliverable shape.** Same as before?

**F6. Verdict policy.** Same: does a debate ever get marked settled?

---

## Sources (plain text)

- Leshno, Lin, Pinkus, Schocken. *Multilayer feedforward networks with a nonpolynomial activation function can approximate any function.* Neural Networks 6(6), 1993.
- Hornik. *Approximation capabilities of multilayer feedforward networks.* 1991.
- Cybenko. *Approximation by Superpositions of a Sigmoidal Function.* 1989.
- Sitzmann, Martel, Bergman, Lindell, Wetzstein. *Implicit Neural Representations with Periodic Activation Functions* (SIREN). NeurIPS 2020. arXiv:2006.09661.
- Garimella, Jha, Reagen. *Sisyphus: A Cautionary Tale of Using Low-Degree Polynomial Activations in Privacy-Preserving Deep Learning.* 2021. arXiv:2107.12342.
- Ismail et al. *Polynomial, trigonometric, and tropical activations.* 2025. arXiv:2502.01247.
- Hendrycks, Gimpel. *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415.
- Misra. *Mish: A Self Regularized Non-Monotonic Activation Function.* BMVC 2020. arXiv:1908.08681.
- Wu et al. *Learning a Single Neuron for Non-monotonic Activation Functions.* AISTATS 2022.
- LeCun, Bottou, Orr, Müller. *Efficient BackProp.* 1998.
- Clevert, Unterthiner, Hochreiter. *Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs).* ICLR 2016. arXiv:1511.07289.
- Rahman et al. *Natural-Logarithm-Rectified Activation Function in Convolutional Neural Networks* (NLReLU). 2019. arXiv:1908.03682.
- *A Comprehensive Survey of 400 Activation Functions.* 2024.
- Rahaman et al. *On the Spectral Bias of Neural Networks.* ICML 2019.
- Xu et al. *Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks.* arXiv:1901.06523.
- Tancik et al. *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.* NeurIPS 2020. arXiv:2006.10739.
- Chrysos et al. *Deep Polynomial Neural Networks.* IEEE TPAMI, 2022.
- Dubey, Singh, Chaudhuri. *Activation Functions in Deep Learning: A Comprehensive Survey and Benchmark.* arXiv:2109.14545.

---

## Further down the rabbit hole

Everything below is real, and grouped so you can go as deep as you want.

**Foundations of universal approximation**
- Cybenko. *Approximation by Superpositions of a Sigmoidal Function.* 1989.
- Hornik. *Approximation Capabilities of Multilayer Feedforward Networks.* 1991.
- Leshno, Lin, Pinkus, Schocken. *Multilayer Feedforward Networks with a Nonpolynomial Activation Function Can Approximate Any Function.* Neural Networks, 1993.
- Barron. *Universal Approximation Bounds for Superpositions of a Sigmoidal Function.* 1993.
- Pinkus. *Approximation Theory of the MLP Model in Neural Networks.* Acta Numerica, 1999.

**Classic activations and their history**
- LeCun, Bottou, Orr, Müller. *Efficient BackProp.* 1998.
- Glorot & Bengio. *Understanding the Difficulty of Training Deep Feedforward Neural Networks.* AISTATS 2010.
- Nair & Hinton. *Rectified Linear Units Improve Restricted Boltzmann Machines.* ICML 2010.
- Maas, Hannun, Ng. *Rectifier Nonlinearities Improve Neural Network Acoustic Models.* ICML 2013. (Leaky ReLU.)
- He et al. *Delving Deep into Rectifiers (PReLU).* ICCV 2015. arXiv:1502.01852.
- Clevert, Unterthiner, Hochreiter. *ELUs.* ICLR 2016. arXiv:1511.07289.
- Klambauer et al. *Self-Normalizing Neural Networks (SELU).* NeurIPS 2017. arXiv:1706.02515.
- Hendrycks & Gimpel. *Gaussian Error Linear Units (GELUs).* 2016. arXiv:1606.08415.
- Ramachandran, Zoph, Le. *Searching for Activation Functions (Swish).* 2017. arXiv:1710.05941.
- Misra. *Mish.* BMVC 2020. arXiv:1908.08681.

**Polynomial activations, and why they persist**
- Garimella, Jha, Reagen. *Sisyphus: Low-Degree Polynomial Activations in Privacy-Preserving Deep Learning.* 2021. arXiv:2107.12342.
- Chrysos et al. *Deep Polynomial Neural Networks.* IEEE TPAMI, 2022.
- Ismail et al. *Polynomial, trigonometric, and tropical activations.* 2025. arXiv:2502.01247.
- Rahman et al. *NLReLU.* 2019. arXiv:1908.03682.

**Periodic activations, Fourier features, and spectral bias**
- Sitzmann et al. *SIREN: Implicit Neural Representations with Periodic Activation Functions.* NeurIPS 2020. arXiv:2006.09661.
- Tancik et al. *Fourier Features Let Networks Learn High Frequency Functions.* NeurIPS 2020. arXiv:2006.10739.
- Rahimi & Recht. *Random Features for Large-Scale Kernel Machines.* NeurIPS 2007.
- Rahaman et al. *On the Spectral Bias of Neural Networks.* ICML 2019. arXiv:1806.08734.
- Xu et al. *Frequency Principle: Fourier Analysis Sheds Light on Deep Neural Networks.* 2019. arXiv:1901.06523.
- Mildenhall et al. *NeRF.* ECCV 2020. (The positional encoding that made Fourier features famous.)

**Learned and adaptive activations**
- Molina, Schramowski, Kersting. *Padé Activation Units.* 2019. arXiv:1901.03892.
- Agostinelli et al. *Learning Activation Functions to Improve Deep Neural Networks.* 2014.
- Apicella et al. *A Survey on Modern Trainable Activation Functions.* 2020. arXiv:2005.00817.

**Surveys**
- Dubey, Singh, Chaudhuri. *Activation Functions in Deep Learning: A Comprehensive Survey and Benchmark.* 2021. arXiv:2109.14545.
- Apicella et al. *Activation Functions in Artificial Neural Networks: A Systematic Overview.* 2021. arXiv:2101.09957.
- Eger, Youssef, Gurevych. *Is it Time to Swish?* EMNLP 2020. arXiv:1901.02671.
- *A Comprehensive Survey of 400 Activation Functions.* 2024.
- Nwankpa et al. *Activation Functions: Comparison of Trends in Practice and Research.* 2018.
