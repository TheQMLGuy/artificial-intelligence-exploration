# Debate 008 — What is deep learning?

**Status:** open — draft 0.1
**Opened:** 2026-10-28
**Editor:** Divyansh Tripathi
**Written by:** AI assistant
**Framing as asked:** "what is deep learning?"

---

## 0. Framing: the name describes an architecture, not a capability

The definition is short. Deep learning is machine learning in which the model is a composition of many learned transformations, so that the features are learned rather than designed by hand. A deep network is a stack, each layer computing a^(l) = f(W^(l) a^(l-1) + b^(l)), with every weight fitted by gradient descent on a loss and the gradients obtained by backpropagation.

That definition is easy and it is not where the argument is. The question that actually divides people is what is doing the work. Three things are bundled into the phrase and they can be separated. There is a **class of models**, meaning compositions of simple parametric layers. There is a **training regime**, meaning first-order optimisation of a model with far more parameters than data points, on a non-convex objective, using gradients that are only approximately informative. And there is an **empirical phenomenon**, which is that this combination works much better than the theory of the period predicted, on problems where nobody expected it to.

The live debate is between two explanations of the phenomenon. One says depth is the thing: composition buys representational efficiency that width cannot match, and that is why the models are structured this way. The other says scale is the thing: given enough data and compute, the architecture barely matters, and depth is one parameter among many rather than the cause. Both have real evidence, and the two are not usually stated in a way that makes them comparable, which is what this chapter tries to fix.

There is a third candidate that neither camp emphasises, which is optimisation. Some of the most surprising behaviour appears neither from depth nor from scale but from how long training runs and where it sits on its own curve, and any account that ignores that will misattribute the effect.

---

## 1. What the object is, precisely

The mathematical content of a deep network is composition. A linear map followed by a pointwise nonlinearity, repeated, gives a function whose pieces are simple and whose whole is not. The compositional structure is what allows a feature computed early to be reused by every later part of the network that needs it, and reuse is why a deep model can represent some functions with far fewer parameters than a shallow one.

Training is gradient descent on a loss, and the enabling fact is that the gradient of a composed function with respect to all of its parameters costs about one extra forward pass. Backpropagation is reverse-mode automatic differentiation, published in its modern form by Rumelhart, Hinton and Williams in 1986, and without that constant-factor cost none of this would be practical. The important thing to notice is that nothing in the loop knows what a feature is. Features are whatever the intermediate layers find useful for reducing the loss, which is the sense in which the representation is learned rather than designed.

What the classical theory does and does not give is worth stating, because it is routinely overread. Universal approximation, the Leshno result, guarantees that one sufficiently wide hidden layer can approximate any continuous function to any accuracy. It is a statement about existence. It says nothing about how many units that requires, nothing about whether the parameters are findable by gradient descent, and nothing about generalisation. Every strong claim for depth has to be made against that background, because a single wide layer already has the expressive power in principle.

---

## 2. The case that depth is the thing

The strongest form of the case is a separation theorem rather than an intuition. Telgarsky, in 2016, exhibited families of functions that a network of depth k computes with polynomially many units and that require exponentially many units to compute at depth k-1. Eldan and Shamir, in the same year, proved a comparable separation for a different function class. These are statements about representation, not about training, and they are the cleanest available evidence that depth is not merely a convenient way to add parameters.

There is a geometric version of the same argument. A rectified network partitions its input space into regions on which it is an affine function, and the number of such regions grows polynomially in width but exponentially in depth. So depth buys a qualitatively different kind of piecewise structure per parameter, which is exactly what a task with hierarchical composition seems to need.

The empirical version is the history. Between roughly 2006 and 2012 the field's central difficulty was that deep networks would not train, and the deep belief network chapter is the record of that struggle. When they did start training, the gains were large and they were specifically attributed to depth: adding layers helped in a way that adding width did not, on the same data and the same compute. And the architectural priors that dominate practice are priors about composition. A convolution is a statement that local structure composes; an attention layer is a statement that relations compose. Both are claims that the useful function is built in stages.

---

## 3. The case that scale is the thing

The opposing case has one slogan and a lot of measurement behind it. Sutton's bitter lesson, in 2019, is the claim that every attempt to build knowledge into a system has eventually been beaten by scaling a general method with more compute, and the history of the field since 2012 is largely consistent with it.

The measurements are scaling laws. Kaplan and colleagues, in 2020, found that test loss falls as a power law in model size, dataset size and compute, smooth enough across many orders of magnitude to be predictive. Hoffmann and colleagues, in 2022, corrected the allocation and showed that for a fixed compute budget the optimal model and the optimal number of training tokens should be scaled together, at roughly twenty tokens per parameter, which is why models got smaller and trained longer. The striking feature of these results for this debate is what is absent from them: the functional form is in total budget, and the architectural details move the constant rather than the exponent. If loss is a power law in scale, then depth is a hyperparameter inside a regime, not the engine of the regime.

There is a second line of evidence, about what overparameterisation is for. Classical statistics says a model with more parameters than data points should overfit, and these models do not. The double-descent curve, from Belkin and colleagues in 2019 and Nakkiran and colleagues in the same year, shows test error rising and then falling again as capacity increases past the interpolation threshold. The lottery-ticket result, from Frankle and Carbin, suggests that a dense network contains small sparse subnetworks that would have trained just as well, which makes capacity look like a search aid rather than a repository of knowledge. On that reading, the thing that matters is having enough room and enough data to find a good small solution, and depth is one way of providing room.

---

## 4. The third candidate: optimisation, and the time axis

Two observations fit neither camp and are worth keeping in view, because they are the ones that embarrass clean stories.

The first is grokking. Power and colleagues, in 2022, trained small networks on algorithmic tasks and found a regime where training accuracy reaches perfection and test accuracy stays at chance for a long time, and then, with no change to the data or the model, generalisation suddenly arrives. Nothing about depth or scale changed at that moment. What changed was how far along its optimisation trajectory the model had travelled. Any account of deep learning that treats generalisation as a property of the architecture has to explain why it can be absent in a perfectly capable model and then appear.

The second is the contested status of emergence. Wei and colleagues, in 2022, reported abilities that appear sharply once models pass a certain scale, which was read as evidence for scale as the driver. Schaeffer, Miranda and Koyejo, in 2023, argued that many of those curves are artefacts of the metrics rather than of the models, since a discontinuous-looking jump in an all-or-nothing accuracy score can be a smooth improvement in a continuous underlying quantity. That dispute is not settled, and it matters here, because the sharpest evidence for the scale story rests on the sharpness of those curves.

---

## 5. What the word is doing, and what is still open

The honest state of the field is that there is an engineering discipline with very good results and a thin theory, and that the word deep is a statement about composition rather than about understanding. It describes a circuit's shape, not its comprehension.

What remains genuinely open is why the training regime generalises at all. There is no accepted theory that explains fitting far more parameters than data points while still generalising, that explains why first-order methods find good solutions in a non-convex landscape, or that predicts which architectural choices will matter at which scale. There are partial accounts, each covering part of the picture, and the field proceeds mostly by measurement.

There is also a live question about whether depth will remain the defining word. Mixture-of-experts models make total parameters a poor description of compute. Retrieval and tool use put capability outside the weights. Spending more compute at inference, rather than in the stack, is now a primary axis of progress. In each case the thing that improves is not the depth of the network, and if that continues then deep learning will name a historical period rather than a mechanism.

---

## 6. What the evidence, read together, suggests

This is a reading, not a verdict. The editor decides what stands.

The definition is not in dispute: composition of learned transformations, trained by gradient descent with backpropagated gradients. What is in dispute is the causal story, and the evidence supports a split verdict that neither camp will enjoy. Depth demonstrably changes what a fixed number of parameters can represent, and the separation theorems are proofs rather than analogies, so the strong scale-only reading is wrong. But the predictive laws that actually guide practice are laws about total budget, and the architectural details enter as constants, so the strong depth-first reading overstates its case. What is missing from both is the optimisation trajectory, which is where grokking lives and where the sharpest anomalies are.

The most defensible statement is that deep learning is a regime rather than a mechanism: enough depth to make composition cheap, enough capacity to make the search easy, and enough data and compute to make the power laws bite. The word names the shape of the thing that happened to work, and the explanation for why it worked is still being written.

---

## 7. Open forks — your call

**F1. Is the chapter a definition, a history, or a causal argument?** The definition takes a paragraph, and the argument is the chapter.

**F2. How much mathematics?** Keep the layer equation and the separation results as statements, or develop the linear-region count and the scaling-law fits properly?

**F3. Do we run an experiment?** A small model trained to grokking on a modular-arithmetic task is the cheapest way to see generalisation arrive after fitting, and it would put the strongest anomaly in the chapter on the page rather than in a citation.

**F4. Do the scaling laws get their own chapter?** They are the most quantitative thing in the field, they are contested in their extrapolation, and they deserve more than a section.

**F5. How much of the theory gap?** The generalisation puzzle, the loss-landscape work and the double-descent literature could each double the length.

**F6. Verdict policy.** Does a chapter whose subject has no accepted theory ever get marked settled?

**F7. Deliverable shape.** Same as before?

---

## Cross-debate notes

Debate 005 is the prehistory of this chapter. The deep belief network existed because deep networks would not train, and it was displaced not by an argument but by rectified units, dropout, initialisation and data, which is the same claim the scale camp makes now.

Debate 002 supplies the fact that keeps the depth case honest. Universal approximation already gives one wide layer the power in principle, so every argument for depth has to be an argument about cost or findability rather than about expressiveness in the abstract.

Debate 007 is where the input representation lives, and the same tension appears there: the distributional prior is the idea, and the better estimator is what changed.

Debate 004 is the cost side of the same object. Once a model is deep and contextual, inference becomes a memory-bandwidth problem, which is why the systems literature exists at all.

---

## Sources (plain text)

Rumelhart, Hinton, Williams. *Learning representations by back-propagating errors.* Nature, 1986.

LeCun et al. *Backpropagation applied to handwritten zip code recognition.* Neural Computation, 1989.

Leshno, Lin, Pinkus, Schocken. *Multilayer feedforward networks with a nonpolynomial activation function can approximate any function.* Neural Networks, 1993.

Krizhevsky, Sutskever, Hinton. *ImageNet classification with deep convolutional neural networks.* NIPS 2012.

Montufar, Pascanu, Cho, Bengio. *On the number of linear regions of deep neural networks.* NIPS 2014.

Telgarsky. *Benefits of depth in neural networks.* COLT 2016. arXiv:1602.04485.

Eldan, Shamir. *The power of depth for feedforward neural networks.* COLT 2016. arXiv:1512.03965.

Sutton. *The bitter lesson.* 2019.

Kaplan et al. *Scaling laws for neural language models.* 2020. arXiv:2001.08361.

Hoffmann et al. *Training compute-optimal large language models.* 2022. arXiv:2203.15556.

Belkin, Hsu, Ma, Mandal. *Reconciling modern machine learning practice and the classical bias-variance trade-off.* PNAS, 2019.

Nakkiran et al. *Deep double descent.* ICLR 2020. arXiv:1912.02292.

Frankle, Carbin. *The lottery ticket hypothesis.* ICLR 2019. arXiv:1803.03635.

Power, Burda, Edwards, Misra, Weinberger. *Grokking: generalization beyond overfitting on small algorithmic datasets.* 2022. arXiv:2201.02177.

Wei et al. *Emergent abilities of large language models.* TMLR 2022. arXiv:2206.07682.

Schaeffer, Miranda, Koyejo. *Are emergent abilities of large language models a mirage?* NeurIPS 2023. arXiv:2304.15004.

---

## Further down the rabbit hole

Everything below is real, and grouped so you can go as deep as you want.

**The foundations**
- Rumelhart, Hinton, Williams. *Learning representations by back-propagating errors.* Nature 323, 1986.
- LeCun et al. *Gradient-based learning applied to document recognition.* IEEE, 1998.
- Leshno et al. *Multilayer feedforward networks with a nonpolynomial activation.* Neural Networks, 1993.

**The depth case**
- Telgarsky. *Benefits of depth in neural networks.* COLT 2016. arXiv:1602.04485.
- Eldan, Shamir. *The power of depth for feedforward neural networks.* COLT 2016. arXiv:1512.03965.
- Montufar et al. *On the number of linear regions of deep neural networks.* NIPS 2014.
- Raghu et al. *On the expressive power of deep neural networks.* ICML 2017. arXiv:1606.05336.

**The scale case**
- Kaplan et al. *Scaling laws for neural language models.* 2020. arXiv:2001.08361.
- Hoffmann et al. *Training compute-optimal large language models.* 2022. arXiv:2203.15556.
- Sutton. *The bitter lesson.* 2019.
- Belkin et al. *Reconciling modern machine learning practice.* PNAS, 2019.
- Nakkiran et al. *Deep double descent.* ICLR 2020. arXiv:1912.02292.
- Frankle, Carbin. *The lottery ticket hypothesis.* ICLR 2019. arXiv:1803.03635.

**Optimisation and the time axis**
- Power et al. *Grokking.* 2022. arXiv:2201.02177.
- Wei et al. *Emergent abilities.* TMLR 2022. arXiv:2206.07682.
- Schaeffer, Miranda, Koyejo. *Are emergent abilities a mirage?* NeurIPS 2023. arXiv:2304.15004.
- Nanda et al. *Progress measures for grokking via mechanistic interpretability.* ICLR 2023. arXiv:2301.05217.

**The theory gap**
- Zhang et al. *Understanding deep learning requires rethinking generalization.* ICLR 2017. arXiv:1611.03530.
- Neyshabur et al. *Exploring generalization in deep learning.* NIPS 2017. arXiv:1706.08947.
- Bahri et al. *Explaining neural scaling laws.* PNAS, 2024. arXiv:2102.06701.

**Code**
- Any small implementation of a multilayer perceptron in PyTorch, plus the `modular addition` grokking setup from the Nanda et al. replication, which is the cheapest entry point to the anomaly.
