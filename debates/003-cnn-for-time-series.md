# Debate 003 — Can a CNN do time series? Why, and how well?

**Status:** open — draft 0.2
**Opened:** 2026-10-22
**Editor:** Divyansh Tripathi
**Written by:** AI assistant
**Framing as asked:** "Can CNN be used for time series problems? Why, and how does it perform?"

---

## 0. Framing: "CNN" is doing a lot of work in the question

The question carries a hidden assumption, which is that "CNN" means the two-dimensional image convolution and that if it can do time series at all it is doing so as a compromise. Neither holds.

The honest answer has three parts, and they are worth separating. First, can it? Yes, but not the vision CNN; the CNN that works on sequences is a specific architecture, the temporal convolutional network or TCN, and it only works after two particular modifications. Second, why does convolution suit a sequence at all? Because it encodes exactly two priors that time series often obey, which are locality and translation invariance. Third, how well? It is competitive to state of the art across every mainstream time-series task, but, as in Debate 001, the margin over Transformers and MLPs is small and sensitive to the evaluation protocol.

---

## 1. Can a CNN do time series? Yes, but a specific CNN

The naive idea is to slide one-dimensional filters along the time axis, exactly as a vision CNN slides two-dimensional filters over an image. That much is true and old; the time delay neural network did it in 1989. But a vanilla sliding convolution has a fatal property, which is that its receptive field grows only linearly with depth. Stack L layers of kernel size k and the network sees 1 + L(k − 1) steps back, so to look a thousand steps into the past you would need hundreds of layers.

The fix is two modifications. The first is causality: an output at time t may be convolved only with inputs from time t and earlier, so that no information leaks from the future into the past. This is what makes the convolution causal, and it is essential for forecasting. The second is dilation: spread the filter taps out by a factor d, so that a kernel of size k reaches across (k − 1)d steps. Increase the dilation exponentially with depth, so that d runs 1, 2, 4, 8 and so on, and the receptive field grows exponentially with depth:

    R = 1 + (k − 1)(2^L − 1)

With k = 3 and L = 10 layers, that is a receptive field of about 2000 steps, from ten layers. Add residual connections to stabilise the depth and you have the TCN, which is a one-dimensional fully-convolutional network with causal convolutions, dilated convolutions and residual blocks. That is the whole trick; the rest of the architecture is ordinary.

---

## 2. Why does convolution suit a time series at all?

There are four mechanisms, and they are the actual answer to the question of why. The first is local pattern detectors with weight sharing. A convolution filter is a learned template, and the same filter is applied at every position along the series, so a pattern such as a spike, a motif or a regime is detected wherever it appears. This is exactly the structure of time-series motifs, which are locally-shaped, recurring subsequences. The second is that the receptive field is a design choice rather than a learned accident. In an RNN, how much history matters is implicit in the hidden state; in a TCN it is a hyperparameter you set with k, d and depth, so you can reason about what the model can and cannot see. The third is parallelism and short gradient paths. There is no sequential dependency across time, so a whole series trains in parallel, and backpropagation runs along the depth rather than along time, so the classic RNN pathologies of vanishing and exploding gradients through time largely disappear. The fourth is that cross-channel mixing comes for free, since a convolution can also run across the variable dimension and capture dependencies between channels, which a per-variable RNN does not do naturally.

Put together, convolution is not a capability but a prior, meaning locality plus translation invariance. It wins when that prior is true of your data.

---

## 3. How well does it perform?

### Sequence modelling in general, the origin

Bai, Kolter and Koltun published the paper that made people take TCNs seriously, in 2018, and it is worth noting what they found. A single generic TCN beat canonical LSTMs and GRUs across a broad suite of tasks that had been designed to benchmark recurrent networks, which is to say on the recurrent networks' home turf.

| Task | LSTM | GRU | TCN |
|---|---|---|---|
| Sequential MNIST (accuracy) | 87.2 | 96.2 | **99.0** |
| Permuted MNIST (accuracy) | 85.7 | 87.3 | **97.2** |
| Copy memory (loss) | 0.0204 | 0.0197 | **3.5e-5** |

Their sharpest finding was not accuracy but memory. Despite the RNN's theoretical ability to remember an unbounded history, the TCN exhibited substantially longer effective memory in practice.

### Forecasting

TimesNet, published at ICLR in 2023, attacks the one-dimensional representation problem directly. It uses an FFT to discover the dominant periods in the series, reshapes the one-dimensional series into a stack of two-dimensional tensors with rows as periods and columns as within-period position, and applies two-dimensional convolutions to capture both intra-period and inter-period variation. It reports consistent state of the art across five tasks.

ModernTCN, published at ICLR in 2024, takes the other route and keeps the series one-dimensional while modernising the convolution, with large kernels of size 51, patchify embedding, and a decoupled block that mixes time, feature and variable dimensions separately. It reports state of the art across five tasks while keeping convolution's efficiency advantage, and its central claim is about the effective receptive field: enlarging the kernel raises the effective receptive field more efficiently than stacking layers, since the field grows roughly linearly in kernel size but only sub-linearly in depth.

The honest caveat is a reproducibility study from 2025, ModernTCN Revisited, which re-ran the benchmark after fixing protocol issues, notably the so-called Drop Last Trick in which the final short batch is silently discarded and results are inflated. It found ModernTCN competitive but frequently behind PatchTST, with a smaller effective receptive field than continuous-kernel convolutional models and a much larger parameter cost. So the CNN-versus-Transformer gap in forecasting is real but small, and, exactly as in Debate 001, it is sensitive to protocol.

### Classification

This is where convolution is strongest, and the evidence is unusually clean. InceptionTime, from 2020, an ensemble of five CNN models, matched HIVE-COTE, a meta-ensemble of thirty-seven classifiers, on the UCR archive, at a tiny fraction of the compute and scaling to millions of series. ROCKET, also from 2020, is the decisive result: it uses random convolutional kernels, with no learning of the filters at all, followed by a plain linear classifier, and reaches state-of-the-art accuracy at roughly a hundredth of the training time, about two hours across all eighty-five UCR datasets against more than six days for InceptionTime.

The conclusion to draw is that if random convolutions plus a linear head already reach the state of the art, then what carries the signal for time-series classification is convolution's inductive bias itself, not the learning of the filters.

Both TimesNet and ModernTCN also report state of the art on anomaly detection and imputation; TimesNet's average anomaly-detection F1 of 86.3 beat the Transformer baselines it was compared against.

---

## 4. Where CNNs fail, and the honest limits

The receptive field is fixed. A TCN can only see R steps back, so if the dependency is longer than your dilation schedule the model is blind to it and cannot grow its memory to find out. Attention adapts, whereas you must know or guess the horizon in advance.

Weight sharing assumes that the pattern means the same thing everywhere, which is the deep assumption and one that non-stationary series violate: regime shifts, trends and time-varying relationships all break translation invariance. This is precisely why every convolutional forecaster bolts on instance normalisation, or RevIN, to re-centre the series so that the shared filter stays valid.

Cross-variable structure needs explicit design, since a per-channel convolution ignores the other channels and mixing them takes extra machinery. Exact long-range retrieval is also hard, because attention can fetch one specific distant point directly whereas a convolution must propagate it through many dilated layers, which is a lossy and approximate route. Finally, the locality prior can be a liability when the signal is genuinely global and the whole history matters equally.

---

## 5. What the evidence, read together, suggests

This is a reading, not a verdict. The editor decides what stands.

Yes, CNNs work on time series, but the working CNN is a TCN, causal and dilated and residual, with a receptive field you design rather than hope for. The mechanism is real and specific, consisting of local pattern detectors, translation invariance, parallel training, short gradient paths and an explicit receptive field. Performance is competitive to state of the art across tasks: TCNs match or beat RNNs on long sequences, TimesNet and ModernTCN rival Transformers on forecasting, and InceptionTime and ROCKET are state-of-the-art class for classification. But the margin over Transformers and MLPs is small and sensitive to protocol, which is the same conclusion Debate 001 reached about every architecture in this space. The cleanest framing is that convolution is a prior rather than a capability, winning when locality and translation invariance are true of the data and being quietly disabled, through normalisation, when they are not.

---

## 6. Open forks — your call

**F1. Scope.** Forecasting only, or all five mainstream tasks, being forecasting, classification, imputation and anomaly detection? Classification is where the evidence is cleanest and forecasting is where the debate is contested.

**F2. Which CNN?** Keep it at the TCN, or go into TimesNet with its one-to-two-dimensional transformation and ModernTCN with its large kernels as the modern branch? The modern branch is where CNN and Transformer have converged.

**F3. Do we run an experiment?** TCN against LSTM against Transformer on one small forecasting set. This is the third debate to reach the same fork, and one experiment could serve Debates 001, 002 and 003 at once.

**F4. Make the receptive-field math a worked example?** Show, with numbers, what k, d and depth buy you. It is cheap, and it is the single most clarifying thing in this artifact.

**F5. Deliverable shape.** Same as Debates 001 and 002?

**F6. Verdict policy.** Same, in that a debate does eventually get marked settled, or not?

---

## Cross-debate notes

This debate has the same shape as the others, and the convergence is the point. Debate 001 found that architecture is rarely the first-order variable, with protocol and data regime dominating, and the same is true here, since ModernTCN's win evaporates under a corrected protocol. Debate 002 found that what makes a component work is the right inductive bias plus good conditioning rather than merely being nonlinear, and here the component is the convolution, whose power is a prior rather than raw expressive capacity.

---

## Sources (plain text)

- Bai, Kolter, Koltun. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271, 2018. (TCN.)
- van den Oord et al. *WaveNet: A Generative Model for Raw Audio.* 2016. (Origin of dilated causal convolutions.)
- Yu, Koltun. *Multi-Scale Context Aggregation by Dilated Convolutions.* 2016.
- Wu, Hu, Liu, Zhou, Wang, Long. *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.* ICLR 2023. arXiv:2210.02186.
- Luo, Wang. *ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis.* ICLR 2024.
- *ModernTCN Revisited: A Reproducibility Study with Extended Benchmarks* (and the companion critical re-evaluation), 2025.
- Ismail Fawaz et al. *InceptionTime: Finding AlexNet for Time Series Classification.* Data Mining and Knowledge Discovery, 2020. arXiv:1909.04939.
- Dempster, Petitjean, Webb. *ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels.* DMKD, 2020. arXiv:1910.13051.
- Hewamalage, Bergmeir, Bandara. *An Experimental Review on Deep Learning Architectures for Time Series Forecasting*, 2021. arXiv:2103.12057.

---

## Further down the rabbit hole

Everything below is real, and grouped so you can go as deep as you want.

**Convolution for sequences**
- van den Oord et al. *WaveNet: A Generative Model for Raw Audio.* 2016. arXiv:1609.03499.
- Yu & Koltun. *Multi-Scale Context Aggregation by Dilated Convolutions.* ICLR 2016. arXiv:1511.07122.
- Lea et al. *Temporal Convolutional Networks for Action Segmentation and Detection.* 2016.
- Bai, Kolter, Koltun. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* 2018. arXiv:1803.01271.

**Forecasting with convolutions**
- Wu et al. *TimesNet.* ICLR 2023. arXiv:2210.02186.
- Luo & Wang. *ModernTCN.* ICLR 2024.
- Liu et al. *SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction.* NeurIPS 2022. arXiv:2106.09305.
- Oreshkin et al. *N-BEATS.* ICLR 2020. arXiv:1905.10437.
- Hewamalage, Bergmeir, Bandara. *An Experimental Review on Deep Learning Architectures for Time Series Forecasting.* 2021. arXiv:2103.12057.
- *ModernTCN Revisited: A Reproducibility Study with Extended Benchmarks.* 2025.

**Classification with convolutions**
- Wang, Oates, Bailis. *Time Series Classification from Scratch with Deep Neural Networks (FCN / ResNet).* IJCNN 2017. arXiv:1611.06455.
- Ismail Fawaz et al. *InceptionTime: Finding AlexNet for Time Series Classification.* DMKD 2020. arXiv:1909.04939.
- Dempster, Petitjean, Webb. *ROCKET.* DMKD 2020. arXiv:1910.13051.
- Dempster et al. *MiniRocket: A Very Fast (Almost) Deterministic Transform.* KDD 2021. arXiv:2012.08791.
- Dempster et al. *Hydra: Competing Convolutional Kernels for Fast and Accurate Time Series Classification.* DMKD 2023. arXiv:2203.10807.
- Dau et al. *The UCR Time Series Archive.* 2018. arXiv:1810.07758.
- Bagnall et al. *The UEA Multivariate Time Series Classification Archive.* 2018. arXiv:1811.00075.

**Surveys & archives**
- Zhao et al. *Convolutional Neural Networks for Time Series Classification: A Review.* 2017. arXiv:1709.05206.
- Bagnall et al. *The Great Time Series Classification Bake Off.* DMKD 2017. arXiv:1602.01711.
- Ismail Fawaz et al. *Deep Learning for Time Series Classification: A Review.* DMKD 2019. arXiv:1809.04356.
- Godahewa et al. *Monash Time Series Forecasting Archive.* 2021. arXiv:2105.06643.
- Middlehurst et al. *Bake off redux: A Review and Experimental Evaluation of Recent Time Series Classification Algorithms.* DMKD 2024.

**Code**
- Repositories: timeseriesAI/tsai; hfawaz/dl-4-tsc; cure-lab/TimesNet; luo-chen/ModernTCN; angus924/rocket.
