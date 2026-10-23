# Debate 001 — Attention-augmented RNNs vs. LSTMs for time series

**Status:** open — draft 0.3
**Opened:** 2026-10-22
**Editor:** Divyansh Tripathi
**Written by:** AI assistant
**Framing as asked:** "RNN with attention vs LSTM for time series data"

---

## 0. A framing problem, up front

The question as posed contains a category error, and untangling it changes what we are actually arguing about. An LSTM is an RNN, so "RNN with attention vs LSTM" cannot be a contest between two kinds of recurrence. Once that is cleared up, there are four distinct architectures on the table.

The first is plain gated recurrence, meaning an LSTM or a GRU: a sequence model with a memory cell and no attention. The second is recurrence with attention bolted on, an RNN or LSTM encoder plus an attention layer, or self-attention stacked on top of the recurrent stack. The third is attention with no recurrence at all, which is the Transformer, where attention replaces the recurrent state entirely. The fourth is what might be called modern recurrence: xLSTM and the linear-RNN family such as Mamba and RWKV, which keep the recurrent structure but adopt the associative memory that attention introduced.

So the real question has three parts. Does attention help a recurrent model on time series? If it does, at what point does it justify dropping recurrence altogether? And is there a third option, namely rebuilding recurrence with the ideas attention brought, which is what xLSTM is betting on? The title keeps the original wording for continuity, but everything below runs along this four-way axis.

Two premises in the question are also worth correcting before we start. The first is that recurrence and attention are exhaustive alternatives, which the xLSTM line shows they are not. The second is that the newer architecture must be the better one, which the protocol critiques in section 4 will complicate.

---

## 1. The case for plain recurrence (LSTM / GRU)

A time series is an ordered sequence, and a recurrent state is built to consume it in order, one step at a time. Attention, by contrast, is permutation-invariant by construction. That property is a virtue for language and a liability for time series, where the position of a point carries much of the signal and has to be re-taught to attention through positional encodings. The recurrent model gets the ordering for free.

Plain recurrence also works on short windows and small datasets. An LSTM does not need a long context to be useful, and it tends to degrade gracefully when data is scarce. Where the series is noisy, non-stationary, or strongly periodic, this is often exactly what you want, and the empirical record here is unusually consistent. Hewamalage, Bergmeir and Bandara trained more than 38,000 models across twelve forecasting problems and some 50,000 series, and found LSTM and CNN to be the strongest alternatives, with LSTM the most accurate. A later data-centric benchmark, published in Forecasting in 2024, found LSTM best under varying noise, frequency and sequence length, winning 38 of 40 configurations; Transformers were best only when the delay length varied, and even there not by a statistically significant margin. A study on electronics price prediction reached a similar conclusion, with a canonical LSTM proving more stable and robust than Informer, Reformer or the vanilla Transformer.

There is also a cost argument. Recurrence uses constant memory per step and is cheap when windows are short, whereas attention is quadratic in sequence length. As windows get shorter, the efficiency advantage moves toward recurrence.

---

## 2. The case for attention

If the value that predicts time t is buried an arbitrary distance in the past, attention can look directly at it. A recurrent state has to carry that value through every intermediate step, which is where it tends to lose it. The data-centric benchmark mentioned above found Transformers best at generalising to delay lengths they had not seen, which is the one axis where they clearly led.

A 2022 comparative analysis of eight architectures across five benchmark datasets, covering finance, weather and sales, found the Transformer producing the lowest error most often and described it as the most versatile of the models tested. Attention, V-LSTM, CNN-LSTM and CV-LSTM were also near the top, although the differences among the leading models were often not statistically significant.

Attention can also improve on a plain LSTM directly. A 2018 study comparing LSTMs with attention mechanisms on five financial time series found that an LSTM with attention outperformed a plain LSTM on four of the five, though with higher variance and real sensitivity to hyperparameters. A 2025 study comparing LSTM-based encoder-decoders with Transformer baselines found the attention-augmented LSTMs competitive with or better than the Transformers when windows were short, and noted that the choice of attention scoring function materially affects accuracy. Attention added to recurrence, then, is a real but modest gain in the right regime.

---

## 3. The case for xLSTM

xLSTM stands for Extended Long Short-Term Memory, and it was published at NeurIPS in 2024 by Beck, Pöppel, Spanring, Auer, Prudnikova, Kopp, Klambauer, Brandstetter and Hochreiter. It is the most serious attempt so far to answer the claim that recurrence lost to attention by rebuilding the recurrent architecture rather than conceding to attention. It keeps the LSTM memory cell and adds two things.

The first is exponential gating, with normalisation and stabilisation, which lets the model revise a stored value when a better match arrives. A sigmoid-gated LSTM finds this difficult. The second is a new memory structure. The sLSTM keeps a scalar memory with memory mixing and adds heads, while the mLSTM replaces the scalar cell with a matrix memory updated by a covariance rule. That matrix memory is, in effect, the same key-value associative retrieval that attention performs, but carried out recurrently. Stacked into residual blocks, the result is a modern recurrent model with linear compute and constant memory in sequence length, and no growing key-value cache.

The case for it, in its authors' terms, rests on four claims. First, it addresses the three limitations that drove people away from LSTMs. The paper is explicit that those limitations were the inability to revise storage decisions, the limited scalar storage capacity, and the lack of parallelisability caused by memory mixing, and it argues that exponential gating, matrix memory and the mLSTM's parallel training mode address them respectively. Second, it beats Transformers and state-space models on language modelling within the paper's controlled setting: on 15 billion tokens of SlimPajama at roughly 400 million parameters, the xLSTM variants achieved the best validation perplexity of the models compared, 13.43 and 13.48 against 14.25 for Llama and 13.70 for Mamba. On the PALOMA suite, xLSTM had lower perplexity than Mamba in 568 of 571 domains. Third, it makes a theoretical claim as well as an empirical one. Because sLSTM keeps memory mixing, it can solve state-tracking tasks such as parity that Transformers and state-space models are argued to be unable to solve, which is a stronger statement than winning a benchmark. Fourth, efficiency is the headline: xLSTM 7B, at 7 billion parameters trained on 2.3 trillion tokens, is reported as the fastest and most memory-efficient model among its 7B-scale comparisons, roughly 50 per cent faster at generation than Mamba, faster than Llama, with the highest prefill throughput and a constant memory footprint against a growing key-value cache.

On time series specifically, which is our subject, it is the strongest recurrent result so far. xLSTMTime beats DLinear consistently and edges PatchTST, doing best at long horizons. xLSTM-Mixer, published at NeurIPS in 2025, reports new state of the art on six of seven datasets and is the top purely supervised model on GIFT-Eval, at one to two orders of magnitude less memory than a strong mixer baseline. Its ablation is worth noting: replacing sLSTM with a plain LSTM or GRU makes performance drop, which suggests the cell itself is doing work.

The counter-case is real, and it should be kept in view. The original language-modelling comparison trained all models with the same learning rate for comparability, which critics argue contradicts peer-reviewed results for RNNs such as HGRN and amounts to tuning one model and not the others. An independent technical report re-ran seven parameter-matched architectures with the learning rate tuned per model and task pair. The authors' own follow-up is measured: xLSTM 7B lands mid-range on the Open LLM Leaderboard v2, behind Llama-3.1-8B, Qwen2.5-7B and Gemma-2-9B, so its win is speed rather than accuracy. Training at 7B was also unstable and needed targeted fixes including RMSNorm, soft-capping the gates and negative input-gate initialisation. The time-series gains, finally, are often small and dataset-specific: xLSTMTime underperforms PatchTST in some settings, and xLSTM-Mixer is weaker on Traffic and ETTh2. Part of the gain comes from the mixer scaffolding and normalisation rather than from the cell alone.

The set of cases together makes a sharper point than any one of them alone. x² fails on expressivity, e^x passes expressivity and fails on conditioning, and xLSTM is the strongest evidence yet that recurrence is not simply dead. Neither nonlinearity nor non-polynomiality is the currency on its own, and in the same way neither recurrence nor attention is the currency here; what matters is the fit between the architecture and the data.

---

## 4. The skeptical middle: is any of this real?

A well-known paper argues that the whole attention-versus-recurrence fight has been measured on a flawed benchmark. Zeng, Chen, Zhang and Xu, in a paper titled *Are Transformers Effective for Time Series Forecasting?*, published at AAAI in 2023, showed that a one-layer linear model with decomposition, called DLinear or LTSF-Linear, beat every sophisticated Transformer on nine standard long-term forecasting benchmarks, often by 20 to 50 per cent. Their argument was that the Transformers' apparent advantage came from using direct multi-step forecasting while their baselines were iterative and therefore accumulated error, and not from attention capturing temporal structure. They also showed that Transformer error often fails to improve as the look-back window grows, which a model with real long-range capability should not do.

The rebuttal came from the Hugging Face Autoformer blog in 2023. It argued that DLinear's edge came from comparing univariate linear models against multivariate Transformers and from the linear model's inability to use covariates such as calendar features, which cannot distinguish a weekend from a weekday. With equal-sized models under the same protocol, the blog reported, Transformers win. Further skepticism came from a 2023 paper titled *Two Steps Forward and One Behind*, which reports that models including Transformer variants can lose to a naive persistence baseline in many cases.

The takeaway is that a large share of the reported gap between architectures is entangled with the evaluation protocol, meaning direct versus iterative forecasting, univariate versus multivariate, the availability of covariates, normalisation, and the look-back window, rather than with the sequence model itself.

---

## 5. What the evidence, read together, suggests

This is a reading, not a verdict. The editor decides what stands.

Architecture is rarely the first-order variable. Evaluation protocol, the data regime including length, noise, periodicity, horizon, data volume and covariates, and normalisation choices all move results more than the choice between attention and recurrence. Recurrence has the better prior for short, noisy, low-data and strongly ordered series, while attention has the better prior for long-range, variable-delay and covariate-rich problems, and only when there is enough data to fit its larger and harder-to-tune parameter space. Attention added to recurrence is a real but modest win in some settings such as finance and short-window forecasting, though it is sensitive to hyperparameters and does not always transfer from synthetic to real data, as a blood-glucose study found when an attention module helped on synthetic data but not on real patient data. Modern recurrence, meaning xLSTM, is the strongest current answer to the claim that recurrence is dead, absorbing attention's associative memory while keeping linear cost and currently leading the standard time-series benchmarks among recurrent models, but its margins are often small, its most-cited language-modelling result is contested on protocol grounds, and its wins are dataset-specific. Taken together, this is evidence that the axis is not settled, rather than evidence that recurrence has won. The strongest honest statement is that the answer is context-dependent.

---

## 6. Open forks — your call

**F1. Scope of the axis.** Keep all four architectures, or narrow to just "attention added to an LSTM, worth it or not?" The four-way version is more honest but sprawling; the two-way version is sharper but drops both the Transformer fight and the xLSTM fight, which is where most of the noise is.

**F2. Task scope.** Forecasting only, or also classification, anomaly detection and imputation? Attention tends to look better once you leave pure forecasting.

**F3. Evidence standard.** Benchmark papers only, or include practitioner reports and the protocol critiques? The DLinear rebuttal only exists at the level of protocol critique.

**F4. Do we run an experiment?** Keep this a reading-based essay, or add a small reproducible benchmark with one or two datasets and LSTM, LSTM-plus-attention and Transformer under a controlled protocol? An experiment would settle the protocol question for our own use, but it is real work.

**F5. Deliverable shape.** One long essay, or a running state of the argument with a living evidence table?

**F6. Verdict policy.** Does a debate ever get marked settled, and if so by what standard: a majority of benchmark papers, a decisive experiment, or your judgment call?

**F7. Weight of xLSTM.** Is xLSTM evidence that recurrence is back, a genuine fourth camp that changes the verdict, or a well-researched special case whose wins are real but narrow?

---

## Cross-debate notes

This debate has the same shape as the others, and the convergence is the point. Debate 002 found that what makes a component work is the right inductive bias plus good conditioning, not merely being nonlinear, and Debate 003 found that a component's power is a prior rather than a capability. The same reading applies here: an architecture is a prior over the data, and it wins when that prior is true.

---

## Sources (plain text)

- Zeng, Chen, Zhang, Xu. *Are Transformers Effective for Time Series Forecasting?* AAAI 2023. arXiv:2205.13504. (DLinear / LTSF-Linear.)
- Hugging Face blog. *Yes, Transformers are Effective for Time Series Forecasting (+ Autoformer)*, 2023. (Rebuttal to DLinear.)
- Hewamalage, Bergmeir, Bandara. *An Experimental Review on Deep Learning Architectures for Time Series Forecasting*, 2021. arXiv:2103.12057.
- *Data-Centric Benchmarking of Neural Network Architectures for the Univariate Time Series Forecasting Task*. Forecasting (MDPI), 2024.
- *A Comparative Analysis of State-of-the-Art Time Series Forecasting Algorithms*, IEEE CSCI 2022.
- *A Comparison of LSTMs and Attention Mechanisms for Forecasting Financial Time Series*, 2018. arXiv:1812.07699.
- *A Comparative Study of LSTM Efficiency vs. Transformer Power for Localized Time Series Forecasting*, 2025.
- *LSTMs and Neural Attention Models for Blood Glucose Prediction*, PMC, 2021.
- *Two Steps Forward and One Behind: Rethinking Time Series Forecasting with Deep Learning*, 2023. arXiv:2304.04553.
- Beck et al. *xLSTM: Extended Long Short-Term Memory.* NeurIPS 2024. arXiv:2405.04517.
- Beck et al. *xLSTM 7B: A Recurrent LLM for Fast and Efficient Inference.* ICML 2025. arXiv:2503.13427.
- Alharthi, Mahmood. *xLSTMTime: Long-Term Time Series Forecasting with xLSTM.* AI (MDPI), 2024. arXiv:2407.10240.
- Kraus et al. *xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories.* NeurIPS 2025. arXiv:2410.16928.
- Ludwig. *Competitiveness of xLSTM Variants across Formal Language Tasks* (independent technical report), 2026.
- *Transformers versus LSTMs for Electronic (price prediction)*, OpenReview.

---

## Further down the rabbit hole

Everything below is real, and grouped so you can go as deep as you want.

**Foundations**
- Hochreiter & Schmidhuber. *Long Short-Term Memory.* Neural Computation, 1997.
- Cho et al. *Learning Phrase Representations using RNN Encoder-Decoder (GRU).* EMNLP 2014. arXiv:1406.1078.
- Bahdanau, Cho, Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015. arXiv:1409.0473.
- Vaswani et al. *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762.
- Lim et al. *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.* 2019. arXiv:1912.09363.
- Salinas et al. *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.* 2017. arXiv:1704.04110.

**The forecasting debate**
- Zhou et al. *Informer.* AAAI 2021. arXiv:2012.07436.
- Wu et al. *Autoformer.* NeurIPS 2021. arXiv:2106.13008.
- Zhou et al. *FEDformer.* ICML 2022. arXiv:2201.12740.
- Nie et al. *A Time Series is Worth 64 Words (PatchTST).* ICLR 2023. arXiv:2211.14730.
- Zeng et al. *Are Transformers Effective for Time Series Forecasting?* AAAI 2023. arXiv:2205.13504. (DLinear.)
- Oreshkin et al. *N-BEATS.* ICLR 2020. arXiv:1905.10437.
- Challu et al. *N-HiTS.* AAAI 2023. arXiv:2201.12886.

**Skeptical / contrarian**
- Hugging Face. *Yes, Transformers are Effective for Time Series Forecasting (+ Autoformer).* 2023. (Rebuttal to DLinear.)
- *Two Steps Forward and One Behind: Rethinking Time Series Forecasting with Deep Learning.* 2023. arXiv:2304.04553.
- Elsayed et al. *Do We Really Need Deep Learning Models for Time Series Forecasting?* 2021. arXiv:2101.02118.
- Makridakis et al. *The M4 Competition.* International Journal of Forecasting, 2020.

**Modern recurrence (the xLSTM thread)**
- Beck et al. *xLSTM: Extended Long Short-Term Memory.* NeurIPS 2024. arXiv:2405.04517.
- Beck et al. *xLSTM 7B.* ICML 2025. arXiv:2503.13427.
- Alharthi & Mahmood. *xLSTMTime.* AI (MDPI) 2024. arXiv:2407.10240.
- Kraus et al. *xLSTM-Mixer.* NeurIPS 2025. arXiv:2410.16928.
- Gu, Goel, Ré. *S4.* ICLR 2022. arXiv:2111.00396.
- Gu & Dao. *Mamba.* 2023. arXiv:2312.00752.
- Peng et al. *RWKV.* 2023. arXiv:2305.13048.
- Katharopoulos et al. *Transformers are RNNs (Linear Attention).* ICML 2020. arXiv:2006.16236.
- Ludwig. *Competitiveness of xLSTM Variants across Formal Language Tasks.* 2026. (Independent report.)

**Surveys & benchmarks**
- Hewamalage, Bergmeir, Bandara. *An Experimental Review on Deep Learning Architectures for Time Series Forecasting.* 2021. arXiv:2103.12057.
- Benidis et al. *Deep Learning for Time Series Forecasting: Tutorial and Literature Survey.* ACM CSUR 2022. arXiv:2004.13408.
- Lim & Zohren. *Time-Series Forecasting with Deep Learning: A Survey.* 2021. arXiv:1909.00560.
- Godahewa et al. *Monash Time Series Forecasting Archive.* 2021. arXiv:2105.06643.
- *GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation.* 2024.

**Blogs & code**
- Hugging Face Autoformer blog (the DLinear rebuttal).
- Repositories: NX-AI/xlstm; yuqinie98/PatchTST; cure-lab/LTSF-Linear.
