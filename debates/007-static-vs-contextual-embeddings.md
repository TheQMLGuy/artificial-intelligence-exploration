# Debate 007 — Static vs contextual embeddings

**Status:** open — draft 0.1
**Opened:** 2026-10-28
**Editor:** Divyansh Tripathi
**Written by:** AI assistant
**Framing as asked:** "static vs contextual embedding"

---

## 0. Framing: three different questions are hiding in one

Asked as a comparison, the question sounds settled, since contextual won and everyone knows it. Before that can be assessed, three separate questions have to be pulled apart, because they have different answers.

The first is about the **unit**. A static embedding gives one vector per word type, so every occurrence of "bank" gets the same numbers. A contextual embedding gives one vector per token occurrence, so the same word in two sentences gets two different vectors. That is a claim about what is being represented.

The second is about the **mechanism**. Something has to read the sentence and produce the token vector, and the history runs from recurrent language models to transformers. That is a claim about architecture, and it is separable from the first question, because you can make a contextual model out of an LSTM and you can make a static one out of a transformer by averaging.

The third is about the **training and the frozenness**. Word2vec and GloVe were trained once and then frozen, and in most uses they sat at the bottom of a task-specific network as a lookup table that was either frozen or fine-tuned. BERT is also pretrained and then fine-tuned, so the "static versus contextual" axis has nothing to do with frozen weights. Conflating the two makes the argument look easier than it is.

There is also a scale confound sitting under all three, which is that contextual models arrived with much larger corpora and much more compute. Any honest comparison has to say what would have happened to static embeddings given the same budget, and the answer is that they get better too.

---

## 1. What a static embedding is

The idea rests on the distributional hypothesis, stated by Harris in 1954 and put memorably by Firth in 1957 as the observation that a word is known by the company it keeps. If meaning is distribution, then a word's meaning can be summarised by the statistics of the contexts it appears in.

Word2vec, in 2013, made this practical. Skip-gram predicts the surrounding words from the centre word and trains a small two-layer network whose input weights become the embedding; the trick was negative sampling, which turns the expensive softmax over the whole vocabulary into a handful of binary decisions against sampled noise words. CBOW does the reverse prediction. The vectors that fall out have a linear structure nobody designed: the offset between "king" and "man" is close to the offset between "queen" and "woman", so analogy becomes vector addition and a nearest-neighbour search.

GloVe, in 2014, took the global co-occurrence matrix instead of sampled windows and fitted a log-bilinear factorisation to it, which is a different route to the same kind of object. FastText, in 2016, represented each word as the sum of its character n-gram vectors, which gives a vector to a misspelling and to a word never seen in training.

All three produce a table with one row per word. That table is the model. Lookup is a row fetch, there is no computation that depends on the sentence, and the vector for a word is fixed at the moment training ends.

---

## 2. The case for static

The strongest case is not nostalgia, it is that a type-level vector is the right object for a large class of jobs, and it is enormously cheaper.

Consider what a static embedding actually buys. It is a table of a few hundred megabytes at most, it needs no forward pass, and retrieval is a nearest-neighbour search in a fixed space, which is the operation that modern vector databases are built to do quickly. For search, deduplication, clustering, topic modelling, recommendation and cross-lingual alignment, the unit of interest genuinely is the word or the phrase, and a per-occurrence vector is an awkward answer to a type-level question. When you want to know whether two documents are about the same thing, you want one vector per document, and the standard way to get it is still to average static vectors or to pool a contextual model down to one.

There is a second argument, about interpretability. A static space is a single object that can be projected, rotated, aligned to another language and inspected once. Sense-specific structure lives in it as directions, and the geometry is stable enough that people build dictionaries and cross-lingual maps on top of it. A contextual model has no such single object: its geometry is per-token, per-layer and per-input, which is why interpreting it is a field of its own.

The third argument is about the confound. Static embeddings trained on modern-scale corpora are much better than the 2013 versions, and the gain from more data and better objectives is large in both families. If the comparison is 2013 word2vec against 2019 BERT, then most of the gap is attributable to everything that changed at once, and only the ablation tells you how much was contextuality itself.

---

## 3. The case that contextuality was necessary

The core argument is polysemy, and it is not a small technicality. A single vector for "bank" has to serve the financial institution, the river edge and the verb, so it can only be a weighted average of the senses, weighted by their corpus frequencies. For a word whose senses are balanced, that average may be close to none of them. The same holds for parts of speech, where "run" as a noun and as a verb share a row.

Contextual models fix this by construction, and the results were large enough to change the field. ELMo, in 2018, ran a bidirectional language model over the sentence and took a learned combination of its layers as the representation, which immediately improved every task it was plugged into. BERT, in 2018 and 2019, replaced the recurrent encoder with a transformer trained on masked language modelling, and the token vectors it produced became the default input to everything for the next several years.

Two findings from the probing literature made the case sharper. The first is that the layers are not interchangeable. Lower layers behave like syntax, higher layers like semantics, and the useful layer differs by task, which is only possible because the representation is computed rather than looked up. The second is that context resolves sense measurably: benchmarks that ask whether two occurrences of a word are used in the same sense, and word-sense disambiguation tasks against a sense inventory, show gains that a static model cannot reach in principle, because a static model's two occurrences are identical by definition.

There is also a structural argument. A contextual encoder absorbs the whole pipeline that used to sit downstream of the embeddings, meaning the tagger, the parser and the chunker, into one pretrained object. That is not a claim about the vector so much as a claim about the economics of building systems, and it is a large part of why the change was total.

---

## 4. The counter-case: contextual models are less contextual than advertised

The interesting objections are not about polysemy but about how much of the promised contextuality is actually there, and the evidence is uncomfortable for the strong version of the claim.

Ethayarajh, in 2019, measured how much a token's vector changes across contexts, and found that in the upper layers of contextual models the representations of different words crowd into a narrow cone, so the variation across contexts shrinks and the geometry becomes anisotropic. The measured contextuality is therefore not uniform across layers, and in the last layers it can be surprisingly low, which is why pooling the final layer is often a poor choice for sentence similarity.

Bommasani, Davis and Cardie, in 2020, tested the reduction directly. They projected contextual embeddings down to static ones with a linear map and found that a substantial part of the contextual representation is explained by a static component, meaning the per-word part carries most of the information for many tasks and the contextual part contributes a smaller, task-dependent increment. If a linear projection recovers most of the behaviour, then "contextual" describes the mechanism more than the information content.

There is also the anisotropy and rogue-dimension line of work, which found that a handful of dimensions dominate the geometry of contextual representations and that removing them changes nearest-neighbour behaviour drastically. And there are simple baselines that keep embarrassing company with large pretrained encoders on some probing tasks, which suggests that part of the measured advantage was the probe rather than the representation.

None of this says static embeddings are equivalent. It says the difference is smaller and more task-dependent than the headline comparison suggests, and that a model which is contextual by construction need not be contextual in its useful subspace.

---

## 5. What actually changed

Set against all of that, the honest accounting of what the shift delivered is narrower and more concrete than "context".

The unit changed from type to token, which made sense resolution possible and made every downstream representation variable-length. The mechanism changed from a lookup table to a function of the sequence, which is why inference cost became a question at all: a contextual encoder must run, and running it over a long context is exactly the memory problem the serving chapter is about. Tokenisation changed from whole words to subwords, which removed the out-of-vocabulary problem and made the vocabulary a design decision rather than a corpus artefact. And the economics changed, in that one pretrained model replaced a stack of task-specific models.

What did not change is the distributional hypothesis. A contextual encoder is still learning from co-occurrence, only now the co-occurrence statistics are conditioned on the whole sequence rather than collapsed per word. The old idea did not lose, it got a much richer estimator.

---

## 6. What the evidence, read together, suggests

This is a reading, not a verdict. The editor decides what stands.

The technical answer is that static embeddings assign one vector per word type from corpus statistics and contextual embeddings assign one vector per token occurrence from a function of the whole sequence. The useful answer is that the two are not rival answers to one question but answers to different questions, and the choice depends on whether the unit of the task is a word type or an occurrence.

The evidence supports a middle position that neither camp likes. Contextuality is necessary for sense-dependent tasks and it is what made the modern stack possible, so it is not a fad. But the measured contextuality of the popular models is lower than the rhetoric, much of their behaviour reduces to a static component, and the comparison that made the case so convincing bundled context together with scale, data and objectives. The clean statement is that contextual representations were the right engineering move for a reason that was partly different from the reason given at the time, which is the same pattern the deep belief network chapter found.

---

## 7. Open forks — your call

**F1. Is the chapter a history, a technical comparison, or a decision guide?** The three want different depths and different verdicts.

**F2. How much mathematics?** Keep the skip-gram objective and the co-occurrence factorisation as sketches, or develop negative sampling, the GloVe loss and the linear-projection reduction properly?

**F3. Do we run an experiment?** A small word2vec on a modest corpus, then a contextual encoder on the same data, then the Bommasani-style linear projection from contextual to static, which would reproduce the reduction at small scale. This is the experiment the chapter actually needs.

**F4. Does the retrieval side get its own treatment?** Sentence embeddings, contrastive training, and the modern embedding models used in vector search are the place where the debate is still commercially live, and they are neither of the two classic positions.

**F5. How much of the anisotropy and rogue-dimension literature?** It is the strongest evidence against the naive story and it is also technical enough to double the chapter's length.

**F6. Verdict policy.** Does a chapter whose subject has a clear practical winner still get marked settled, or does the winner not settle the argument?

**F7. Deliverable shape.** Same as before?

---

## Cross-debate notes

Debate 004 is the direct consequence of this one. A contextual embedding is a function of the whole sequence, so the sequence has to be re-read on every step, and the KV cache exists because of that. Static embeddings need no cache at all, which is a fair summary of the trade the field made.

Debate 002 asked what an activation function has to do and part of the answer was about frequency. The same question appears in the anisotropy result, where the representation's usable variation is decided by the geometry of the space rather than by the capacity of the encoder.

Debate 006 found that a component's power is a prior rather than a capability, and the same is true here: the distributional hypothesis is the prior, and contextuality is a much better estimator of it rather than a different idea.

---

## Sources (plain text)

Mikolov, Chen, Corrado, Dean. *Efficient estimation of word representations in vector space.* 2013. arXiv:1301.3781.

Mikolov, Sutskever, Chen, Corrado, Dean. *Distributed representations of words and phrases and their compositionality.* NIPS 2013. arXiv:1310.4546.

Pennington, Socher, Manning. *GloVe: global vectors for word representation.* EMNLP 2014.

Bojanowski, Grave, Joulin, Mikolov. *Enriching word vectors with subword information.* 2016. arXiv:1607.04606.

Peters et al. *Deep contextualized word representations.* NAACL 2018. arXiv:1802.05365.

Devlin, Chang, Lee, Toutanova. *BERT: pre-training of deep bidirectional transformers for language understanding.* NAACL 2019. arXiv:1810.04805.

Ethayarajh. *How contextual are contextualized word representations?* EMNLP 2019. arXiv:1909.00512.

Bommasani, Davis, Cardie. *Interpreting pretrained contextualized representations via reductions to static embeddings.* ACL 2020.

Timkey, van Schijndel. *All bark and no bite: rogue dimensions in transformer language models obscure representational quality.* EMNLP 2021. arXiv:2105.11259.

Harris. *Distributional structure.* Word, 1954.

Firth. *A synopsis of linguistic theory.* 1957.

---

## Further down the rabbit hole

Everything below is real, and grouped so you can go as deep as you want.

**The static era**
- Mikolov et al. *Efficient estimation of word representations in vector space.* 2013. arXiv:1301.3781.
- Mikolov et al. *Distributed representations of words and phrases and their compositionality.* NIPS 2013. arXiv:1310.4546.
- Pennington, Socher, Manning. *GloVe.* EMNLP 2014.
- Bojanowski et al. *Enriching word vectors with subword information.* 2016. arXiv:1607.04606.
- Levy, Goldberg. *Neural word embedding as implicit matrix factorization.* NIPS 2014. (Why word2vec and GloVe are the same thing in disguise.)

**The contextual turn**
- Peters et al. *Deep contextualized word representations.* NAACL 2018. arXiv:1802.05365.
- Devlin et al. *BERT.* NAACL 2019. arXiv:1810.04805.
- Radford et al. *Improving language understanding by generative pre-training.* 2018.
- Tenney, Das, Pavlick. *BERT rediscovers the classical NLP pipeline.* ACL 2019. arXiv:1905.05950.

**How contextual is it, really**
- Ethayarajh. *How contextual are contextualized word representations?* EMNLP 2019. arXiv:1909.00512.
- Bommasani, Davis, Cardie. *Reductions to static embeddings.* ACL 2020.
- Timkey, van Schijndel. *Rogue dimensions in transformer language models.* EMNLP 2021. arXiv:2105.11259.
- Rogers, Kovaleva, Rumshisky. *A primer in BERTology.* TACL 2020. arXiv:2002.12327.

**The modern embedding stack**
- Reimers, Gurevych. *Sentence-BERT.* EMNLP 2019. arXiv:1908.10084.
- Gao, Yao, Chen. *SimCSE: simple contrastive learning of sentence embeddings.* EMNLP 2021. arXiv:2104.08821.
- Muennighoff et al. *MTEB: massive text embedding benchmark.* 2022. arXiv:2210.07316.

**Code**
- `gensim` for word2vec, GloVe and FastText; `Hugging Face transformers` and `sentence-transformers` for the contextual side.
