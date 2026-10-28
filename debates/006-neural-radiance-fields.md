# Debate 006 — What are neural radiance fields?

**Status:** open — draft 0.1
**Opened:** 2026-10-28
**Editor:** Divyansh Tripathi
**Written by:** AI assistant
**Framing as asked:** "what are neural radiance fields?"

---

## 0. Framing: the name is a description of a function, not of a system

The phrase reads as though it names a thing you could point at, and it is more precise than that in one respect and less precise in another. The precise part is the word field. A radiance field is a function on space: give it a point in three dimensions and a direction, and it returns the colour of the light arriving from that direction at that point, together with a density saying how much material sits there. The imprecise part is neural, because the most influential successor to the original model has no neural network in it at all, and the most popular one today has no field either.

So the honest framing is that a neural radiance field is a way of representing a scene as a continuous function, plus a way of turning that function into pixels, plus a training procedure that recovers the function from photographs alone. Those three parts are separable, and nearly every argument in the literature is about which of them was doing the work.

A second framing choice matters as well, which is whether the subject is the model or the task. As a model it is a specific multilayer perceptron with a specific input encoding. As a task it is novel view synthesis from posed images, which is much older than the name and still exists independently of it.

---

## 1. The model, in one paragraph and one equation

The original model, from Mildenhall and colleagues in 2020, is a multilayer perceptron with five inputs and four outputs. The inputs are the position, three coordinates in space, and a viewing direction, two angles. The outputs are a density, which is a single non-negative number, and a colour, three numbers. Written compactly, the network is

    F(x, y, z, theta, phi)  ->  (sigma, r, g, b)

with one deliberate restriction: the density depends only on position, while the colour depends on both position and direction. That restriction is the model's central physical commitment. Density is a property of matter, so it must be the same from every angle, and a point in space cannot be solid when viewed from the left and empty when viewed from the right. Colour is allowed to change with direction, which is what lets the model represent specular highlights and reflections. Multi-view consistency is therefore enforced by the architecture rather than by a loss term.

---

## 2. Turning the function into pixels

A photograph is a measurement of light arriving along rays, so the model is rendered by integrating along a ray. For a camera ray r(t) = o + t.d, the colour is

    C(r) = integral of  T(t) . sigma(r(t)) . c(r(t), d)  dt

where T(t), the transmittance, is the exponential of the negative integral of density from the camera to t, so it says how much light survives the trip. That is classical volume rendering, and it has an exact discrete form once the ray is cut into intervals.

The practical version does not integrate exactly. It samples the ray, evaluates the network at each sample, and applies the quadrature rule that the original paper derives, which is exact for a piecewise-constant density and colour. Sampling uniformly along the whole ray would waste nearly all its budget on empty space, so the model is trained in two passes: a coarse network on stratified uniform samples, then a fine network on samples drawn from the coarse network's own density distribution. That hierarchical scheme is why the architecture is often written as two networks, and it is one of the three or four decisions that made the thing work at all.

The training signal is then embarrassingly simple. There is no 3D supervision, no depth, no mesh, no point cloud. The loss is the squared difference between rendered and observed pixel colours, summed over rays sampled from a set of photographs whose camera poses are already known. A three-dimensional model is learned from two-dimensional supervision, and this is the fact that most reliably surprises people when they first meet the idea.

---

## 3. Why it worked: the positional encoding

The part that is easy to skip and impossible to omit is the input encoding. A multilayer perceptron given raw coordinates is strongly biased toward smooth, low-frequency functions, so it fits the overall colour gradient of a scene and misses the fine detail. The fix is to lift each coordinate into a set of sinusoids before it enters the network,

    gamma(p) = ( sin(2^k pi p), cos(2^k pi p) )  for k = 0 ... L-1

with L of 10 for position and 4 for direction in the original. This maps a scalar into twenty numbers, and it is the same construction the transformer uses for token positions, arrived at independently and for a related reason. The effect is that a small change in position becomes a large change in the input vector, so the network can represent high frequencies without needing enormous weights. Removing this encoding costs more quality than almost any other single change, which is worth remembering because it means the neural network was never the interesting part of the representation.

The numbers, for calibration. The original model trained for 100,000 to 300,000 iterations, taking one to two days on a single V100, per scene. On the synthetic dataset of eight rendered objects it reached about 31.85 dB peak signal-to-noise ratio. On real forward-facing scenes, with 20 to 62 photographs each, it was qualitatively better than everything before it, and the improvement was large enough that the comparisons looked like a category error rather than a ranking.

---

## 4. The debates

The first is whether this is a representation or a renderer. The evidence for the renderer reading is that you can delete the network and keep the result. Plenoxels replaced the multilayer perceptron with a voxel grid of spherical harmonics and kept the volume-rendering loss, training in minutes without any neural network. TensoRF factorised the grid into tensors and did the same. Instant-NGP replaced the input encoding with a multiresolution hash table and cut training from days to seconds or minutes, at equal or better quality. Each of these kept the loss and the renderer and discarded the network, which suggests that the essential contribution was the differentiable volume-rendering formulation, not the multilayer perceptron. The evidence for the representation reading is that the function it learns is genuinely continuous and genuinely three-dimensional, so you can query it anywhere, which a voxel grid cannot do without interpolation.

The second is implicit against explicit, and this is the live one. In 2023, 3D Gaussian splatting represented a scene as a few million anisotropic Gaussian blobs with opacity and colour, rendered by rasterisation rather than by ray marching, and reached real-time frame rates at 1080p with training in minutes. Against that, a per-scene multilayer perceptron that needs a day and a fast GPU looks like a research artefact. The counter-argument is that splatting inherits the NeRF pipeline wholesale, meaning the posed images, the photometric loss and the structure-from-motion initialisation, and that it degrades differently, since an explicit set of blobs has no notion of an unobserved region and no compact prior over what a scene is. Whether the field model is dead or merely one point on a curve is genuinely unsettled.

The third is per-scene optimisation against generalisation. The original model fits one scene and one scene only, and needs hours to do it. A family of follow-ups, including pixelNeRF and IBRNet, made the reconstruction a function of the input images and produced novel views of unseen scenes in a forward pass. By 2024 the large reconstruction models were producing a usable asset from a single image in seconds. If generalisation is the goal, then the per-scene version was a proof of concept rather than a product, and the interesting question moved.

The fourth is whether the camera poses are part of the problem. Almost all of this work assumes poses known in advance, usually from a structure-from-motion pipeline like COLMAP, which itself fails on textureless surfaces and on scenes with little parallax. Pose-free variants exist, including BARF and NeRF--, and they work, but they are more fragile and they are not the default. The hidden dependency is worth stating plainly, because a method that appears to reconstruct a scene from photographs is really reconstructing it from photographs plus a solved camera trajectory.

The fifth is whether a radiance field understands anything. It has no semantics, no object identity, no physics and no notion of a surface, only density. It cannot tell you that the thing on the left is a chair, and it cannot be lit differently after the fact. Semantic and panoptic extensions bolt labels onto the field, and inverse-rendering extensions bolt on materials and lights, but the base object is a view-synthesis model and calling it a scene representation is already a slight overreach.

The sixth is cost, and it is the one that decided the field's direction. Training a NeRF is memory-bound in the same way that inference is memory-bound in the serving chapter, and the fixes were the same kind of fix, meaning caching and restructuring rather than more arithmetic. Hash grids won because they traded memory for speed in a way the hardware liked. Splatting won because rasterisation is what the hardware is actually built for. The lesson repeats: the winning representation was not the most elegant one, it was the one that matched the machine.

---

## 5. What the evidence, read together, suggests

This is a reading, not a verdict. The editor decides what stands.

The technical answer is that a neural radiance field is a continuous function from position and direction to density and colour, trained only on posed photographs by making a differentiable volume renderer agree with them. The historically accurate answer is that the idea, rather than the network, was the contribution: differentiable volume rendering as a training signal for a 3D representation, with the sinusoidal encoding making a small network able to hold high frequencies.

What makes the debate worth keeping open is that each of the three parts has since been replaced in a different direction. The network was replaced by hash grids and voxel tensors. The field was replaced by explicit primitives. The per-scene fit was replaced by feed-forward reconstruction. The loss survived all three, and it is the part nobody argues about, which is usually a sign that it was the load-bearing part.

---

## 6. Open forks — your call

**F1. Which is the chapter about?** The model as published, the rendering formulation, or the task of novel view synthesis? The three want different depths.

**F2. How much of the mathematics?** Keep the rendering integral and the encoding as they are, or derive the quadrature rule, the transmittance recursion, and the way the two-pass sampling weights are computed?

**F3. Do we run an experiment?** A tiny NeRF on a synthetic scene, implemented in a page or two of PyTorch, trained for a few minutes, is genuinely within reach and would make the encoding argument empirical. The alternative is a comparison of the encoding against no encoding, which is the cheapest version of the central ablation.

**F4. Does Gaussian splatting get its own chapter?** It is now the default in practice, it is a different mathematical object, and it may deserve separate treatment rather than a section.

**F5. How far into the applied territory?** Robotics, autonomous driving, and street-scale reconstruction are where the money is, and they bring different constraints, meaning dynamic scenes, unknown poses, and real-time budgets.

**F6. Verdict policy.** Same as before?

**F7. Deliverable shape.** Same as before?

---

## Cross-debate notes

Debate 003 asked whether a convolution is the right prior for a time series, and found that a component's power is a prior rather than a capability. The same question appears here in a sharper form, since the positional encoding is a prior over frequency content and the density-is-view-independent restriction is a prior over physics, and both do more work than the network does.

Debate 004 found that the binding constraint in serving is memory bandwidth rather than arithmetic, and the radiance field literature reached the same conclusion from the other end: hash grids, voxel tensors and rasterised splats all won by moving fewer bytes, not by computing less.

Debate 002 asked why tanh and not a polynomial, and the answer there was partly about frequency. The positional encoding is the same insight used deliberately, which is that a network's ability to represent high frequencies is a property of its inputs before it is a property of its depth.

---

## Sources (plain text)

Mildenhall, Srinivasan, Tancik, Barron, Ramamoorthi, Ng. *NeRF: Representing scenes as neural radiance fields for view synthesis.* ECCV 2020. arXiv:2003.08934.

Barron, Mildenhall, Tancik, Hedman, Martin-Brualla, Srinivasan. *Mip-NeRF: A multiscale representation for anti-aliasing neural radiance fields.* ICCV 2021. arXiv:2103.13415.

Barron, Mildenhall, Verbin, Srinivasan, Hedman. *Mip-NeRF 360.* CVPR 2022. arXiv:2111.12077.

Martin-Brualla et al. *NeRF in the Wild.* CVPR 2021. arXiv:2008.02268.

Müller, Evans, Schied, Keller. *Instant neural graphics primitives with a multiresolution hash encoding.* ACM TOG, 2022. arXiv:2201.05989.

Fridovich-Keil et al. *Plenoxels: radiance fields without neural networks.* CVPR 2022. arXiv:2112.05131.

Chen et al. *TensoRF: tensorial radiance fields.* ECCV 2022. arXiv:2203.09517.

Kerbl, Kopanas, Leimkuehler, Drettakis. *3D Gaussian splatting for real-time radiance field rendering.* ACM TOG, SIGGRAPH 2023. arXiv:2308.04079.

Yu et al. *pixelNeRF: neural radiance fields from one or few images.* CVPR 2021. arXiv:2012.02190.

Lin et al. *BARF: bundle-adjusting neural radiance fields.* ICCV 2021. arXiv:2104.06405.

Tancik et al. *Fourier features let networks learn high frequency functions in low dimensional domains.* NeurIPS 2020. arXiv:2006.10739.

---

## Further down the rabbit hole

Everything below is real, and grouped so you can go as deep as you want.

**The original and its direct line**
- Mildenhall et al. *NeRF.* ECCV 2020. arXiv:2003.08934.
- Barron et al. *Mip-NeRF.* ICCV 2021. arXiv:2103.13415.
- Barron et al. *Mip-NeRF 360.* CVPR 2022. arXiv:2111.12077.
- Barron et al. *Zip-NeRF: anti-aliased grid-based neural radiance fields.* ICCV 2023. arXiv:2304.06706.

**Speed and the death of the multilayer perceptron**
- Müller et al. *Instant neural graphics primitives.* 2022. arXiv:2201.05989.
- Fridovich-Keil et al. *Plenoxels.* CVPR 2022. arXiv:2112.05131.
- Chen et al. *TensoRF.* ECCV 2022. arXiv:2203.09517.
- Reiser et al. *Merkat: merging spatial data structures for neural fields.* 2023.

**Explicit representations**
- Kerbl et al. *3D Gaussian splatting.* SIGGRAPH 2023. arXiv:2308.04079.
- Yu et al. *Mip-Splatting.* CVPR 2024. arXiv:2311.16493.
- Chen, Lee. *3D Gaussian splatting as Markov chain Monte Carlo.* NeurIPS 2024. arXiv:2404.09591.

**Poses, dynamics and generalisation**
- Lin et al. *BARF.* ICCV 2021. arXiv:2104.06405.
- Wang et al. *NeRF--: neural radiance fields without known camera parameters.* 2021. arXiv:2102.07064.
- Yu et al. *pixelNeRF.* CVPR 2021. arXiv:2012.02190.
- Wang et al. *NeRFDiff / LRM: large reconstruction model for single image to 3D.* 2023. arXiv:2311.04400.

**Applications and systems**
- Tancik et al. *Nerfstudio: a modular framework for neural radiance field development.* SIGGRAPH 2023. arXiv:2302.04264.
- Tancik et al. *Block-NeRF: scalable large scene neural view synthesis.* CVPR 2022. arXiv:2202.05240.
- Mildenhall et al. *Local light field fusion*, and the LLFF dataset the original paper evaluates on.

**Code**
- `nerfstudio-project/nerfstudio`, `NVlabs/instant-ngp`, `graphdeco-inria/gaussian-splatting`, and the original `bmild/nerf` in TensorFlow.
