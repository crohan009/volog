---
layout: post
comments: false
title: "Reward Forcing: Real-Time Controllable Streaming Video Synthesis"
subtitle: "A Technical Deep Dive into the Evolution of Video Diffusion Models"
date: 2025-01-15 00:01:01
tags: paper-review, diffusion-models, video-generation, AI, ML, transformers
---

> The quest for generating long, coherent, high-fidelity video in real-time with interactive control over narrative and dynamics has led to a groundbreaking synthesis called **Reward Forcing**. This post explores the technical chronology from DDPMs to this state-of-the-art paradigm.

<!--more-->

---
<h3> Contents </h3>

{: class="table-of-content"}
* TOC
{:toc}

---

## **Introduction: The Quest for Real-Time Video**

The ultimate goal of modern video generation research is to produce long, coherent, high-fidelity video in real-time, with the ability to interactively steer the narrative and control dynamic properties like motion. This ambitious objective requires solving multiple interconnected challenges: **quality**, **speed**, **streaming capability**, and **controllability**.

![Reward Forcing Overview]({{ '/assets/images/reward_forcing/01_overview.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 1. The progression from noise to coherent video through key stages: DDPM, DMD/AR, and Self-Forcing/Sinks.*

**Reward Forcing** represents a synthesis of innovations that addresses all these challenges. To understand it, we need to trace the technical roadmap that led to its development.

---

## **The Roadmap: A Directed Graph of Innovation**

Each technology in this journey builds upon the last, solving a critical limitation. The path is chronological to understand the synthesis that is "Reward Forcing."

![Innovation Roadmap]({{ '/assets/images/reward_forcing/02_roadmap.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 2. The directed graph showing how technologies build upon each other chronologically.*

---

## **1. Foundation: Denoising Diffusion for Quality**

### The DDPM Revolution

Denoising Diffusion Probabilistic Models (DDPMs) established the foundation for unprecedented sample quality, surpassing GANs with more stable training objectives.

![DDPM Mechanics]({{ '/assets/images/reward_forcing/03_ddpm.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 3. Core mechanics of DDPM: forward process (noise addition) and reverse process (denoising).*

#### Forward Process (Fixed)
The forward process gradually adds Gaussian noise to data:

$$q(x_t|x_{t-1}) := \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)$$

This allows direct sampling at any timestep:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$$

where $$\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$$.

#### Reverse Process (Learned)
A neural network $$\epsilon_\theta(x_t, t)$$ learns to predict the noise, enabling iterative denoising from $$X_T$$ back to $$X_0$$.

#### Training Objective

$$L_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, t) \right\|^2 \right]$$

**Strengths:**
- Unprecedented sample quality and diversity
- Stable training dynamics

**Limitation:** The iterative sampling process (e.g., T=1000 steps) is prohibitively slow for real-time applications.

---

## **2. Distribution Matching Distillation (DMD) for Speed**

### Solving the Speed Problem

DMD addresses the massive computational cost of DDPMs by training a "student" model to directly map noise to clean samples in a single step.

![DMD Distillation]({{ '/assets/images/reward_forcing/04_dmd.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 4. DMD distillation: Teacher (multi-step DDPM) vs Student (one-step generator).*

#### Core Idea
Instead of a simple L2 loss on individual samples, DMD minimizes the divergence between the output distributions of teacher and student models. This is achieved by matching the scores (gradients of log-probability) of both distributions.

#### Technical Approach
- Can be formulated with GAN-style losses or minimizing Fisher/KL divergence
- Allows U-Net or DiT backbone to function as a one-step generator
- Drastically reduces inference time

**Contribution to Reward Forcing:** Establishes distillation as the core mechanism for achieving real-time video generation speed.

---

## **3. The Autoregressive Shift for Streaming**

### Why Streaming Requires Architectural Change

Standard video diffusion models use **bidirectional attention**, processing all frames simultaneously. This approach is:
- **Non-causal**: Cannot generate frames sequentially
- **Computationally explosive**: $$O(N^2)$$ complexity with video length

![Attention Comparison]({{ '/assets/images/reward_forcing/05_attention.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 5. Bidirectional attention vs Causal attention: the shift required for streaming video.*

### The Solution: Causal Attention

Reformulating video generation as a sequential process:

$$P(F_{1:N}) = \prod_{i=1}^N p(F_i | F_{<i})$$

**Benefits:**
- **Causal**: Frames processed sequentially
- **Linear complexity**: $$O(N)$$ with video length
- **Enables KV-Cache**: Dramatically reduces redundant computation

### New Challenges Introduced

1. **Exposure Bias**: Model trained on ground-truth past frames but generates using its own (potentially imperfect) outputs at inference. Errors accumulate catastrophically.

2. **Long-Term Consistency**: Finite KV cache or sliding windows lose long-range context, causing semantic drift.

---

## **4. Self-Forcing: Solving Exposure Bias**

### The Train-Test Distribution Gap

Standard autoregressive training (Teacher Forcing) creates a fundamental mismatch: the model never sees its own errors during training, leading to catastrophic error accumulation during long inference rollouts.

![Self-Forcing Loop]({{ '/assets/images/reward_forcing/06_self_forcing.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 6. The Self-Forcing training loop: autoregressive generation with distribution-matching loss.*

### Core Innovation

**Self-Forcing** bridges this gap by exposing the model to its own generated distribution during training:

1. Model generates a full video sequence autoregressively
2. Each frame is conditioned on previously generated frames
3. A **distribution-matching loss** (DMD, SID, or GAN) is applied to the entire generated sequence

This forces the generator's output distribution $$p_\theta(X_{1:N})$$ to match the real data distribution $$p_{data}(X_{1:N})$$.

**Contribution to Reward Forcing:** Provides the stable training paradigm that enables robust, long-horizon generation.

---

## **5. Memory Solutions: From Frame Sink to EMA-Sink**

### The Memory Decay Problem

Simple sliding windows or finite KV caches cause context decay. The model forgets early events, leading to:
- Flickering
- Object disappearance
- Semantic drift

![Memory Solutions]({{ '/assets/images/reward_forcing/07_memory.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 7. Memory solutions: Streaming Long Tuning and the Frame Sink mechanism.*

### Early Solution: Frame Sink

**LongLive (2025)** introduced the Frame Sink concept:
- KV cache tokens from the **first frame** are designated as "sink tokens"
- These tokens are **never evicted** from the cache
- Anchors the video to its initial state

**Limitation:** Static—only preserves context from the first frame.

### Advanced Solution: EMA-Sink

**Reward Forcing** introduces a **dynamic global memory** via Exponential Moving Average:

$$EMA\_Sink\_KV = (1 - \alpha) \cdot EMA\_Sink\_KV + \alpha \cdot Evicted\_KV$$

![EMA-Sink]({{ '/assets/images/reward_forcing/08_ema_sink.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 8. The EMA-Sink: dynamic global memory that updates with evicted KV tokens.*

#### How It Works

1. Maintain a local KV cache window (last L frames)
2. When oldest frame's KV tokens are evicted, update global EMA-Sink buffer
3. At each generation step, queries attend to **both**:
   - Local window of recent frames
   - Dynamic EMA-Sink buffer

**Strengths:**
- Continuously updated summary of entire video history
- Superior long-range coherence
- Adapts to evolving scenes

---

## **6. The Control Problem: RLHF for Video**

### Motivation

We can generate long, consistent videos—but how do we control their dynamics? How can we bias the model toward high-action sequences or aesthetically pleasing motion?

![RLHF Framework]({{ '/assets/images/reward_forcing/09_rlhf.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 9. The RLHF framework: Reward-Weighted Regression (RWR) and Direct Preference Optimization (DPO).*

### The RLHF Framework

1. **Learn Preferences**: Train a Reward Model $$r(x, y)$$ on human preference data
2. **Align the Policy**: Fine-tune generation to maximize reward while staying close to reference:

$$\max_{p_\theta} \mathbb{E}_{x_0 \sim p_\theta} [r(x_0, y)] - \beta D_{KL}[p_\theta || P_{ref}]$$

### Two Key Alignment Algorithms

#### Reward-Weighted Regression (RWR)
A simple offline method that fine-tunes with weighted regression—weights determined by reward scores:

$$L_{RWR}(\theta) = \mathbb{E}[\exp(r(x_0, y)) ||v - v_\theta(x_t, t, y)||^2]$$

*"Do more of what gets high rewards."*

#### Direct Preference Optimization (DPO)
An RL-free alternative working directly on preference pairs $$(x_w, x_l)$$:

$$L_{DPO}(\pi_\theta; \pi_{ref}) = \mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right) \right]$$

---

## **7. Rewarded-DMD: Biased Distillation**

### The Innovation

Standard DMD is "unbiased"—it distills the teacher's entire distribution. **Rewarded-DMD (Re-DMD)** modifies this to incorporate reward signals.

![Rewarded-DMD]({{ '/assets/images/reward_forcing/10_rewarded_dmd.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 10. Rewarded-DMD: incorporating reward weights into the distillation process.*

### Core Idea

During distillation, weight the loss for each teacher-generated sample by its reward model score:

1. Teacher model generates batch of video samples
2. Reward model assigns scores to each sample
3. DMD loss is weighted by these reward scores
4. Student model learns a **biased distribution** toward high-reward outputs

### Result

The one-step student model doesn't just replicate the teacher—its output distribution is **intentionally skewed** toward high-reward regions. It learns to generate high-motion, high-quality videos **by default**.

---

## **8. Reward Forcing: The Complete Synthesis**

### Core Architecture

**Reward Forcing** unifies all innovations into a complete training paradigm:

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backbone** | Causal Autoregressive DiT | Sequential frame generation |
| **Training Foundation** | Self-Forcing | Eliminate exposure bias |
| **Memory System** | EMA-Sink + Local KV-Cache | Long-range consistency |
| **Distillation** | Rewarded-DMD | One-step generation biased toward rewards |

![Reward Forcing Training Loop]({{ '/assets/images/reward_forcing/11_training_loop.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 11. The complete Reward Forcing training loop: integrating all components.*

### Training Loop

1. **Multi-step Teacher** generates batch of video samples
2. **Reward Model** assigns scores (weights)
3. **Rewarded-DMD Loss** combines teacher samples with reward weighting
4. **Self-Forcing Loss** applied holistically to generated sequence
5. **Student Model** (Causal DiT) performs autoregressive rollout using EMA-Sink + KV-Cache
6. Parameters updated via combined loss

---

## **9. Comparison with Prior Methods**

![Comparison]({{ '/assets/images/reward_forcing/12_comparison.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 12. How Reward Forcing builds upon Self-Forcing and LongLive.*

### vs. Self-Forcing

| Self-Forcing | Reward Forcing Adds |
|--------------|---------------------|
| Solves exposure bias | **Explicit Control** via Re-DMD |
| Enables stable AR rollouts | **Advanced Memory** via EMA-Sink |

### vs. LongLive

| LongLive | Reward Forcing Improves |
|----------|------------------------|
| Real-time streaming with KV-cache | **Dynamic Memory** (EMA-Sink vs static Frame Sink) |
| Static Frame Sink | **Quality & Control** via reward-optimized generation |
| | **Training Robustness** via Self-Forcing integration |

---

## **10. Capabilities Showcase**

![Showcase]({{ '/assets/images/reward_forcing/13_showcase.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 13. Showcase of Reward Forcing capabilities: high-motion dynamics, coherence, and control.*

Reward Forcing enables several breakthrough capabilities:

1. **High-Motion Dynamics**: Rewarded-DMD naturally biases toward energetic, dynamic video content

2. **Ultra-Long Coherence**: EMA-Sink maintains consistency across arbitrarily long sequences

3. **Interactive Control**: Causal generation allows real-time narrative steering

4. **Aesthetic Control**: Custom reward models enable fine-grained style and atmosphere control

---

## **11. Future Directions**

![Future Directions]({{ '/assets/images/reward_forcing/14_future.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 14. Future research directions for Reward Forcing.*

### Multi-Reward Optimization
Balancing competing rewards simultaneously (motion, aesthetics, text-alignment)

### Scaling Laws
Investigating how performance scales with model size, data, and reward model accuracy

### Beyond Pixels
Applying Reward Forcing to:
- Music generation
- Robotics control
- Multi-modal agents

---

## **Conclusion**

**Reward Forcing** represents a watershed moment in video generation research. By synthesizing:

- **DDPMs** for quality foundation
- **DMD** for real-time speed
- **Autoregressive architectures** for streaming
- **Self-Forcing** for robustness
- **EMA-Sink** for long-range memory
- **Rewarded-DMD** for controllable dynamics

...we arrive at a complete solution for generating real-time, controllable, coherent streaming video.

The future of video generation is **streaming**, **controllable**, and **interactive**—and Reward Forcing shows us the path to get there.

---

## **References**

- Ho et al., 2020 - Denoising Diffusion Probabilistic Models
- Nichol & Dhariwal, 2021 - Improved Denoising Diffusion
- Yin et al., 2024 - Distribution Matching Distillation
- Huang et al., 2025 - Self-Forcing
- LongLive, 2025 - Streaming Video with Frame Sink
- VideoAlign, 2025 - Video RLHF
- Rafailov et al., 2024 - Direct Preference Optimization

---
