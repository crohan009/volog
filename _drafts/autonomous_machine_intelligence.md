---
layout: post
title:  "Designing Autonomous General Intelligence"
date:   2023-01-01
categories: AI
tags: AI, machine-intelligence
---

>  A design for autonomous machine intelligence - a look at certain key ideas on how machines can learn to learn representations, reason, plan, and interact with the world. 

<!--more-->

---
<h3> Contents </h3>

{: class="table-of-content"}
* TOC
{:toc}

---

# Designing Autonomous General Intelligence

**Paper Review**  
[A Path Towards Autonomous Machine Intelligence (6/27/22)](https://scholar.google.com/scholar?q=A%20Path%20Towards%20Autonomous%20Machine%20Intelligence%20Version%200.9.2%2C%202022-06-27) 
- Yann Lecun


## Background
Humans and a few other animals exhibit a generalized learning ability. For example, humans can learn to drive a car with approximately 20 hours of practice, demonstrating sample efficiency. They can also figure out how to act in a situation never encountered before, showcasing adaptive transfer learning. Intelligence can be defined as the "ability to accomplish complex goals". This definition is broad enough to include all definitions since understanding, self-awareness, problem solving, learning, planning, creativity, capacity for logic, etc are all examples of complex goals that one might have. The three challenges in General AI research today are: learning to represent the world, to predict and to act largely by observation, reasoning and planning in ways that are compatible with gradient-based learning, and representing percepts & action plans in a hierarchical manner — multi-level abstraction at multiple levels of abstraction and at multiple time scales.


## Main contributions of the paper
The paper presents an overall cognitive architecture in which all modules are differentiable and many of them are trainable. It introduces a non-contrastive self-supervised learning paradigm that produces representations that are simultaneously informative and predictable. The paper also proposes JEPA and Hierarchical JEPA, non-generative architectures for predictive world models that learn a hierarchy of representations. It also suggests a way to use H-JEPA as the basis of predictive world models for hierarchical planning under uncertainty. The paper proposes an architecture for intelligent agents and provides possible solutions to the three challenges.

## Model Architecture for Autonomous Intelligence
The architecture consists of several modules including the Configurator (CM), Perception (PM), World Model (WMM), Cost (RM), Intrinsic Cost (ICM), Critic (TCM), Short-Term Memory (STMM), and Actor (AM).

### Typical Perception-Action Loops
#### Mode-1: Reactive Behavior
In this mode, the perception module, PM (through an encoder module), extracts a representation of the state of the world s[0] = Enc(x). The policy module, a component of the actor, produces an action as a function of the state a[0] = A(s[0]). The cost module, CM is differentiable, its output f[0] = C(s[0]) is indirectly influenced by previous actions through the external world. In this mode, gradients of the cost f[[0] with respect to actions can only be estimated by polling the world with multiple perturbed actions, but that is slow and potentially dangerous. This mode corresponds to classical policy-gradient methods in RL.

#### Mode-2: Reasoning and Planning using World Model
In this mode, the PM computes s[0] = P(x), and the CM computes and stores C(s[0]). The AM proposes (a[0], . . . , a[t], . . . , a[T]) to be fed to the WMM for evaluation. The WMM predicts one or several likely sequence of world state representations from the proposed action sequence (s[1], . . . , s[t], . . . , s[T]). The CM estimates a total cost of the predicted state sequence: F(x) = ∑t=1 C(s[t]). The AM proposes a new action sequence with lower cost. After converging on a low-cost action sequence, the AM sends the first action (or first few actions) in the low-cost sequence to the effectors. After every action, the states and associated costs from the intrinsic cost and the critic are stored in the short-term memory. This procedure is also known as Model-Predictive Control (MPC) with receding horizon in the optimal control literature.

#### Mode-2 to Mode-1: Learning New Skills
Mode 2 is onerous, so we can run on Mode-2, produce an optimal action sequence (a[0], . . . , a[t], . . . , a[T]), and the parameters of the policy module: A(s[t]) are updated to minimize a divergence measure between its output and the optimal action at that time D(a[t], A(s[t])). Once properly trained, the policy module can be used to directly produce an action in Mode-1 a[0] = A(s[0]). The policy module can be seen as performing a form of amortized inference. This process allows the agent to use the full power of its world model and reasoning capabilities to acquire new skills that are then “compiled” into a reactive policy module that no longer requires careful planning.

#### Reasoning as Energy Minimization
The process of elaborating on a suitable action sequence in Mode-2 can be seen as a form of “reasoning” based on simulation using the world model, and optimization of the energy with respect to action sequences.

### The Cost Module as the Driver of Behavior
The CM is composed of the ICM module which is immutable ICi(s) and the Critic (TCM) or Trainable Cost TCj(s), which is trainable. Each submodule imparts a particular behavioral drive to the agent. The weights ui and vj are modulated by the CM, allowing the agent to focus on different subgoals at different times. The ICM is where the basic behavioral nature of the agent is defined (amygdala). It measures things like “pain”, “hunger”, and “instinctive fears”, such as external, force overloads, dangerous electrical, chemical, or thermal environments, excessive power consumption, low levels of energy reserves in the power source, etc. The role of the critic (TC) is to anticipate long-term outcomes with minimal use of the onerous world model and to allow the configurator to make the agent focus on accomplishing subgoals with a learned cost.

### Training the Critic
The critic can be trained to predict future intrinsic energy values by retrieving a past state vector sτ and an intrinsic energy at a later time IC(sτ+δ). The parameters of the critic can then be optimized to minimize a prediction loss: ||IC(sτ+δ) − TC(sτ)||2. At a general level, this is similar tocritic training methods used in such reinforcement learning approaches as A2C.

## Designing and Training the World Model
Designing architectures and training paradigms for the world model constitute the main obstacles towards real progress in AI over the next decades. The quality of the world model will greatly depend on the diversity of state sequences, or triplets of (state, action, resulting state) it is able to observe while training. The world model must be able to meaningfully represent this possibly-infinite collection of plausible predictions. The world model must be able to make predictions at different time scales and different levels of abstraction.

### Self-Supervised Learning (SSL)
SSL is a learning paradigm in which a learning system is trained to capture the mutual dependencies between its inputs. Importantly, we do not impose that the model be able to predict y from x. The reason is that there may be an infinite number of y that are compatible with a given x. The general formulation can be done with the framework of Energy-Based Models (EBM): A scalar-valued function F(x, y) that produces low energy values when x and y are compatible and higher values when they are not.

### Handling Uncertainty with Latent Variables
A latent variable is an input variable whose value is not observed but inferred. It is used to represent information about y that cannot be extracted from x. A latent-variable EBM (LVEBM) is a parameterized energy function that depends on x, y, and z: Ew(x, y, z). When presented with a pair (x, y) the inference procedure of the EBM finds a value of the latent variable z that minimizes the energy.

### Training Energy-Based Models
There are two methods for training EBMs: Contrastive Methods and Regularized Methods. Contrastive Methods push down on the energy of training samples (blue dots) and pulls up on the energies of suitably-placed contrastive samples (green dots). Regularized Methods push down on the energy of training samples and use a regularizer term that minimizes the volume of low-energy regions. This regularization has the effect of “shrink-wrapping” the regions of high data density within the low-energy regions, to the extent that the flexibility of the energy function permits it.

### Joint Embedding Predictive Architecture (JEPA)
JEPA is an architecture where two variables x and y are fed to two encoders producing two presentations sx and sy. The predictor may depend on a latent variable, z. The energy is simply the prediction error in representation space. The overall energy is obtained by minimizing over z. The main advantage of JEPA is that it performs predictions in representation space.

### Training a JEPA
The regularized JEPA training criteria are to maximize the information content of sx about x, maximize the information content of sy about y, make sy easily predictable from sx, and minimize the information content of the latent variable z used in the prediction.

### Hierarchical JEPA (H-JEPA)
H-JEPA is an extension of the architecture to handle prediction at multiple time scales and multiple levels of abstraction. Low level representations contain a lot of details about the input and can be used to predict in the short term. High-level, abstract representation may enable long-term predictions, at the cost of eliminating a lot of details.

<!-- ### Hierarchical Planning

The paper proposes an architecture for **intelligent agents** that can handle *hierarchical planning*. The architecture includes a *world model module* that predicts possible future world states based on imagined action sequences proposed by the *actor module*. This world model module is a kind of "simulator" of the relevant aspects of the world. The predictions are performed within an abstract representation space that contains information relevant to the task at hand. Ideally, the world model would manipulate representations of the world state at multiple levels of abstraction, allowing it to predict over multiple time scales.

The *actor module* computes proposals for sequences of actions and outputs actions to the effectors. The actor proposes a sequence of actions to the world model. The world model predicts future world state sequences from the action sequence, and feeds it to the cost. Given a goal defined by the cost (as configured by the configurator), the cost computes the estimated future energy associated with the proposed action sequence. Since the actor has access to the gradient of the estimated cost with respect to the proposed action sequence, it can compute an optimal action sequence that minimizes the estimated cost using gradient-based methods. If the action space is discrete, dynamic programming may be used to find an optimal action sequence. Once the optimization is completed, the actor outputs the first action (or a short sequence of actions) to the effectors. This process is akin to *model-predictive control* in optimal control. -->

<!-- ### Handling Uncertainty

The paper also discusses how to handle uncertainty in the predictions of the world model. The world model must be able to represent multiple possible predictions of the world state. The natural world is not completely predictable. This is particularly true if it contains other intelligent agents that are potentially adversarial. But it is often true even when the world only contains inanimate objects whose behavior is chaotic, or whose state is not fully observable. The paper suggests that the world model should be able to represent multiple plausible predictions and represent uncertainty in the predictions.


### Keeping track of the state of the world
A typical action of an agent will only modify a small portion of the state of the world. This suggests that the state of the world should be maintained in some sort of writable memory. A conventional key-value associative memory can be used for this purpose. The output of the world model at a giventime step is a set of query-value pairs (q[i], v[i]), which are used to modify existing entries in the world-state memory, or to add new entries. -->

### Data Streams
An agent can learn about how the world works through five modes of information gathering: Passive observation, Active fovenation, Passive agency, Active egomotion, and Active agency. Modes 2, 4, and 5 allow the agent to collect information that maximizes its understanding of the environment. The main open question is how much can be learned using passive observation (modes 1, 2, 4), how much requires egomotion (mode 3), and how much requires full agency (mode 5).

## Designing and Training the Actor
The role of the actor module is threefold: inferring optimal action sequences that minimize the cost, producing multiple configurations of latent variables that represent the portion of the world state the agent does not know, and training policy networks for producing Mode-1 actions.

## Designing the Configurator
The Configurator is necessary for two reasons: Hardware reuse and Knowledge sharing. Its most important function is to set subgoals for the agent and configure the cost module for this subgoal. This is an area open for future investigation.

## Related Work
The related work includes Trained World Models, Model-Predictive Control, Hierarchical Planning, Energy-Based Models and Joint-Embedding Architectures, and Human and animal cognition.

## Discussion, Limitations, Broader Relevance
There are several aspects missing from the Proposed Model, such as whether a Hierarchical JEPA can be built and trained from videos, how precisely to regularize the latent variable so as to minimize its information content, and the current proposal does not prescribe a particular way for the actor to infer latent variable instantiations and optimal action sequences. The question remains whether the type of reasoning proposed here can encompass all forms of reasoning humans are capable of. The paper also discusses the broader relevance of the proposed approach, whether this architecture could be the basis of a model of animal intelligence, and whether this could be a path towards machine common sense. The paper concludes that scaling is not enough and reward is not enough for training the world model. It can be done by taking actions and getting rewards, and by predicting the world state.
