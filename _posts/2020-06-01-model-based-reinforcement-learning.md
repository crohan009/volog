---
layout: post
comments: false
title: "Model-Based Reinforcement Learning"
date: 2020-06-05 17:31:00
tags: model-based-reinforcement-learning review long-read
---

> In this post, we are briefly going to go over Model Based Reinforcement Learning (MBRL), starting the basics all the way to the current SOTA algorithms in the MBRL literature.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## What is Reinforcement Learning?

Say, we have an agent in an unknown environment and this agent can obtain some rewards by interacting with the environment. The agent ought to take actions so as to maximize cumulative rewards. In reality, the scenario could be a bot playing a game to achieve high scores, or a robot trying to complete physical tasks with physical items; and not just limited to these.


![Illustration of a reinforcement learning problem]({{ '/assets/images/RL_illustration.png' | relative_url }})
{: style="width: 70%;" class="center"}
*Fig. 1. An agent interacts with the environment, trying to take smart actions to maximize cumulative rewards.*

### Key Concepts

Now Let's formally define a set of key concepts in RL.

The agent is acting in an **environment**.How the environment reacts to certain actions is defined by a **model** which we may or may not know. The agent can stay in one of many **states** ($$s \in \mathcal{S}$$) of the environment, and choose to take one of many **actions** ($$a \in \mathcal{A}$$) to switch from one state to another. Which state the agent will arrive in is decided by transition probabilities between states ($$P$$). Once an action is taken, the environment delivers a **reward** ($$r \in \mathcal{R}$$) as feedback. 

The model defines the reward function and transition probabilities. We may or may not know how the model works and this differentiate two circumstances:
- **Know the model**: planning with perfect information; do model-based RL. When we fully know the environment, we can find the optimal solution by [Dynamic Programming](https://en.wikipedia.org/wiki/Dynamic_programming) (DP). Do you still remember "longest increasing subsequence" or "traveling salesmen problem" from your Algorithms 101 class? LOL. This is not the focus of this post though. 
- **Does not know the model**: learning with incomplete information; do model-free RL or try to learn the model explicitly as part of the algorithm. Most of the following content serves the scenarios when the model is unknown.

The agent's **policy** ($$\pi(s)$$) provides the guideline on what is the optimal action to take in a certain state with <span style="color: #e01f1f;">**the goal to maximize the total rewards**</span>. Each state is associated with a **value** function ($$V(s)$$) predicting the expected amount of future rewards we are able to receive in this state. In other words, the value function quantifies how good a state is. Both policy and value functions are what we try to learn in reinforcement learning. 


![Categorization of RL Algorithms]({{ '/assets/images/RL_algorithm_categorization.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 2. Summary of approaches in RL based on whether we want to model the value, policy, or the environment. (Image source: reproduced from David Silver's RL course [lecture 1](https://youtu.be/2pWv7GOvuf0).)*


The interaction between the agent and the environment involves a sequence of actions and observed rewards in time, $$t=1, 2, \dots, T$$. During the process, the agent accumulates the knowledge about the environment, learns the optimal policy, and makes decisions on which action to take next so as to efficiently learn the best policy. Let's label the state, action, and reward at time step t as $$S_t$$, $$A_t$$, and $$R_t$$, respectively. Thus the interaction sequence is fully described by:

$$
S_1, A_1, R_2, S_2, A_2, \dots, S_T
$$

$$S_T$$ is the terminal state.


Terms you will encounter a lot when diving into different categories of RL algorithms:
- **Model-based**: Rely on the model of the environment; either the model is known or the algorithm learns it explicitly.
- **Model-free**: No dependency on the model during learning.
- **On-policy**: Use the deterministic outcomes or samples from the target policy to train the algorithm.
- **Off-policy**: Training on a distribution of transitions or episodes produced by a different behavior policy rather than that produced by the target policy.


## References

[1] Yuxi Li. [Deep reinforcement learning: An overview.](https://arxiv.org/pdf/1701.07274.pdf) arXiv preprint arXiv:1701.07274. 2017.

---

*If you notice mistakes and errors in this post, please don't hesitate to contact me and I'll be sure to correct them right away !*