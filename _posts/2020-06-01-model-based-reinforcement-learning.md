---
layout: post
comments: false
title: "Model-Based Reinforcement Learning (MBRL)"
date: 2020-06-05 17:31:00
tags: 
---

> This post briefly summarizes the niche field of Model Based Reinforcement Learning (MBRL), starting from the basics all the way to the current SOTA algorithms.

<!--more-->

 <!-- For some <b>fantastic</b> notes on Reinforcement Learning, check out Lilian Weng's notes [here](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html).  -->

---
<h3> Contents </h3>

{: class="table-of-content"}
* TOC
{:toc}

---

## The Basics (RL in a nutshell) 

- Any **agent** and its **environment** may be modelled as a high dimensional, discrete time, state-action space. 
- If $$s_t$$ denotes the **state** of the robot, $$u_t$$ denotes the **action** at time $$t$$, then $$s_{t+1}$$ is state of the robot at time $$t+1$$, and then the transition probability can be written as: 
$$ 
P(s_{t+1} | s_t, u_t)
$$. 
- A rule that the robot can follow, to perform the action $$u_t$$ when it is in state $$s_t$$  is called the **policy**, commonly denoted by 
$$
\pi(u_t|s_t)
$$.
- A **reward** or **cost** function indicates how expensive or costly it might be to be in a particular state and take some action. It may be expressed as 
$$
R(s_t, u_t) = \mathbb{E} \big[ R_{t+1} | s_t, u_t \big]
$$.
- A **Value function** then measures the goodness of a state, or how rewarding a state or an action is by prediction of future rewards: $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^T \gamma^k R_{t+k+1}$$, where $$\gamma \in [0,1]$$ is a discount factor which weights near term rewards more than rewards in the distant future.
- There are two sorts of value functions which arise in the RL literature. The first is the **State-Value**. This is the expected return if we are in state $$s_t$$ at time $$t$$: 
$$
V_{\pi}(s) = \mathbb{E} \big[ G_{t} | s_t \big]
$$. 
The other sort of value is the **Action-Value** or the **Q-value** which represents the expected return for a given state and action pair: 
$$
Q_{\pi}(s,u) = \mathbb{E} \big[ G_{t} | s_t, u_t\big]
$$.

<br>


![Illustration of a reinforcement learning problem]({{ '/assets/images/agent_env.png' | relative_url }})
{: style="width: 40%;" class="center"}
*Fig. 1. Basic RL Problem: An agent interacts with with environment (the world) by performing actions. It's environment in turn gives it rewards or penalizes it for poor actions. The corresponding next state of the agent is also returned by the environment (MBRL algorithms use predictive models to generate 'hypothetical next-states'). The agent uses this information to figure out its future actions to maximize its future rewards.*

---

## What is MBRL ?

Reinforcement Learning techniques that use a predictive model (anything from a neural network or a simple regression)  to generate hypothetical training samples of the world.

### Overview of MBRL algorithms

![RL map]({{ '/assets/images/RL_algo_map.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 2. A non-exhaustive map of the major RL algorithms out there.*

The sub-categories (first suggested [here](https://arxiv.org/abs/1907.02057)) haven't been widely estabilshed and are subject to change due to large overlaps between the listed algorithms.

---
*Before heading straight into the MBRL algorithms, it is probably a good idea to look at the important MFRL techniques out there first, since a lot of it will be utilized within MBRL algorithms.*

---

## What about MFRL ?

MFRL (Model-Free RL) techniques bypass the need to model the world. The control policy is learned directly. These algorithms may be broadly categorized as follows:

### 1. On-Policy MFRL

In On-policy algorithms, the training samples are collected according to a target policy which is the same policy that is optimized.

#### 1.1 Actor-Critic (AC) algorithms

The policy and the value function comprise of the two main components of policy gradient algorithm. The value function can be used to assist the policy update, while the learned policy is useful to evaluate the value function. Therefore AC algorithms contain two models:

- **Actor**: learns the <u>policy</u> via model parameters $$\theta$$ of the policy $$\pi_{theta}$$.
- **Critic**: learns the <u>value function</u> through parameters $$\phi$$. (This may be the action-value 
    $$
    Q_{\phi}(a|s)
    $$  or the state-value 
        $$
    V_{\phi}(s)
    $$).

#### 1.2 [TRPO algorithms](https://arxiv.org/abs/1502.05477) 

Gradient updates can potentially update policy parameters drastically which has been shown to decrease the stability of policy gradient algorithms. TRPO mitigates this issue by enforcing a [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) constraint on the size of the policy update.

The TRPO algorithm defines the following [constrained optimization](https://en.wikipedia.org/wiki/Constrained_optimization#:~:text=In%20mathematical%20optimization%2C%20constrained%20optimization,of%20constraints%20on%20those%20variables.) problem:

$$
\underset{\theta}{\text{maximize}} \ \ \mathbb{E}_t  \Bigg( \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)} \Bigg) \hat{A}_t \\
\text{subject to:} \ \ \mathbb{E}_t \Big[ KL( \theta_{old}(.|s_t), \theta(.|s_t) ) \Big] \leq \delta, \; i = 1, \ldots, m.
$$

$$\theta_{old}$$ are the policy parameters before the update. The algorithm uses an *estimated* **advantage function** $$\hat{A}(.)$$. The **constrained optimization** described above enforces the distribution of the policy parameters of the old and new distribution to have a **KL divergence** within a parameter, $$\delta$$. This prevents the new policy from diverging too far from the previous policy distribution.

- **Advantage function**:  $$\hat{A}_t = R_t - b(s_t)$$ 
    - $$R_t$$ is the **expected future returns**, $$R_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} \ r_t$$. 
        - The **Q-value** ($$Q^{\pi}(s,u)$$), i.e, **Action-Value function**, can be used as an estimate of $$R_t$$.
    - $$b(s_t)$$ is the **baseline future rewards**. 
        - The **state-value function** $$V^{\pi}(s)$$ is a good choice for the baseline rewards.
    - Therefore, it together - Estimated Advantage function:  

    $$\hat{A}_t = Q^{\pi}(s,u) - V^{\pi}(s)$$


#### 1.3 [PPO algorithms](https://arxiv.org/abs/1707.06347) 

PPO simplifies the constraint of TRPO with a **clipped surrogate objective** while maintaining similar performance. Let the ratio between the old and new policies be:

$$
\begin{equation}
    \begin{split}
        r(\theta) = \frac{\pi_{\theta}(a|s)}{\pi_{\theta_{old}}(a|s)}
    \end{split}
\end{equation}
$$

Then the TRPO objective function is given by:

$$
\begin{equation}
    \begin{split}
        J^{TRPO}(\theta) = \mathbb{E} \Big[ r(\theta) \ \hat{A}_{\theta_{old}}(s,a) \Big]
    \end{split}
\end{equation}
$$

PPO modifies this by imposing a constraint by forcing $$r(\theta)$$ to remain within $$(1-\epsilon)$$ and $$(1+\epsilon)$$ (where $$\epsilon$$ is a hyperparameter).

$$
\begin{equation}
    \begin{split}
        J^{PPO}(\theta) = \mathbb{E} \Big[ min \Big( r(\theta) \hat{A}_{\theta_{old}}(s,a) , clip(r(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{\theta_{old}}(s,a) \Big) \Big]
    \end{split}
\end{equation}
$$

### 2. Off-Policy MFRL

Off policy methods utilize two different policies for training samples, the **behavior policy** and policy optimization, the **target policy**. This leads to the follwing advantages:
- **Experience Replay** - past episodes can be used to improve sample efficiency.
- **Exploration** - Collecting samples from a the behaviour policy (which is different from the target policy) leads to better exploration.

Some examples of Off-policy algorithms are: 
- Q-learning
- DQN
- DDPG
- TD3
- SAC

## MBRL Algorithms

### Preliminaries

The **State space** $$S \in R^n$$. refers to the state of an agent in its environment. We assume that it is a finite, discretised, n-dimensional space. The a **Action space** $$A \in R^m$$, is similarly defined on an m-dimensional space. The two primary components of any MBRL algorithm are:

- #### Model Learning 

A **Transition function / Model** $$f: SxA \rightarrow S$$ maps any given state of the system and an action performed to the next state. In modern MBRL algorithms, function approximators such as neural networks are commonly used to make a local approximation of the state transition of the model. Lets refer to this **function approximator** as $$\hat{f}_{\phi}$$ parameterized by $$\phi$$. The objective of the model is to learn a function that minimizes the distances between predicted state and the ground truth states:

$$
\begin{equation} 
    \min_{\phi} \sum_{t=0}^T \left\| s_{t+1} - \hat{f_{\phi}}(s_t, a_t) \right\|_2^2
\end{equation}
$$

If denote $$\hat{f_{\phi}}(.)$$ as $$\hat{s}$$, then an alternative measure of distance would be to decouple function approximator from the ground truth trajectory as follows:

$$
\begin{equation} 
    \min_{\phi} \sum_{\ t=1}^T \left\| s_{t+1} - \hat{f_{\phi}}(\hat{s_t}, a_t) \right\|_2^2
\end{equation}
$$

The <u>only difference is that the approximator now relies on its own past prediction for its predictions</u>. This allows us to compare the deviation of the predicted trajectories of our function approximators from the ground truth trajectories.

- #### Policy Learning 

The goal of any MBRL algorithm is to learn a **policy function**, 
$$
\pi_{\theta}(a|s)
$$ that is the probability of taking an action $$a$$ given a state $$s$$, that maximizes the expected sum of future rewards. Here, the **reward function** (or negative Cost function) 
$$
r: SxA \rightarrow R
$$, can be any heuristic function that enables an agent to perform a particular job (for example: walking or following a defined trajectory). The objective function may then be written as the following optimization equation:

$$
\begin{equation}
    \hat{\eta}(\theta ; \phi) := \mathbb{E}_{\hat{\tau}} \Big[ \sum^{T}_{t=0} r(s_t, a_t) \Big]
\end{equation}
$$

where 
$$
\hat{\tau} = (s_0, a_0, s_1, a_1, ...)
$$, 
$$s_0$$ follows an initial state distribution $$ s_0 ~ \rho_0: S \rightarrow R_{+}$$, the sequence of actions is given by the learned policy $$a_t ~ \pi_{\theta}(.|s_t)$$,the states are predicted by the function approximator $$s_{t+1} = \hat{f}_{\phi}(s_t, a_t)$$ over a time horizon $$T$$.

The gradient computation of the objective function (2.3) can be estimated as follows:

$$
\begin{equation}
    \nabla_{\theta} \hat{\eta} = \mathbb{E} \Big[ \nabla_{\theta} \sum^{T}_{t=0} r(s_t, a_t) \Big]
\end{equation}
$$

This gradient is computed across all the time steps across the time horizon, $$T$$ and propagated backwards. This method is called **Backpropagation through time** (BPTT). A simple vanilla MBRL algorithm now be formulated using the aforementioned steps described in Algorithm 1 below.

---
**Algorithm 1: Vanilla Model-Based Deep Reinforcement Learning**
1. Initialize a policy $$\pi_{\theta_0}$$ and a model $$\hat{f_{\phi}}$$
2. Initialize an empty dataset $$D$$.
3. Repeat:
    1. Collect samples from real enviromnemt $$f$$ using $$\pi_\theta$$ and add them to $$D$$.
    2. Train the model $$\hat{f_{\phi}}$$ using $$D$$.
    3. Repeat (until performance stops improving)
        1. Collect fictious samples from $$\hat{f_{\phi}}$$ using $$\pi_\theta$$.
        2. Update the policy using **BPTT** on fictitious samples.
        3. Estimate the performance $$\hat{\eta}(\theta, \phi)$$.

---


### Causes of MBRL performance stagnation

[Wang, Tingwu et al.](https://arxiv.org/abs/1907.02057) highlight three primary challenges that have been known to limit MBRL research from surpassing its MFRL counterparts.

1. **Dynamics Bottleneck**: Algorithms with learned model dynamics are stuck at local minima which is worse than using ground-truth dynamics. Therefore, performance does not increase when more data is collected.
2. **Planning Horizon Dilemma**: Increasing planning horizon provides better rewards estimation, but the performance drops due to modelling errors and the curse of dimensionality.
3. **{Early Termination  Dilemma**: early termination is commponly used in MFRL for more directed exploration, achieving faster learning. Similar performance gains are not yet observed in MBRL algorithms.

<h2> Algorithms </h2>

In the following sections we take a look the latest state of the art MBRL algorithms.  

![MBRL map]({{ '/assets/images/MBRL_map.png' | relative_url }})
{: style="width: 50%;" class="center"}
*Fig. 3. A non-exhaustive map of some of the major MBRL SOTA algorithms out there.*

### 1. Dyna Styled Algorithms

The Dyna architecture, as formulated by [Stutton (1991)](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=711FEF6BA26BBF98C28BC111B26F8761?doi=10.1.1.48.6005&rep=rep1&type=pdf), introduced a **fundamental idea**: **planning by <i>"trying things in your head"</i>**. The general Dyna algorithm alternates between performing RL on real world data and imagined data from a learned world model as follows:

---
**Algorithm 2: Generic Dyna Algorithm**
1. Observe the world's state and reactively choose an action based on it
2. Observe resultant reward and new state.
3. Apply RL to this experience.
4. Update action model based on this experience.
5. Repeat $$k$$ times:
    1. Choose a hypothetical world state and action.
    2. Predict resultant reward and new state using action model.
    3. Apply RL to this hypothetical experience.

---

In the above algorithm, **Steps 1-3** correspond to performing <u>RL on real world data</u>, **Step 4** is when the algorithm <u>learns a world model</u>, and finally **Steps 6-8** is <u>RL performed on the hypothetical world model</u>. As described in the paper, the Dyna framework suffers from the following disadvantages:

1. **Requires Large Memory** - to store backed-up evaluations/reactions associated with each state-action pair
2. **Hierarchical Planning** - Dyna plans at the level of individual actions. Therefore, there is no hierarchy for global planning and optimization.
3. **Ambiguous/Hidden States** - A robot cannot unambiguously determine the world's state as much of it is hidden from it.
4. **Reliance on Supervised Learning** - The traditinal Dyna framework uses a table based approach sufferes form the curse of dimensionality.
5. **Ensuring Variety in Behaviors** - This is the exploration versus exploitation trade off.
6. **Taskability** - Dyna is based on the reward maximization of only one goal. Therefore, in it's original form it is not well suited for multiple tasks.


#### 1.1 (ME-TRPO) [Model Ensemble TRPO](https://arxiv.org/abs/1802.10592) 

Three modifications are made to the Vanilla MBRL Algorithm in ME-TRPO:

1. **Model Ensemble**: An ensemble of dynamic models \{ $$f_{\phi_1},..., f_{\phi_K}$$ \} are used to fit the real world data by
    - training via standard supervised learning.
    - The models only differ by initial weights and the order in which mini-batches are sampled from the dataset.
2. **Policy Optimization**: Utilizing TRPO to optimize policy over the model ensemble.
    - A model is randomly chosen out of the ensemble to predict to the next state.
    - This prevents the policy from over- fitting to any single model.
3. **Policy Validation**: The model ensemble is used to monitor the policy's performance over validation data by:
    - computing a ratio of models in which the policy improves:
$$ 
\frac{1}{K} \sum_{k=1}^{K} \mathbb{1} \Big[ \hat{\eta}(\theta_{new},\phi_k) > \hat{\eta}(\theta_{old},\phi_k) \Big]
$$
    - Training over fictitious data samples continues as long as this computed ratio exceeds a certain threshold.


### 2. Policy Search with BPTT

#### 2.1. [PILCO](https://arxiv.org/abs/1805.00909)

In PILCO, Gaussian Processes (GPs) are used to learn the dynamics of the environment. The policy $$\pi_{\theta}$$ maximizes an objective function via optimization by computing analytic derivatives of the objective with respect to the policy parameters $$\theta$$. 

The core components of PILCO are:

1. **Dynamics Model Learning**

- **inputs**: ($$x_{t-1}$$, $$u_{t-1}$$)$$\in \mathbb{R}^{D+F}$$,
-  **targets**: $$\triangle_{t} = x_t - x_{t-1}+\varepsilon$$ (where $$\varepsilon$$ $$\sim$$ N(0, $$\Sigma_{\varepsilon}$$) and 
- $$\Sigma_{\varepsilon}=diag([\sigma_{\varepsilon_1},...,\sigma_{\varepsilon_D}])$$), 
are used to learn the **one step GP predictions**:

$$
\begin{equation}
    \begin{split}
        p(x_t | x_{t-1}, u_{t-1}) & = \mathbb{N}(x_t | \mu_t, \Sigma_t) \\
        \mu & = x_{t-1} + \mathbb{E}_f [\triangle_t] \\
        \Sigma_t & = var_f [\triangle_t]
    \end{split}
\end{equation}
$$

Given $$n$$ training inputs and targets the posterior GP hyperparameters are learned by **Evidence Maximization**.

2. **Policy Evaluation**

The policy $$\pi$$ is evaluated by computing the expected return:

$$
\begin{equation}
    \begin{split}
        J^{\pi}(\theta) = \sum_{t=0}^T \mathbb{E}_{x_t}[c(x_t)] , \ \ \ \ \ \ \ \ x_0 \sim \mathbb{N} (\mu_0, \Sigma_0) 
    \end{split}
\end{equation}
$$

3. **Policy Improvement**

The policy is improved by computing the gradient of the expected return:

$$
\begin{equation}
    \begin{split}
        \nabla_{\theta} \mathbb{E}_{x_t}[c(x_t)] = \frac{\delta \varepsilon_t}{\delta \theta}
         & = \Big( \frac{\delta \varepsilon_t}{\delta p(x_t)} \Big) \Big( \frac{\delta p(x_t)}{\delta \theta} \Big) \\
         & = \Big( \frac{\delta \varepsilon_t}{\delta \mu_t} \frac{\delta \mu_t}{\delta \theta}+ \frac{\delta \varepsilon_t}{\delta \Sigma_t}\frac{\delta \Sigma_t}{\delta \theta}\Big) \Big( \frac{\delta p(x_t)}{\delta p(x_{t-1})}\frac{\delta p(x_{t-1})}{\delta \theta}+ \frac{\delta p(x_{t})}{\delta \theta}\Big) 
    \end{split}
\end{equation}
$$

The chain rule is applied to further expand $$\frac{\delta \mu_t}{\delta \theta}$$ and $$\frac{\delta p(x_t)}{\delta p(x_{t-1})}$$.


The overall algorithm is summarized below.

---
**Algorithm 3: PILCO Algorithm**
1. **Init:** Sample controller parameters $$\theta \sim \mathbb{N}(0, I)$$ <br> Apply random control signals and record data.
2. Repeat:
    1. Learn probabilistic (GP) dynamics model using all data.
    2. Model-based policy search
    3. Repeat:
        1. Approximate inference for policy evaluation: get $$J^{\pi}(\theta)$$.
        2. Gradient-based policy improvement: get $$\frac{\delta J^\pi (\theta)}{\delta \theta}$$
        3. Update parameters $$\theta$$ (L-BFGS).
    4. Until:  convergence; **return** $$\theta$$
    5. Set $$\pi$$* $$\leftarrow \pi (\theta$*$)$$
    6. Apply $$\pi$$* to system (single trial / episode) and record data.
3. Until:  task is learned.

---

### 3. Shooting Algorithms

#### 3.1. [PETS](https://arxiv.org/abs/1805.12114)

The PETS algorithm uses an ensemble of neural network models to capture and differentiate between epistemic and aleatoric uncertainties in the model's dynamics. 

**Aleatoric Uncertainty**: (inherent system stochasticity) it is representative of the unknowns that differ each time we run teh same experiments.

**Epistemic Uncertainty**: (subjective uncertainty; due to limited data) these due to uncertainties in things one could know in theory, but not in practice. Example: due to measurements inaccuracies, missing data.

The key contributions of the paper are as follows:
1. **Neural Network (NN) Dynamics Models**

    1. **Probabilistic NNS (P)**:
        - NN outputs parameterize a probability distribution 
        - These NNs capture **aleatoric** uncertainty.
        - Loss function: 

        $$
        Loss_{P}(\theta) = -\sum_{n=1}^N log \ \widetilde{f}_{\theta}(s_{n+1} | s_n, a_n)
        $$ (Negative log prediction probability)
    2. **Deterministic NNS (D)**:
        - This can be viewed as a special case of Probabilistic NNs
        - The outputs are delta distributions centered around point predictions.
        - $$
        \widetilde{f}_{\theta}(s_t, a_t)
        $$: $$
        \widetilde{f}_{\theta}(s_{t+1}|s_t,a_t) = \delta (s_{t+1} - \widetilde{f}_{\theta} (s_t, a_t) ) 
        $$
        - Loss function: 

        $$
        Loss_{F}(\theta) = \sum_{n=1}^N | s_{n+1} - f_{\theta}(s_n, a_n) |
        $$ (Mean Squared Error Loss)
    3. **Ensemble NNs**:
        - 'B' models are parameterized by $\theta_b$; $$(b=1,...,B)$$ for model $$\widetilde{f}_{\theta_b}$$.
        - The resulting predictive probability distribution is given by: 
        $$
        \widetilde{f}_{\theta} = \frac{1}{\beta} \sum_{b=1}^\beta \widetilde{f}_{\theta_b}
        $$.
        - Each model is trained via Bootstrapping, that is, uniform random sampling with replacement, from a dataset $$\mathbb{D}$$ to create $$\mathbb{D}_i$$ subsamples.
2. **Trajectory Sampling (TS)**
To predict plausible state trajectories, P particles are propagated from an initial state $$s_0$$ by $$s_{t+1}^p \sim f_{\theta_{b(p,t)}} (s_t^p, a_t)$$ according to a particular bootstrapped model $$b(p,t)$$ in {1,...,B}. The two variants of TS as described by the paper are as follows:
    1. $$\mathbf{TS-1}$$ : particles uniformly resample a bootstrap model every time step.
    2. $$\mathbf{TS-\infty}$$ : particle bootstraps remain the same during a trial.



---
**Algorithm 3: PETS Algorithm**
1. **Init:** data $\mathbb{D}$ with a random controller for on trial.
2. **For** Trial k=1 to K **do**:
    1. Train a PE dynamics model $$\widetilde{f}$$ given $$\mathbb{D}$$.
    2. **For** Trial k=1 to TaskHorizon **do**:
        1. **For** Actions sampled $$a_{t:t+T} \sim CEM(.)$$
            1. Propagate state particles $$s_\tau^p$$ using TS and $$f|\{ \mathbb{D}, a_{t:t+T}\}$$
            2. Evaluate actions as $$\sum_{\tau=t}^{t+\tau} \frac{1}{T} \sum_{p=1}^P r(s_\tau^p, a_\tau)$$
            3. Update CEM(.) distribution.
        2. Execute first action $$a_t$$* form optimal actions $$a_{t:t+\tau}*$$
        3. Record outcome: $$ \mathbb{D} \leftarrow \mathbb{D} \cup \{ s_t, a_t$*$, s_{t+1} \} $$

---

<!-- 
### 4. Unsupervised Algorithms

#### 4.1. [DADS](https://arxiv.org/abs/1907.01657)  -->


## Experiments

### 1. Model Learning

MBRL algorithms mainly prioritize the policy optimization problem when it comes to robot control. 

This section of experiments focuses on the model learning portion of MBLR. For this purpose we attempt to learn the model dynamics of a Double inverted Pendulum, a simple yet hard model to approximate due to its chaotic dynamics. 

![MuJoCo Double Inverted Pendulum]({{ '/assets/images/double_inv_pendulum.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 4. The Double Inverted Pendulum model on a cart constrained in 2 dimensions.*

The model has the following states:

$$
\begin{equation*}
    \begin{split}
        x = \begin{bmatrix}
            x &
            v_x &
            pos_x &
            cos(\theta) &
            sin(\theta) &
            \dot{\theta} &
            cos(\gamma) &
            sin(\gamma) &
            \dot{\gamma} 
            \end{bmatrix}^T
    \end{split}
\end{equation*}
$$

#### Models

<h4> Model1: Baseline model </h4>

A simple linear regression model is used to establish a baseline for prediction model states. 

![Baseline model]({{ '/assets/images/NN_model1.png' | relative_url }})
{: style="width: 35%;" class="center"}
*Fig. 5. Simple linear regression model used as the Baseline model.*

This model performs a linear transformation from $\mathbb{R}^{10}$ to $\mathbb{R}^9$.

<h4> Model2: Baseline Neural Network (NN) model1 </h4>

A secondary neural network model comprising of five densely connected layers with a ReLU non-linearity in-between the layers is also tested. 

![Model2]({{ '/assets/images/NN_model2.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 6. A fully connected Neural network model used as a secondary model.*

Both the models so far accept 10 input features comprising of 9 states , i.e, the current state of the robot, and  and 1 action  on the cart. They output 9 features  representing the next predicted state.

<h4> Model3: NN model2 </h4>

This model has the same architecture as Model 2, but instead of predicting the next state, it predicts the change in state ($$\Delta s_t$$) that takes the model from the current state to the next state.


#### Loss functions

We use the following trajectory roll out Mean Squared Error (MSE) Loss functions for the experiments (described in the [model learning section](https://crohan009.github.io/rolog/2020/06/05/model-based-reinforcement-learning.html#model-learning)):


$$
\begin{equation} 
    Loss_{1} =  \sum_{t=0}^T \left\| s_{t+1} - \hat{f_{\phi}}(s_t, a_t) \right\|_2^2
\end{equation}
$$

This is the basic Mean Squared Error loss that takes the MSE error between ever single predicted state and its ground truth state. 


$$
\begin{equation} 
    Loss_{2} = \sum_{\ t=1}^T \left\| s_{t+1} - \hat{f_{\phi}}(\hat{s_t}, a_t) \right\|_2^2
\end{equation}
$$

This MSE loss is generalized so that we may be able to perform a prediction of multiple states for a specified prediction time horizon. The only difference here is that the next predicted is dependent on the previous predicted state. 

<h4> Baseline Model 1 Predictions </h4>

![AvsP_plots_cos_action_model1]({{ '/assets/images/AvsP_plots_cos_action_model1.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 7. Predicted state comparison for multi-istep MSE loss function (Type 2). The figures show the actual and predicted states of Model1 that correspond to a 100 step trajectory with a cosine action at each step. The baseline model1 was pre-trained on null actions was then re-trained on trajectories in which the action was a cosine function of the time step.*


<h4> NN Model 1 Predictions </h4>

![AvsP_plots_cos_action_model2]({{ '/assets/images/AvsP_plots_cos_action_model2.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 8. Predicted state comparison for multi-istep MSE loss function (Type 2). The figures show the actual and predicted states of Model2 that correspond to a 100 step trajectory with a cosine action at each step. Model2 was pre-trained on null actions was then re-trained on trajectories in which the action was a cosine function of the time step.*

<h4> NN Model 2 Predictions </h4>

![AvsP_plots_cos_action_model3]({{ '/assets/images/AvsP_plots_cos_action_model3.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 9. Predicted state comparison for multi-istep MSE loss function (Type 2). The figures show the actual and predicted states of Model3 that correspond to a 100 step trajectory with a cosine action at each step. Model3 was pre-trained on null actions was then re-trained on trajectories in which the action was a cosine function of the time step.*

![A-P_plots_cos_action_model3]({{ '/assets/images/A-P_plots_cos_action_model3.png' | relative_url }})
{: style="width: 100%;" class="center"}
*Fig. 10. The figure illustrates the drift in the predicted states. The plots how the absolute difference between predicted and actual states over 100 step trajectory with a cosine action at each step.*

<i>**(to be continued)**</i>

<!-- #### Prediction difference plots

<h4> Baseline Model 1 |Actual - Predicted| </h4>

<h4> NN Model 1 Predictions |Actual - Predicted| </h4>

<h4> NN Model 2 Predictions |Actual - Predicted| </h4> -->


<!-- ## References


[1] 

--- -->

<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

*Please don't hesitate to contact me with any errors. I'll be sure to correct them right away !*