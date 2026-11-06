# binomial-trinomial-option-pricing
Implementation and numerical analysis of CRR binomial and Boyle trinomial models for option pricing, with convergence studies and Richardson extrapolation.

# Overview

This project explores numerical methods for option pricing using recombining binomial and trinomial trees.

It focuses on two models:

Cox-Ross-Rubinstein (CRR) model: a binomial tree based on two possible movements of the underlying asset at each time step.

Boyle model: a trinomial tree that introduces three possible movements, providing an alternative numerical approximation.
The project investigates their pricing accuracy, convergence behavior, and the use of Richardson extrapolation to accelerate convergence toward the Black-Scholes benchmark.

# Mathematical Framework

The models are developed under the risk-neutral framework.

The underlying asset is assumed to follow a geometric Brownian motion:

[
dS_t = rS_t,dt + \sigma S_t,dW_t
]

where:
(S_t) is the underlying asset price.

(r) is the risk-free interest rate.

(\sigma) is the volatility.

(W_t) is a standard Brownian motion.

The Black-Scholes model is used as a reference for evaluating the numerical accuracy of the tree-based methods.

# Implemented Models

# 1. CRR Binomial Model

The CRR model represents the evolution of the underlying asset through two possible movements at each time step:

[
u=e^{\sigma\sqrt{\Delta t}},
\qquad
d=\frac{1}{u}
]

The option price is obtained through backward induction using risk-neutral probabilities.

The implementation considers:

European option pricing.

American option pricing.

Numerical convergence toward the Black-Scholes price.

# 2. Boyle Trinomial Model

The Boyle model extends the binomial framework by introducing three possible movements at each time step:

Upward movement.

No movement.

Downward movement.

The model uses a recombining trinomial tree and risk-neutral transition probabilities.

Its numerical behavior is compared with the CRR model, particularly in terms of convergence and pricing accuracy.

# 3. Richardson Extrapolation

Richardson extrapolation is implemented to improve numerical accuracy by combining estimates obtained with different time-step discretizations.

The objective is to reduce discretization error and accelerate convergence toward the theoretical option price.

# Numerical Experiments

The project includes numerical experiments to:

Compare CRR and Boyle option prices.

Study convergence as the number of time steps increases.

Analyze pricing errors relative to the Black-Scholes benchmark.

Evaluate the impact of Richardson extrapolation.

Compare the convergence behavior of binomial and trinomial trees.

# Learning Objectives

This project aims to develop a deeper understanding of:

Numerical option pricing.

Risk-neutral valuation.

Binomial and trinomial recombining trees.

Backward induction.

Numerical convergence and discretization errors.

Richardson extrapolation.
