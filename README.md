# Reinforcemnt Learning for OpenAI Gym

This repo contains several implementations of reinforcement learning techniques for OpenAI Gym environments. The files are broken up by the class of algorithm, each containing one or more different strategies. Can be run by calling `python [filename] --FLAGS`. Requirements are mainly Gym, Numpy, and Matplotlib for plotting. Classes of algorithms and strategies are listed below:

* Dynamic Programming
    * Value Iteration
    * Policy Iteration
* Monte Carlo
    * First Visit Epsilon-Greedy
    * Every Visit Epsilon-Greedy
    * Epsilon-Greedy with Importance Sampling
* Temporal Difference
    * SARSA
    * Q Learning
* Evolutionary Algorithms
    * Augmented Random Search

# TODO

* Improved logging and plotting
* Deep Q Networks
* Policy Gradients