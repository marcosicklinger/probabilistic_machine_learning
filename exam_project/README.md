# Probabilistic Machine Learning exam project

### Description

This repositories contains the modules implemented to complete the final exam project for the Probabilistic Machine Learning course of the Data Science and Scientific Computing master degree course at the University of Trieste (a.y. 2020/2021).

The problem to be solved is the following one: approximate, by means of gaussian processes the action value function of the goal search RL problem.

More details in the pdf presentation.

### Structure

The repository contains eight python files:

- ***Agents.py***

  where agents classes are defined

- ***Environments.py***

  where environments classes are defined (both discrete and continuous frameworks)

- ***Utils.py***

  where some utility functions are defined

- ***model.py***, ***DGPQModel.py***

  where discrete (used for comparison with Q-learning tabular algorithm) and continuous (GPs framework) models are defined, respectively

- ***plot_times.py***, ***render.py***, ***policy.py***

  where tools for plotting time performances, rendering episodes' animations and plotting final policies are defined