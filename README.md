# Cumulative SafeOpt -- No Regret Safe Bayesian Optimization

![Bayesian Optimization animation](/doc/animation.gif?raw=true "Bayesian optimization animation")

This repository contains the code for the gated exploration mechanism CSafeOpt. 

## Setup

```
#With poetry
poetry install

#With pip/venv
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

It might be necessary to create a wandb account at wandb.ai if not already existing.

## Examples

To master the mars exploration problem with the gated aquisition function and comparison to all baselines, run the following commands:

```
poetry install --with examples
python examples/mars_demo.py
```
