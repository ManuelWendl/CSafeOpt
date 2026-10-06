# Cumulative SafeOpt -- No Regret Safe Bayesian Optimization

This repository contains the code for the gated exploration mechanism CSafeOpt. It extends the implementation
of GoSafeOpt (see [Acknowledgement](#acknowledgement)).

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

## Statistical evaluation

The statistical evaluation compares CSafeOpt with CSafeOptSimple, SafeOpt, SafeUCB, GoOSE, ISE-BO and StageOpt over 10 safe initial configurations per benchmark. All results and the figure are written to `examples/statistical_evaluation/`. Install the example dependencies first:

```
poetry install --with examples
```

**1. Double bottleneck, mirage, mountain car and pendulum** (10 starts within 0.1 of the domain width around each benchmark's nominal safe seed; 100, 200, 200 and 800 rounds):

```
python examples/multistart.py \
  --benchmarks double_bottleneck --benchmarks mirage --benchmarks mountaincar --benchmarks pendulum \
  --algorithms CSafeOpt --algorithms CSafeOptSimple --algorithms SafeOpt --algorithms SafeUCB \
  --algorithms GoOSE --algorithms ISE-BO --algorithms StageOpt \
  --out-dir examples/statistical_evaluation
```

**2. Mars** (10 landing sites within 2 m of the nominal landing site, 400 rounds). The first command samples the landing sites and runs CSafeOpt; the second runs the baselines at the same sites (each Mars run needs about 9 GB of memory, so lower `--jobs` if necessary):

```
python examples/mars_demo.py random-landings --algorithms CSafeOpt \
  --out examples/statistical_evaluation/mars.png
python examples/mars_demo.py random-landings-extend --out examples/statistical_evaluation/mars.png \
  --algorithms CSafeOptSimple --algorithms SafeOpt --algorithms SafeUCB --algorithms GoOSE \
  --algorithms ISE-BO --algorithms StageOpt --jobs 4
```

**3. Figure** (mean cumulative regret with one standard error over the starts):

```
python examples/multistart_bars.py --results-dir examples/statistical_evaluation \
  --replace mars_near_nominal=examples/statistical_evaluation/mars.csv \
  --out examples/statistical_evaluation/regret_bars.png
```

## Acknowledgement

This code extends the implementation of GoSafeOpt by Sukhija et al. 

```
@article{sukhija2023gosafeopt,
  title={GoSafeOpt: Scalable safe exploration for global optimization of dynamical systems},
  author={Sukhija, Bhavya and Turchetta, Matteo and Lindner, David and Krause, Andreas and Trimpe, Sebastian and Baumann, Dominik},
  journal={Artificial Intelligence},
  volume={320},
  year={2023},
  publisher={Elsevier}
}
```
