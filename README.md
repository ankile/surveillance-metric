# I See You — _Robust Measurement of Adversarial Behavior_

This repository contains the code to download the raw blockchain data and run the analysis in the paper "I See You—_Robust Measurement of Adversarial Behavior_."

## Code

Navigate to where you want to house this code and run this command

```bash
git clone --recursive git@github.com:ankile/surveillance-metric.git
```

_Note: The `recursive` option is important to ensure that you get the code that runs the Uniswap simulation_

Navigate into the repo with

```bash
cd surveillance-metric
```

Now, install the required packages to run this code by following the below steps.

Preferably, activate a virtual environment before installing anything, you can use anything, but we'll show it with `venv` here.

```bash
python -m venv venv
source venv/bin/activate
``

Now install the simulator code

```bash
pip install -e uniswap-v3-sim
```

Finally, install this package and its dependencies

```bash
pip install -e .
```


## Data



## Calculate the metrics

## Run the analyses on the metrics
