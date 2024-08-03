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
```

Now install the simulator code

```bash
pip install -e uniswap-v3-sim
```

Finally, install this package and its dependencies

```bash
pip install -e .
```


## Data


### Raw data

We have aggregated raw data from different sources to enable the analyses herein. Most of the Uniswap data is from the service Allium, MEV Boost data is from an open repo: .

To download the raw data, run the following commands in the root of the repository (planning to add the opportunity to specify a different path if that's nice to have)

```bash
curl -O https://surveillance-metric.s3.amazonaws.com/uniswap-raw-data.zip
unzip uniswap-raw-data.zip
rm uniswap-raw-data.zip
```

This is the necessary data to be able to run the code that calculates the metric over the blockchain data. If you only want to look at the result after running the analyses, download the following data as well.


### Metric calculation data

_NOTE: This file does not exist yet!_

```bash
curl -O https://surveillance-metric.s3.amazonaws.com/uniswap-surveillance-metric-v1.zip
unzip uniswap-surveillance-metric-v1.zip
rm uniswap-surveillance-metric-v1.zip
```


## Calculate the metrics

To run the code that populates the dataframes that we later run analyses on, run

```bash
python -m surveillance_metric.computation_scripts.calculate_surveillance_metric
```

## Run the analyses on the metrics
