# I See You — _Robust Measurement of Adversarial Behavior_

This repository contains the code to download the raw blockchain data and run the analysis in the paper "I See You—_Robust Measurement of Adversarial Behavior_."

## Code

This readme assumes that the folder containing the code has been downloaded and unzipped.

Ensure you've avigated into the repo with

```bash
cd surveillance-metric
```

Now, install the required packages to run this code by following the below steps.

Preferably, activate a virtual environment before installing anything, you can use anything, but we'll show it with `venv` here.

```bash
python -m venv venv
source venv/bin/activate
```

You can alternatively use, e.g., `conda` if you have that installed

```bash
conda create -n defi python=3.11 -y
conda activate defi
```

Then, install this package and its dependencies

```bash
pip install -e .
```


## Download input data data

We have aggregated raw data from different sources to enable the analyses herein, all intended for academic use:

- Uniswap data is from the service Allium: [https://app.allium.so/](https://app.allium.so/)
- MEV Boost data is from an open repo: [https://mevboost.pics/data.html](https://mevboost.pics/data.html)

To download all the data used in this study, run the following commands in the root of the repository

```bash
curl -O https://surveillance-metric.s3.amazonaws.com/input_data.zip
unzip input_data.zip
rm input_data.zip
```

You should now have a folder in the root of the project folder called `input_data` with several more folders inside.

This is the necessary data to run the code that calculates the metric using blockchain data. If you only want to look at the result after running the analyses, rather download the following data.



## Calculate the metrics

To run the code that populates the data frames that we later run analyses on, run

```bash
python -m surveillance_metric.calculate_surveillance_metric
```

This assumes the



## Download metric calculation output data

To download the output from running the metric calculation scr


```bash
curl -O https://surveillance-metric.s3.amazonaws.com/metric_calculation_output.zip
unzip metric_calculation_output.zip
rm metric_calculation_output.zip
```



## Run the analyses on the metrics
