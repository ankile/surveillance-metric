#!/bin/bash
#SBATCH --job-name=mev_boost_metric
#SBATCH --output=mev_boost_metric_%A_%a.out
#SBATCH --error=mev_boost_metric_%A_%a.err
#SBATCH --array=0-128
#SBATCH --partition=xxx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=20GB
#SBATCH --time=3-00:00:00

# Set the base command
BASE_CMD="python -m surveillance_metric.mev_boost_data_metric"

# Run the command with the correct arguments
$BASE_CMD --n-partitions 128 --partition-index $SLURM_ARRAY_TASK_ID


