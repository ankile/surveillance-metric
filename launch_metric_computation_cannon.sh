#!/bin/bash
#SBATCH --job-name=mev_boost_metric
#SBATCH --output=mev_boost_metric_%A_%a.out
#SBATCH --error=mev_boost_metric_%A_%a.err
#SBATCH --array=0-128
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=20GB
#SBATCH --time=3-00:00:00
#SBATCH --account=parkes_low_priority


# Print all environment variables
env


# Activate your virtual environment if you're using one
# source /path/to/your/venv/bin/activate

# Set the base command
BASE_CMD="python -m surveillance_metric.computation_scripts.mev_boost_data_metric"

# Print the command that will be executed
echo "Command to be executed:"
echo "$BASE_CMD --n-partitions 16 --partition-index $SLURM_ARRAY_TASK_ID"

# Run the command with the correct arguments
$BASE_CMD --n-partitions 128 --partition-index $SLURM_ARRAY_TASK_ID


