#!/bin/bash
#SBATCH -p vision-pulkitag-a6000
#SBATCH -q vision-pulkitag-free-cycles
#SBATCH --job-name=mev_boost_metric
#SBATCH --output=mev_boost_metric_%A_%a.out
#SBATCH --error=mev_boost_metric_%A_%a.err
#SBATCH --array=0-15
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=08:00:00

# Load any necessary modules or activate virtual environment here
# For example:
# module load python/3.8
# source /path/to/your/venv/bin/activate
cd /tmp/ankile/surveillance-metric-2
# conda activate defi

# Set the base command
BASE_CMD="python -m surveillance_metric.computation_scripts.mev_boost_data_metric -n 16"

# Run the command with the current array task ID as the -i parameter
$BASE_CMD -i $SBATCH_ARRAY_TASK_ID

# Deactivate virtual environment if used
# deactivate