#!/bin/bash

#SBATCH --job-name=my_wind_load_job
#SBATCH --time=0-01:30:00
#SBATCH --gres=gpu:nvidia-a100:1
#SBATCH --mem-per-cpu=16G
#SBATCH --output="/data/bio-eng-llm/llm_repo/python_repos/testing_eve_job/$USER/my_python_job/logs/%x-%j.log"

# Change directory within the script


# Load any necessary modules, if require
module load Anaconda3
source activate /data/bio-eng-llm/miniconda3/envs/gnsindy_3
module load python

# Execute the Python script
python /data/bio-eng-llm/ReSTEP/wind_load_gpu_to_cpu.py