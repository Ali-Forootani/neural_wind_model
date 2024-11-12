#!/bin/bash

#SBATCH --job-name=my_wind_train_hybrid_job
#SBATCH --time=0-70:30:00
#SBATCH --gres=gpu:nvidia-a100:1
#SBATCH --mem-per-cpu=32G
#SBATCH --constraint a100-vram-80G
#SBATCH --output="/data/bio-eng-llm/llm_repo/python_repos/testing_eve_job/$USER/my_python_job/logs/%x-%j.log"

# Change directory within the script


# Load any necessary modules, if require
source /data/bio-eng-llm/virtual_envs/dnn_env/bin/activate
module load python

# Execute the Python script
python /data/bio-eng-llm/ReSTEP/training_wind_psr_lstm_transform.py

