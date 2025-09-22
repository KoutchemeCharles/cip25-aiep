#!/bin/bash
#
#SBATCH --job-name=cpu
#
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#
#SBATCH --chdir=SET_THE_PATH/cip25-aiep
#SBATCH --output=SET_THE_PATH/cip25-aiep/logs/%A.log

code_folder="SET_THE_PATH/cip25-aiep"
export PYTHONPATH=$code_folder
export HF_HOME=SET_THE_PATH
export OPENAI_API_KEY=""
export HF_API_TOKEN=""
export HF_TOKEN=""
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=
export WANDB_INIT_TIMEOUT=120

source PATH
conda activate diag

base_path=$code_folder/config/experiments/$folder/
config_path="${base_path}${SLURM_ARRAY_TASK_ID}.json"
python3 scripts/run.py --config ${config_path}

