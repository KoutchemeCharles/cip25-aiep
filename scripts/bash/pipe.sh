#!/bin/bash

## Important variables
code_folder=""
name="diagnostic_feedback"
config_dir=config/experiments/$name

## Useful scripts and script shortcuts
cpu_script=scripts/bash/generic/cpu.sh
gpu_script=scripts/bash/generic/gpu.sh
multi_gpu_train=scripts/bash/generic/multi_gpu.sh

log_folder="--output=$code_folder/logs/$name/%A_%a.log"

## Models
teacher=config/model/remote/gpt-4.1.yaml
teacher2=config/model/remote/gpt-4.1-mini.yaml
student=config/model/local/qwen-2.5-coder.yaml

## Data 
train_data=config/data/cip_4.yaml
eval_data=config/data/cip_3.yaml

## Generation tasks
teacher_feedback="config/task/feedback/generate_teacher_feedback_with_reasoning.yaml"
feedback="config/task/feedback/generate_feedback_with_reasoning.yaml"
judging="config/task/judge/sag_judge_with_reasoning.yaml"

## Training task 
sft_training=config/task/train/sft/train_sft.yaml
dpo_training=config/task/train/dpo/train_dpo.yaml

## Here there should be other scripts to perform data preprocessing omitted for privacy reasons

## First, let's generate example grading and feedback using the teacher model
python3 scripts/generate_config.py --name $name --model $teacher --dataset config/data/cip4/cip_4_diag1.yaml --task $teacher_feedback
python3 scripts/generate_config.py --name $name --model $teacher --dataset config/data/cip4/cip_4_diag2.yaml --task $teacher_feedback
python3 scripts/generate_config.py --name $name --model $teacher --dataset config/data/cip4/cip_4_diag4.yaml --task $teacher_feedback
python3 scripts/generate_config.py --name $name --model $teacher --dataset config/data/cip4/cip_4_diag5.yaml --task $teacher_feedback
# sbatch --wait --export=folder=$name --array=0 $log_folder $cpu_script

## Then, let's supervised finetune the small student model on those generations
train_dataset=(
    $config_dir/0.json
    $config_dir/1.json
    $config_dir/2.json
    $config_dir/3.json
)
python3 scripts/generate_config.py --name $name --model $student --dataset "${train_dataset[@]}" --task config/task/train/sft/train_sft_v1.yaml
# sbatch --wait --export=folder=$name --array=4 $log_folder $multi_gpu_train

## Generate on the training set again for a final round of RL, we use the teacher generation for later easier judging  
python3 scripts/generate_config.py --name $name --model $config_dir/4.json --dataset $config_dir/0.json --task $feedback
python3 scripts/generate_config.py --name $name --model $config_dir/4.json --dataset $config_dir/1.json --task $feedback
python3 scripts/generate_config.py --name $name --model $config_dir/4.json --dataset $config_dir/2.json --task $feedback
python3 scripts/generate_config.py --name $name --model $config_dir/4.json --dataset $config_dir/3.json --task $feedback
# sbatch --wait --export=folder=$name --array=5-8 $log_folder $gpu_script

## Evaluating with the judge
python3 scripts/generate_config.py --name $name --model $teacher --dataset $config_dir/5.json --task $judging
python3 scripts/generate_config.py --name $name --model $teacher --dataset $config_dir/6.json --task $judging
python3 scripts/generate_config.py --name $name --model $teacher --dataset $config_dir/7.json --task $judging
python3 scripts/generate_config.py --name $name --model $teacher --dataset $config_dir/8.json --task $judging
# sbatch --wait --export=folder=$name --array=9-12 $log_folder $cpu_script

## Train with DPO and KTO
train_dataset=(
    # teacher generations for support 
    $config_dir/0.json
    $config_dir/1.json
    $config_dir/2.json
    $config_dir/3.json
    # student generations 
    $config_dir/9.json
    $config_dir/10.json
    $config_dir/11.json
    $config_dir/12.json
)
## Train with DPO
python3 scripts/generate_config.py --name $name --model $config_dir/4.json --dataset "${train_dataset[@]}" --task config/task/train/dpo/train_dpo_v10.yaml
# sbatch --wait --export=folder=$name --array=13 $log_folder $multi_gpu_train

## CIP-3 evaluation
# Teacher generations
python3 scripts/generate_config.py --name $name --model $teacher2 --dataset config/data/cip3/cip_3_diag1.yaml --task $teacher_feedback
python3 scripts/generate_config.py --name $name --model $teacher2 --dataset config/data/cip3/cip_3_diag2.yaml --task $teacher_feedback
python3 scripts/generate_config.py --name $name --model $teacher2 --dataset config/data/cip3/cip_3_diag4.yaml --task $teacher_feedback
python3 scripts/generate_config.py --name $name --model $teacher2 --dataset config/data/cip3/cip_3_diag5.yaml --task $teacher_feedback
# sbatch --wait --export=folder=$name --array=14-17 $log_folder $cpu_script

# SFT generations
python3 scripts/generate_config.py --name $name --model $config_dir/4.json --dataset $config_dir/14.json --task $feedback
python3 scripts/generate_config.py --name $name --model $config_dir/4.json --dataset $config_dir/15.json --task $feedback
python3 scripts/generate_config.py --name $name --model $config_dir/4.json --dataset $config_dir/16.json --task $feedback
python3 scripts/generate_config.py --name $name --model $config_dir/4.json --dataset $config_dir/17.json --task $feedback
# sbatch --wait --export=folder=$name --array=18-21 $log_folder $gpu

# DPO generations
python3 scripts/generate_config.py --name $name --model $config_dir/13.json --dataset $config_dir/14.json --task $feedback
python3 scripts/generate_config.py --name $name --model $config_dir/13.json --dataset $config_dir/15.json --task $feedback
python3 scripts/generate_config.py --name $name --model $config_dir/13.json --dataset $config_dir/16.json --task $feedback
python3 scripts/generate_config.py --name $name --model $config_dir/13.json --dataset $config_dir/17.json --task $feedback
# sbatch --wait --export=folder=$name --array=22-25 $log_folder $gpu_script

## Evaluating with the judge
judging_dataset1=(
    $config_dir/18.json
    $config_dir/19.json
    $config_dir/20.json
    $config_dir/21.json
)
judging_dataset2=(
    $config_dir/22.json
    $config_dir/23.json
    $config_dir/24.json
    $config_dir/25.json
)
python3 scripts/generate_config.py --name $name --model $teacher2 --dataset "${judging_dataset1[@]}" --task $judging
python3 scripts/generate_config.py --name $name --model $teacher2 --dataset "${judging_dataset2[@]}" --task $judging
# sbatch --wait --export=folder=$name --array=27 $log_folder $cpu_script

ds=(
    $config_dir/26.json
    $config_dir/27.json
)
# python3 scripts/analyze_results.py --dataset "${ds[@]}"

## Evaluation on CIP-5
# Inference on CIP-5 with GPT-4
# Experienced students
python3 scripts/generate_config.py --name $name --model $teacher --dataset config/data/cip5/es/cip_5_diag1_es.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $teacher --dataset config/data/cip5/es/cip_5_diag2_es.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $teacher --dataset config/data/cip5/es/cip_5_diag3_es.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $teacher --dataset config/data/cip5/es/cip_5_diag4_es.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $teacher --dataset config/data/cip5/es/cip_5_diag5_es.yaml --task $feedback
# Normal students
python3 scripts/generate_config.py --name $name --model $teacher --dataset config/data/cip5/cip_5_diag1.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $teacher --dataset config/data/cip5/cip_5_diag2.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $teacher --dataset config/data/cip5/cip_5_diag3.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $teacher --dataset config/data/cip5/cip_5_diag4.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $teacher --dataset config/data/cip5/cip_5_diag5.yaml --task $feedback
# sbatch --export=folder=$name --array=28-37 $log_folder $cpu_script


slm=$config_dir/13.json # Using DPO which obtains good results at the end -> v7 dpo for now is the best

# Inference on CIP-5 with the small language model
# Experienced students
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/es/cip_5_diag1_es.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/es/cip_5_diag2_es.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/es/cip_5_diag4_es.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/es/cip_5_diag5_es.yaml --task $feedback
# Normal students
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/cip_5_diag2.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/cip_5_diag5.yaml --task $feedback
# Batching this one because takes a lot of time 
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/chunks/diag1/cip_5_diag1_chunk5.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/chunks/diag1/cip_5_diag1_chunk4.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/chunks/diag1/cip_5_diag1_chunk3.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/chunks/diag1/cip_5_diag1_chunk2.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/chunks/diag1/cip_5_diag1_chunk1.yaml --task $feedback
# Batching this one because takes a lot of time 
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/chunks/diag4/cip_5_diag4_chunk5.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/chunks/diag4/cip_5_diag4_chunk4.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/chunks/diag4/cip_5_diag4_chunk3.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/chunks/diag4/cip_5_diag4_chunk2.yaml --task $feedback
python3 scripts/generate_config.py --name $name --model $slm --dataset config/data/cip5/chunks/diag4/cip_5_diag4_chunk1.yaml --task $feedback
# sbatch --wait --export=folder=$name --array=38-53 $log_folder $gpu_script

## Other scripts to perform annotation analysis using Argilla omitted for privacy reasons
