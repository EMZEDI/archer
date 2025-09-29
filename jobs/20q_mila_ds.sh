#!/bin/bash
#SBATCH --job-name=archer-20q-ds
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --partition=short-unkillable
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source .env
accelerate launch --config_file scripts/config/accelerate_config/deepspeed_zero2_config.yaml scripts/run.py --config-name archer_20q