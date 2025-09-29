#!/bin/bash
#SBATCH --job-name=archer-20q-ds-2nodes-tamia
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --account=aip-rrabba
#SBATCH --mail-user=shahrad_m@icloud.com
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR MASTER_PORT

source .env
srun accelerate launch \
  --config_file scripts/config/accelerate_h100_config/deepspeed_zero2_config.yaml \
  --num_machines 2 --num_processes 8 \
  scripts/run.py --config-name accelerate_h100_config/archer_20q