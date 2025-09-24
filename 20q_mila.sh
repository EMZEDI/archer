#!/bin/bash
#SBATCH --job-name=archer-20q
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100l:2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --partition=main
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source .env
uv run scripts/run.py --config-name archer_20q