#!/usr/bin/env bash
#SBATCH --job-name=build_ds
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/build_ds_%j.out
#SBATCH --error=logs/build_ds_%j.err
set -uo pipefail
cd "$HOME/spine-detector"
source "$HOME/mambaforge/etc/profile.d/conda.sh"; conda activate spinedet
python -m src.build_dataset
