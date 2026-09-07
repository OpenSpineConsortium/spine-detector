#!/usr/bin/env bash
#SBATCH --job-name=spinedet_dl
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
set -uo pipefail
cd "$HOME/spine-detector"
export PATH="$HOME/mambaforge/bin:$PATH"
python scripts/download_sources.py
