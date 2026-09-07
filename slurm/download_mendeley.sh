#!/usr/bin/env bash
#SBATCH --job-name=dl_mendeley
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl_mendeley_%j.out
#SBATCH --error=logs/dl_mendeley_%j.err
set -uo pipefail
cd "$HOME/spine-detector"
export PATH="$HOME/mambaforge/bin:$PATH"
python scripts/download_sources.py --only mendeley_716 cm_spine
