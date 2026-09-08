#!/bin/bash
#SBATCH --job-name=dl_buu_drive
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dl_buu_drive_%j.out
#SBATCH --error=logs/dl_buu_drive_%j.err
# BUU-LSPINE full release from the Google Drive folder linked by github.com/North-Github/BUU-LSPINE,
# throttled (slurm/download_buu_drive.py) because Drive rate-limits after ~70 fast requests.
# Resumable; if it stops on quota, resubmit with --begin=now+12hours.
set -uo pipefail
source "${HOME}/mambaforge/etc/profile.d/conda.sh"; conda activate spinedet
cd "${HOME}/spine-detector"
mkdir -p data/raw/buu_drive logs
echo "start $(date)"
python slurm/download_buu_drive.py --out data/raw/buu_drive --sleep "${SLEEP:-2.5}"
rc=$?
echo "exit ${rc} $(date)"
find data/raw/buu_drive -type f | wc -l; du -sh data/raw/buu_drive
