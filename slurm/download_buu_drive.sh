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
# BUU-LSPINE v1 from the Google Drive folder linked in github.com/North-Github/BUU-LSPINE
# (the README calls it the full 3,600-patient release; the portal's 3,600 set is form-gated,
# so whatever this folder holds is what is public). gdown handles Drive folders.
set -uo pipefail
source "${HOME}/mambaforge/etc/profile.d/conda.sh"; conda activate spinedet
cd "${HOME}/spine-detector"
mkdir -p data/raw/buu_drive logs
pip install --quiet --upgrade gdown 2>&1 | tail -1
echo "start $(date)"
gdown --folder --continue "https://drive.google.com/drive/folders/1pSqtxYT_B2GHweoDAVB_32JNsHykuZK4" -O data/raw/buu_drive
rc=$?
echo "gdown exit ${rc} $(date)"
find data/raw/buu_drive -type f | wc -l; du -sh data/raw/buu_drive
find data/raw/buu_drive -maxdepth 3 | head -30
