#!/usr/bin/env bash
#SBATCH --job-name=dl_vindr
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dl_vindr_%j.out
#SBATCH --error=logs/dl_vindr_%j.err
# VinDr-SpineXR (PhysioNet, restricted: registered user + signed DUA). Credentials come from
# ~/.netrc (mode 600), never from the command line. The DUA forbids redistribution: this data
# stays on the grid, never on Hugging Face, a laptop or a shared drive.
set -uo pipefail
cd "$HOME/spine-detector"
mkdir -p data/raw/vindr
cd data/raw/vindr
wget -r -N -c -np -nH --cut-dirs=2 -nv \
     https://physionet.org/files/vindr-spinexr/1.0.0/ 2>&1 | tail -n 50
echo "exit $?"
find . -type f | wc -l
du -sh --apparent-size .
