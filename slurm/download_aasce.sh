#!/usr/bin/env bash
#SBATCH --job-name=dl_aasce
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl_aasce_%j.out
#SBATCH --error=logs/dl_aasce_%j.err
# AASCE MICCAI 2019 (SpineWeb Dataset 16) from the Kaggle re-upload of the 481 labelled
# training images; the original SpineWeb host is down. Kaggle API token in ~/.kaggle/kaggle.json.
set -uo pipefail
cd "$HOME/spine-detector"
source "$HOME/mambaforge/etc/profile.d/conda.sh"; conda activate spinedet
python -c "import kaggle" 2>/dev/null || pip install -q kaggle
mkdir -p data/raw/aasce
kaggle datasets download -d wahyurahmaniar/aasce-miccai-2019-x-ray-dataset -p data/raw/aasce --unzip
echo "exit $?"
find data/raw/aasce -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn | head
du -sh --apparent-size data/raw/aasce
