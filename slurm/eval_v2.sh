#!/bin/bash
#SBATCH --job-name=sd_eval_v2
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/sd_eval_v2_%j.out
#SBATCH --error=logs/sd_eval_v2_%j.err
# Evaluate detect_v2 once its training job has ended. The training job's own eval stage
# looked for runs/detect_v2*/weights/best.pt, but ultralytics wrote the run under
# runs/pose/runs/detect_v2 (the task directory is inserted under --project), so it found
# nothing. This job resolves the weights wherever they landed.
set -uo pipefail
cd "$HOME/spine-detector"
source "$HOME/mambaforge/etc/profile.d/conda.sh"; conda activate spinedet
W=$(ls -t runs/pose/runs/detect_v2*/weights/best.pt runs/detect_v2*/weights/best.pt 2>/dev/null | head -1)
echo "using weights: ${W}"
[ -n "${W}" ] || { echo "no best.pt found"; exit 1; }
tail -3 "$(dirname "$(dirname "${W}")")/results.csv" || true
python -m src.run_eval --weights "${W}" \
    --manifest data/interim/manifest.parquet --images data/interim/images \
    --out runs/detect_v2/eval
echo "=============== EVAL DONE ==============="
ls runs/detect_v2/eval
