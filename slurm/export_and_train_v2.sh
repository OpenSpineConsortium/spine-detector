#!/usr/bin/env bash
#SBATCH --job-name=sd_train_v2
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=30:00:00
#SBATCH --output=logs/sd_train_v2_%j.out
#SBATCH --error=logs/sd_train_v2_%j.err
# detect_v2 — the class-agnostic corner detector retrained on the corpus after the
# 2,000-pair BUU subset and AASCE were ingested (about 26,000 instances against 6,073 for
# detect_v1, and AP goes from 125 instances to about 12,000). Everything else is held
# identical to slurm/export_and_train.sh (model, epochs, image size, augmentation, split
# seed) so the difference in the evaluation is the data. Writes to data/yolo_v2 and
# runs/detect_v2; detect_v1 is left untouched as the baseline.
set -uo pipefail
cd "$HOME/spine-detector"
source "$HOME/mambaforge/etc/profile.d/conda.sh"; conda activate spinedet
echo "=============== YOLO EXPORT (v2) ==============="
python -m src.export_yolo --manifest data/interim/manifest.parquet \
    --images data/interim/images --out data/yolo_v2 --symlink
if [ ! -f data/yolo_v2/data.yaml ]; then echo "export produced no data.yaml"; exit 1; fi
for sp in train val test; do
  echo "  $sp: $(ls data/yolo_v2/images/$sp 2>/dev/null | wc -l) images"
done
echo "=============== TRAIN (v2) ==============="
python -m src.train --data data/yolo_v2/data.yaml --model yolo11m-pose.pt \
    --epochs 120 --imgsz 1024 --batch 16 --device 0 --workers 12 \
    --project runs --name detect_v2
echo "=============== PREDICT + EVALUATE (v2) ==============="
W=$(ls -t runs/detect_v2*/weights/best.pt 2>/dev/null | head -1)
echo "using weights: $W"
python -m src.run_eval --weights "$W" \
    --manifest data/interim/manifest.parquet --images data/interim/images \
    --out runs/detect_v2/eval || echo "eval stage failed (non-fatal)"
echo "=============== DONE ==============="
