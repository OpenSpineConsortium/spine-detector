#!/usr/bin/env bash
#SBATCH --job-name=sd_train
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=20:00:00
#SBATCH --output=logs/sd_train_%j.out
#SBATCH --error=logs/sd_train_%j.err
set -uo pipefail
cd "$HOME/spine-detector"
source "$HOME/mambaforge/etc/profile.d/conda.sh"; conda activate spinedet

echo "=============== YOLO EXPORT ==============="
python -m src.export_yolo --manifest data/interim/manifest.parquet \
    --images data/interim/images --out data/yolo --symlink
if [ ! -f data/yolo/data.yaml ]; then echo "export produced no data.yaml"; exit 1; fi
for sp in train val test; do
  echo "  $sp: $(ls data/yolo/images/$sp 2>/dev/null | wc -l) images"
done

echo "=============== TRAIN ==============="
# fliplr is 0 in src/train.py: a horizontal flip is anatomically impossible on a spine
# radiograph and silently destroys corner-slot semantics.
python -m src.train --data data/yolo/data.yaml --model yolo11m-pose.pt \
    --epochs 120 --imgsz 1024 --batch 16 --device 0 --workers 12 \
    --project runs --name detect_v1

echo "=============== PREDICT + EVALUATE ==============="
W=$(find runs -name best.pt -newermt "-1 day" | head -1)
echo "using weights: $W"
python -m src.run_eval --weights "$W" \
    --manifest data/interim/manifest.parquet --images data/interim/images \
    --out runs/detect_v1/eval || echo "eval stage failed (non-fatal)"
echo "=============== DONE ==============="
