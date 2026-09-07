#!/usr/bin/env bash
#SBATCH --job-name=gpu_check
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/gpu_check_%j.out
#SBATCH --error=logs/gpu_check_%j.err
set -uo pipefail
cd "$HOME/spine-detector"
echo "== nvidia-smi =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
source "$HOME/mambaforge/etc/profile.d/conda.sh"
conda activate spinedet
python - <<'PY'
import torch
print("torch", torch.__version__, "built for cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device="cuda")
    print("matmul ok:", float((x @ x).sum()) != 0)
PY
