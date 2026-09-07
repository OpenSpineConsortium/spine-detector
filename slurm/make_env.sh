#!/usr/bin/env bash
#SBATCH --job-name=spinedet_env
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=logs/make_env_%j.out
#SBATCH --error=logs/make_env_%j.err
set -euo pipefail
cd "$HOME/spine-detector"
source "$HOME/mambaforge/etc/profile.d/conda.sh"

# The scientific stack comes from conda-forge, NOT pip. Twice now pip has failed to
# match a wheel on this cluster and fallen back to building from source, where the
# system GCC (< 9.3) cannot compile numpy. conda-forge ships prebuilt binaries with
# their own toolchain, so the compiler question never arises.
rm -rf "$HOME/mambaforge/envs/spinedet"
mamba create -y -n spinedet -c conda-forge python=3.11 \
    "numpy<2.2" "pandas<3" pyarrow opencv scikit-image shapely tqdm pytest \
    openpyxl matplotlib pyyaml requests pillow
conda activate spinedet

# torch from the PyTorch index (cu124 for the H200s), with PyPI kept as a fallback
# so its dependency tree resolves. ultralytics and pydicom are pip-only.
pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple
pip install --no-cache-dir --no-deps ultralytics
pip install --no-cache-dir pydicom gdown "ultralytics-thop" psutil py-cpuinfo

python - <<'PY'
import torch, ultralytics, cv2, pandas, numpy, shapely
print("python      ", __import__("sys").version.split()[0])
print("numpy       ", numpy.__version__)
print("pandas      ", pandas.__version__)
print("opencv      ", cv2.__version__)
print("shapely     ", shapely.__version__)
print("torch       ", torch.__version__, "| cuda", torch.version.cuda)
print("ultralytics ", ultralytics.__version__)
PY
