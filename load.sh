#!/bin/bash
#SBATCH --job-name=animal-audio
#SBATCH --output=results/log_%j.out
#SBATCH --error=results/log_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail
module load cuda/12.1 || true

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export MPLBACKEND=Agg

PROJECT_ROOT="/scratch/ssamie/mfcc-animal-sound"
VENV_ACTIVATE="/scratch/ssamie/myenv/bin/activate"

SR=16000
N_MFCC=20
FRAMES=256

ARCH="mfcc_lstm"         
EPOCHS=40                
BATCH=8                 
LR=5e-4                  
DEVICE="cuda"

FEATURES_NPZ="results/features.npz"
CLASS_MAP="results/classes.json"
CKPT="results/best_model.pt"

echo "[INFO] cd ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"
mkdir -p results data/animals

echo "[INFO] activate venv"
source "${VENV_ACTIVATE}"

echo "=== Step 1: Preprocess data ==="
python src/preprocessing.py \
  --in_dir data/animals \
  --out_npz "${FEATURES_NPZ}" \
  --class_map "${CLASS_MAP}" \
  --sr ${SR} --n_mfcc ${N_MFCC} --frames ${FRAMES}

echo "=== Step 2: Train model (${ARCH}) ==="
python src/train.py \
  --features "${FEATURES_NPZ}" \
  --class_map "${CLASS_MAP}" \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH} \
  --lr ${LR} \
  --device ${DEVICE} \
  --arch ${ARCH} \
  --ckpt "${CKPT}" \

[ -f results/loss_curve.png ] && echo "[INFO] loss curve: results/loss_curve.png"

echo "=== Step 3: Evaluate model (${ARCH}) ==="
python src/eval.py \
  --model "${CKPT}" \
  --features "${FEATURES_NPZ}" \
  --class_map "${CLASS_MAP}" \
  --device ${DEVICE} \
  --arch ${ARCH} \
  --batch_size ${BATCH} \
  | tee results/eval_report.txt

echo "=== Step 4: Build HTML report ==="
python scripts/visualise_results.py \
  --report_txt results/eval_report.txt \
  --cm_png results/confusion_matrix.png \
  --out_html results/report.html

echo "=== Step 5: Plot waveforms + MFCCs (Matplotlib) ==="
python scripts/inference.py \
  --do_plot \
  --species auto \
  --sr ${SR} --n_mfcc ${N_MFCC} --frames ${FRAMES} \
  --focus_seconds 3.0 --max_points 24000 || true
  # add --do_predict --ckpt results/best_model.pt --class_map results/classes.json to also write infer_predictions.csv

echo "=== DONE ==="
echo "Report: results/report.html"
echo "Loss curve: results/loss_curve.png"
echo "Waveforms: results/*_waveforms_concat_colored.png"
echo "MFCCs: results/mfcc_plots/*/mfcc_*.png"
