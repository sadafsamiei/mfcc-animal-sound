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

# ---- HPC clamps & headless plotting ----
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export MPLBACKEND=Agg

# ---- Paths ----
PROJECT_ROOT="/scratch/ssamie/mfcc-animal-sound"
VENV_ACTIVATE="/scratch/ssamie/myenv/bin/activate"

# ---- Feature / model config ----
SR=16000
N_MFCC=40
FRAMES=384

ARCH="mfcc_lstm"          # or "mfcc_cnn"
EPOCHS=40
BATCH=8
LR=5e-4
DEVICE="cuda"

FEATURES_NPZ="results/features.npz"
CLASS_MAP="results/classes.json"
CKPT="results/best_model.pt"

echo "[INFO] cd ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"
mkdir -p results data/animals data/test

echo "[INFO] activate venv"
# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"

# -------------------- 1) Preprocess --------------------
echo "=== Step 1: Preprocess data ==="
python src/preprocessing.py \
  --in_dir data/animals \
  --out_npz "${FEATURES_NPZ}" \
  --class_map "${CLASS_MAP}" \
  --sr ${SR} --n_mfcc ${N_MFCC} --frames ${FRAMES}

# -------------------- 2) Train --------------------
echo "=== Step 2: Train model (${ARCH}) ==="
python src/train.py \
  --features "${FEATURES_NPZ}" \
  --class_map "${CLASS_MAP}" \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH} \
  --lr ${LR} \
  --device ${DEVICE} \
  --arch ${ARCH} \
  --ckpt "${CKPT}"

[ -f results/loss_curve.png ] && echo "[INFO] loss curve: results/loss_curve.png"

# -------------------- 3) Evaluate --------------------
echo "=== Step 3: Evaluate model (${ARCH}) ==="
python src/eval.py \
  --model "${CKPT}" \
  --features "${FEATURES_NPZ}" \
  --class_map "${CLASS_MAP}" \
  --device ${DEVICE} \
  --arch ${ARCH} \
  --batch_size ${BATCH} \
  | tee results/eval_report.txt

# -------------------- 4) HTML report --------------------
echo "=== Step 4: Build HTML report ==="
python scripts/visualise_results.py \
  --report_txt results/eval_report.txt \
  --cm_png results/confusion_matrix.png \
  --out_html results/report.html

# -------------------- 5) Waveforms + MFCCs --------------------
echo "=== Step 5: Plot waveforms + MFCCs (Matplotlib) ==="
python scripts/inference.py \
  --do_plot \
  --plot_combined \
  --species auto \
  --sr ${SR} --n_mfcc ${N_MFCC} --frames ${FRAMES} \
  --focus_seconds 3.0 --max_points 24000 || true

# -------------------- 6) OSR: robust threshold tuning --------------------
echo "=== Step 6: OSR threshold tuning on s1..s10 ==="
TMP_CSV="results/osr_tmp.csv"
best_acc="0.0"
BEST_MSP=""
BEST_ENT=""

# explicit grids (avoid float seq portability issues)
MSP_GRID=(0.55 0.60 0.65 0.70 0.75 0.80 0.85)
ENT_GRID=(1.00 1.10 1.20 1.30 1.40 1.50 1.60)

# prefer GNU timeout if available
TIMEOUT_BIN=$(command -v timeout || true)
PER_RUN_TIMEOUT=60s

i=0
total=$(( ${#MSP_GRID[@]} * ${#ENT_GRID[@]} ))
for m in "${MSP_GRID[@]}"; do
  for e in "${ENT_GRID[@]}"; do
    i=$((i+1))
    echo "[OSR] (${i}/${total}) msp=${m} ent=${e}"

    # run on CPU to avoid slow CUDA init during grid search
    if [[ -n "${TIMEOUT_BIN}" ]]; then
      ${TIMEOUT_BIN} ${PER_RUN_TIMEOUT} python -m src.models.osr_classifier \
        --root "${PROJECT_ROOT}" \
        --test_dir "${PROJECT_ROOT}/data/test" \
        --arch ${ARCH} \
        --ckpt "${CKPT}" \
        --sr ${SR} --n_mfcc ${N_MFCC} --frames ${FRAMES} \
        --device cpu \
        --feature_mode auto --cmvn \
        --msp_thresh ${m} --ent_thresh ${e} \
        --save_csv "${TMP_CSV}" || echo "[WARN] run timed out or failed (msp=${m}, ent=${e})"
    else
      python -m src.models.osr_classifier \
        --root "${PROJECT_ROOT}" \
        --test_dir "${PROJECT_ROOT}/data/test" \
        --arch ${ARCH} \
        --ckpt "${CKPT}" \
        --sr ${SR} --n_mfcc ${N_MFCC} --frames ${FRAMES} \
        --device cpu \
        --feature_mode auto --cmvn \
        --msp_thresh ${m} --ent_thresh ${e} \
        --save_csv "${TMP_CSV}" || echo "[WARN] run failed (msp=${m}, ent=${e})"
    fi

    # parse accuracy if file exists
    if [[ -f "${TMP_CSV}" ]]; then
      acc=$(awk -F, '/^accuracy,/{print $2}' "${TMP_CSV}" | tail -n1)
      if [[ -n "${acc}" ]]; then
        # compare as floats using awk
        better=$(awk -v a="${acc}" -v b="${best_acc}" 'BEGIN{print (a>b)?"1":"0"}')
        if [[ "${better}" == "1" ]]; then
          best_acc="${acc}"
          BEST_MSP="${m}"
          BEST_ENT="${e}"
          echo "[OSR] new best acc=${best_acc} @ MSP=${BEST_MSP} ENT=${BEST_ENT}"
        fi
      fi
    fi
  done
done

if [[ -z "${BEST_MSP}" || -z "${BEST_ENT}" ]]; then
  echo "[WARN] No valid OSR results parsed; using defaults MSP=0.60 ENT=1.20"
  BEST_MSP="0.60"
  BEST_ENT="1.20"
else
  echo "[INFO] Tuned thresholds -> MSP=${BEST_MSP}  ENT=${BEST_ENT}  (acc=${best_acc})"
fi

# -------------------- 7) OSR: final pass with tuned thresholds --------------------
echo "=== Step 7: OSR final run with tuned thresholds ==="
python -m src.models.osr_classifier \
  --root "${PROJECT_ROOT}" \
  --test_dir "${PROJECT_ROOT}/data/test" \
  --arch ${ARCH} \
  --ckpt "${CKPT}" \
  --sr ${SR} --n_mfcc ${N_MFCC} --frames ${FRAMES} \
  --device cpu \
  --feature_mode auto --cmvn \
  --msp_thresh "${BEST_MSP}" \
  --ent_thresh "${BEST_ENT}" \
  --save_csv results/osr_predictions.csv \
  | tee results/osr_stdout.txt

echo "=== DONE ==="
echo "Report:      results/report.html"
echo "Loss curve:  results/loss_curve.png"
echo "Waveforms:   results/*_waveforms_concat_colored.png"
echo "MFCCs:       results/mfcc_plots/*/mfcc_*.png"
echo "OSR tuning:  best MSP=${BEST_MSP}, ENT=${BEST_ENT}, acc=${best_acc}"
echo "OSR final:   results/osr_predictions.csv"
