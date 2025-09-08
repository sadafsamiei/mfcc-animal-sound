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

PROJECT_ROOT="/scratch/ssamie/MFCC-animal-sound"
VENV_ACTIVATE="/scratch/ssamie/myenv/bin/activate"

source "$VENV_ACTIVATE"

echo ">>> Step 1: Preprocessing audio into MFCC features"
python "$PROJECT_ROOT/scripts/preprocessing.py"

echo ">>> Step 2: Generating MFCC + Log-Mel plots and waveforms"
python "$PROJECT_ROOT/src/plot_mfcc.py"

echo ">>> Step 3: Training model (LSTM with Attention)"
python "$PROJECT_ROOT/train.py"

echo ">>> Step 4: Evaluating model on validation set"
python "$PROJECT_ROOT/eval.py"

echo ">>> Step 5: Exporting model to ONNX format"
python "$PROJECT_ROOT/scripts/onnx_processing.py"

echo ">>> Step 6: Running OOSR classifier"
python "$PROJECT_ROOT/src/osr_classifier.py"

echo ">>> Step 7: Plotting distance-to-centroid histograms"
python "$PROJECT_ROOT/utils/util_centroid.py"

echo ">>> Step 8: Running Uncertainty Estimation (Softmax Entropy + MC Dropout + ROC)"
python "$PROJECT_ROOT/src/osr_uncertainty.py"

echo ">>> Step 9: Plotting Confusion Matrix"
python "$PROJECT_ROOT/utils/confusion_matrix.py"

echo ">>> Step 10: Embedding Space Analysis (t-SNE + KMeans + Logistic Regression)"
python "$PROJECT_ROOT/utils/embedding_analysis.py"

echo ">>> Step 11: Explainability (Attention Visualizations)"
python "$PROJECT_ROOT/utils/explainability.py"

echo ">>> Step 12: Adversarial Sensitivity Analysis (Gaussian + FGSM)"
python "$PROJECT_ROOT/utils/adversarial_sensitivity.py"

echo ">>> All steps completed successfully. Results are in $PROJECT_ROOT/results"
