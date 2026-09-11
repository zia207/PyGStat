#!/bin/bash
# Executes every tutorial notebook in-place (populating code outputs), then
# renders the whole Quarto site (index.md, install.md, Tutorial/*.ipynb)
# to docs/. Logs progress per-notebook so it can be tailed while running.
set -uo pipefail
cd /home/zia207/Github/PyGStat

export CUDA_VISIBLE_DEVICES=-1
export JAVA_HOME=/home/zia207/.jre/jdk-17.0.20.1+1-jre
export PATH="$JAVA_HOME/bin:$PATH"

LOG=/tmp/tutorial_run_log.txt
: > "$LOG"

NOTEBOOKS=(
  "Tutorial/01_getting_started.ipynb"
  "Tutorial/02_variogram_modeling.ipynb"
  "Tutorial/03_simple_kriging.ipynb"
  "Tutorial/04_ordinary_kriging.ipynb"
  "Tutorial/05_universal_kriging.ipynb"
  "Tutorial/06_cokriging.ipynb"
  "Tutorial/07_01_regression_kriging_scikit_learn.ipynb"
  "Tutorial/07_02_regression_kriging_H20.ipynb"
  "Tutorial/07_03_regression_kriging_pytorch.ipynb"
  "Tutorial/07_04_regression_krigingDNN_TF.ipynb"
  "Tutorial/07_05_regression_kriging_gnn.ipynb"
  "Tutorial/08_indicator_kriging.ipynb"
  "Tutorial/09_kriging_3d.ipynb"
  "Tutorial/10_soft_kriging.ipynb"
  "Tutorial/10_disjunctive_kriging.ipynb"
  "Tutorial/11_empirical_bayesian_kriging.ipynb"
  "Tutorial/12_factorial_kriging.ipynb"
  "Tutorial/13_krigingST.ipynb"
  "Tutorial/14_kriging_CNNLSTM.ipynb"
  "Tutorial/15_kriging_STGNN.ipynb"
  "Tutorial/16_kriging_STGTN.ipynb"
  "Tutorial/17_01_poisson_Kriging_apt_atp.ipynb"
  "Tutorial/17_02_poisson_cokriging.ipynb"
  "Tutorial/17_03_poisson_kriging_ST.ipynb"
  "Tutorial/18_sgsim.ipynb"
  "Tutorial/19_sisim.ipynb"
)

TOTAL=${#NOTEBOOKS[@]}
I=0
FAILED=()
for nb in "${NOTEBOOKS[@]}"; do
  I=$((I+1))
  echo "[$I/$TOTAL] $(date '+%H:%M:%S') START  $nb" | tee -a "$LOG"
  t0=$(date +%s)
  if jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=2400 \
      --ExecutePreprocessor.kernel_name=python3 \
      "$nb" >> "$LOG" 2>&1; then
    t1=$(date +%s)
    echo "[$I/$TOTAL] $(date '+%H:%M:%S') OK     $nb ($((t1-t0))s)" | tee -a "$LOG"
  else
    t1=$(date +%s)
    echo "[$I/$TOTAL] $(date '+%H:%M:%S') FAILED $nb ($((t1-t0))s)" | tee -a "$LOG"
    FAILED+=("$nb")
  fi
done

echo "=================================================" | tee -a "$LOG"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "ALL $TOTAL NOTEBOOKS EXECUTED SUCCESSFULLY" | tee -a "$LOG"
else
  echo "${#FAILED[@]} notebook(s) FAILED:" | tee -a "$LOG"
  for f in "${FAILED[@]}"; do echo "  - $f" | tee -a "$LOG"; done
fi

echo "" | tee -a "$LOG"
echo "$(date '+%H:%M:%S') Rendering Quarto site..." | tee -a "$LOG"
if quarto render >> "$LOG" 2>&1; then
  echo "$(date '+%H:%M:%S') QUARTO RENDER OK" | tee -a "$LOG"
else
  echo "$(date '+%H:%M:%S') QUARTO RENDER FAILED" | tee -a "$LOG"
fi
echo "DONE_MARKER" >> "$LOG"
