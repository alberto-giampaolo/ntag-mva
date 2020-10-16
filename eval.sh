#!/bin/bash
source exec_card.sh

model_card=$models_dir/$eval_model/model_card.sh
source $model_card

# PYTHON="/disk02/usr6/elhedri/miniconda3/envs/py3/bin/python"
# PYTHON=python3
PYTHON="/home/giampaol/miniconda3/bin/python"

echo "Evaluating model:" $eval_model
model_perf="$models_dir/$eval_model/roc_test.csv"
if [ -f "$model_perf" ]; then
    echo "Found performance file."
    echo "Evaluating performance..."
    $PYTHON eval.py $model_perf
else 
    echo "Calculating BDT performance..."
    $PYTHON save_perf.py
    echo "Evaluating performance..."
    $PYTHON eval.py $model_perf
fi
