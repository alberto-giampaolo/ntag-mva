#!/bin/bash
source exec_card.sh

model_card=$models_dir/$apply_model/model_card.sh
source $model_card

export n10th_eval=$apply_n10th
export ntag_hdf5_dir_train=$apply_signal
export dset=0.0
export extra_dset=1.0
export ps_eff=$apply_ps_eff
export ps_bg=$apply_ps_bg

# export ntag_hdf5_dir_train_t2k=$apply_t2k
# export dset_t2k=0.0
# export extra_dset_t2k=1.0


PYTHON="/home/giampaol/miniconda3/bin/python"

echo "Applying model:" $apply_model
echo "To datasets:
$ntag_hdf5_dir_train (signal)
$ntag_hdf5_dir_train_t2k (background)"
apply_perf="$models_dir/$apply_model/${apply_label}_roc.csv"
if [ -f "$apply_perf" ]; then
    echo "Found performance file."
    echo "Evaluating performance..."
    $PYTHON eval.py $apply_perf
else 
    echo "Calculating BDT performance..."
    $PYTHON save_perf.py $apply_label
    echo "Evaluating performance..."
    $PYTHON eval.py $model_perf $apply_label
fi
