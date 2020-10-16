#!/bin/bash
source exec_card.sh
# PYTHON="/disk02/usr6/elhedri/miniconda3/envs/py3/bin/python"
PYTHON="/home/giampaol/miniconda3/bin/python"

export results_dir=$models_dir/$model_name
mkdir -p $results_dir
model_card=$results_dir/model_card.sh

# Save training params
sed -n '/# Training/, $ p' < exec_card.sh > $model_card
sed -i '1i\'$"#!/bin/bash\n" $model_card

# Add evaluation params
sed -i '/# Evaluation/, $ d' $model_card
echo "# Evaluation ----------------------------
# single model
export eval_model=$model_name
export eval_label=$model_name
export n10th_eval=$n10th
export ntrees_eval=$ntrees
export extra_dset=0.0
export extra_dset_t2k=0.0
export ps_eff=1.0
export ps_bg=1.0
# ---------------------------------------" >> $model_card 

train=`$PYTHON training_bdt.py 2>$results_dir/$model_name.err 1>$results_dir/$model_name.log`
# screen -S $model_name -d -m  $train
$train
