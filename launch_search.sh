#! /bin/bash
source ~/.bash_profile

# Parameter file location to generate models
PARAM_DIR="/home/llr/t2k/giampaolo/srn/ntag-mva/models/BDT/grid_2_n10thr5_50k/params"

# script with single iteration of hyperparameter search
# python script wrapped in shell script
ITER_WRAPPER="/home/llr/t2k/giampaolo/srn/ntag-mva/search_iteration.sh"

# T3 cluster queue and T3 submit command
QUEUE="long"
SUBMIT="/opt/exp_soft/cms/t3/t3submit"

for file in $PARAM_DIR/* ; do
  if [ -e "$file" ] ; then # Check whether file exists.
     echo $SUBMIT -$QUEUE -avx $ITER_WRAPPER $file
     $SUBMIT -$QUEUE -avx $ITER_WRAPPER $file
  fi
done