#!/usr/bin/env bash

PARAM_DIR="/home/llr/t2k/giampaolo/srn/ntag-mva/models/BDT/test_search/params"
OUT_PREFIX="BDT/test_search/"
SCRIPT="/home/llr/t2k/giampaolo/srn/ntag-mva/param_search_bdt.py"

PYTHON="/usr/bin/python3"
QUEUE="long"
SUBMIT="/opt/exp_soft/cms/t3/t3submit"



for file in $PARAM_DIR/* ; do
  if [ -e "$file" ] ; then   # Check whether file exists.
     echo $SUBMIT -$QUEUE $PYTHON $SCRIPT $file $OUT_PREFIX${file: -3:1}
     $SUBMIT -$QUEUE $PYTHON $SCRIPT $file $OUT_PREFIX${file: -3:1}
  fi
done
