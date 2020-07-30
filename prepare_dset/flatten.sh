#! /bin/bash
source ~/.bash_profile
which python

ROOTFILES_DIR="/data_CMS/cms/giampaolo/mc/darknoise4_new/root"
ROOTFILES_DIR2="/data_CMS/cms/giampaolo/mc/td-ntag-dset-lowN10/root/mc_truth"

# python script wrapped in shell script (wrapper needed to properly launch jobs)
WRAPPER="/home/llr/t2k/giampaolo/srn/ntag-mva/prepare_dset/flatten_ttree.sh"

# T3 cluster queue and T3 submit command
QUEUE="long"
SUBMIT="/opt/exp_soft/cms/t3/t3submit"


for file in $ROOTFILES_DIR/ntag.r[0-9][0-9][0-9][0-9][0-9].*.root ; do
  if [ -e "$file" ] ; then # Check whether file exists.

     FILE2=${file/$ROOTFILES_DIR/$ROOTFILES_DIR2}
     FILE2=${FILE2//ntag/skdetsim}
     FILE2=${FILE2/skdetsim/ntag}
     echo $SUBMIT -$QUEUE $WRAPPER $file $FILE2
     $SUBMIT -$QUEUE $WRAPPER $file $FILE2
  fi
done

