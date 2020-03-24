#! /bin/bash
source ~/.bash_profile
which python

ROOTFILES_DIR="/data_CMS/cms/giampaolo/mc/darknoise/root_flat"

# python script wrapped in shell script (wrapper needed to properly launch jobs)
WRAPPER="/home/llr/t2k/giampaolo/srn/ntag-mva/prepare_dset/root_hdf5.sh"

# T3 cluster queue and T3 submit command
QUEUE="short"
SUBMIT="/opt/exp_soft/cms/t3/t3submit"

for file in "$ROOTFILES_DIR"/ntag.r[0-9][0-9][0-9][0-9][0-9].*.root ; do
  if [ -e "$file" ] ; then # Check whether file exists.

    echo $SUBMIT -$QUEUE $WRAPPER $file
    "$SUBMIT" "-$QUEUE" "$WRAPPER" "$file"
  fi
done