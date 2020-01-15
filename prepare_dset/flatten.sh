#! /bin/bash
source ~/.bash_profile
which python

# python script wrapped in shell script (wrapper needed to properly launch jobs)
WRAPPER="/home/llr/t2k/giampaolo/srn/ntag-mva/prepare_dset/flatten_ttree.sh"

# T3 cluster queue and T3 submit command
QUEUE="long"
SUBMIT="/opt/exp_soft/cms/t3/t3submit"

for i in {100..199..5}; do
    echo $SUBMIT -$QUEUE $WRAPPER $i $((i+5))
    $SUBMIT -$QUEUE -avx $WRAPPER $i $((i+5))
done
