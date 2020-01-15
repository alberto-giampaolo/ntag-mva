#! /bin/bash
source ~/.bash_profile
which python

SEARCH_ITER="/home/llr/t2k/giampaolo/srn/ntag-mva/param_search_bdt.py"

# Do 1 iteration of hyperparameter search
# $1: parameter file
python $SEARCH_ITER $1