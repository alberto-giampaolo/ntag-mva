#!/bin/bash
source ../exec_card.sh

ulimit -c 0 # Disable core dumps

fl_name=$PJM_JOBNAME # Parse job name to get file name
fl=$ntag_root_dir/$fl_name
fl_skdetsim=$skdetsim_dir/${fl_name/ntag.r/skdetsim.r}
fl_flat=$ntag_root_dir_flat/$fl_name
fl_h5=$ntag_hdf5_dir/${fl_name/%.root/.h5}

PY=/usr/bin/python2
if [ $preprocess_bg = true ]; then
    $PY flatten_ttree.py $fl $fl_flat
elif [ $cap_time -eq "-1" ]; then
    $PY flatten_ttree.py $fl $fl_flat -mc $fl_skdetsim
else
    $PY flatten_ttree.py $fl $fl_flat -ct $cap_time
fi

# Use ROOT 6 for hdf5 conversion with rootpy
source /disk02/usr6/giampaol/root/root/bin/thisroot.sh
$PY root_hdf5.py $fl_flat $fl_h5
