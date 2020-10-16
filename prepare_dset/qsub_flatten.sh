#!/bin/bash
source ../exec_card.sh

ulimit -c 0 # disable core dumps

# Create log and error dirs
logdir=$ntag_root_dir_flat/log
errdir=$ntag_root_dir_flat/err
mkdir -p $logdir
mkdir -p $errdir
mkdir -p $ntag_hdf5_dir

for fl in $ntag_root_dir/*ntag*.root; do
    jobname=${fl#$ntag_root_dir/}
    
    # runno=${jobname:9:-8}
    # if [ "$runno" -lt 72627 ]; then
    #     continue
    # fi

    # Job limit
    jobrunning=`qstat -a $queue | grep $USER | wc -l`
    echo $jobrunning" jobs running"
    while [ "$jobrunning" -gt "$maxjobs" ]
    do
        jobrunning=`qstat -a $queue | grep $USER | wc -l`
    done
    echo "qsub -q $queue -o $logdir/$jobname.out -e $errdir/$jobname.err -r $jobname ./run_flatten.sh"
    qsub -q $queue -o $logdir/$jobname.out -e $errdir/$jobname.err -r $jobname ./run_flatten.sh
done
