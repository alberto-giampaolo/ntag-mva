#!/bin/bash
export skofl=/home/giampaol/skofl

# Cluster params
queue=ALL
maxjobs=200

# Preprocessing -------------------------
# skdetsim_dir=/disk02/usr6/giampaol/mc/gd_ibd/skdetsim/1_90 # MC truth
# ntag_root_dir=/disk02/usr6/giampaol/mc/gd_ibd/ntag/1_90 # original ntag dset
# ntag_root_dir_flat=/disk02/usr6/giampaol/mc/gd_ibd/ntag/flat # preprocessing
# ntag_hdf5_dir=/disk02/usr6/giampaol/mc/gd_ibd/ntag/h5 # hdf5 format for loading

# ntag_root_dir=/disk02/usr6/elhedri/SK2p2MeV/signal/srn/ntag/t2k # original ntag dset

# ntag_root_dir=/disk02/usr6/giampaol/mc/gd_gamma/ntag #/root # original ntag dset
# ntag_root_dir_flat=/disk02/usr6/giampaol/mc/gd_gamma/ntag/flat # preprocessing
# ntag_hdf5_dir=/disk02/usr6/giampaol/mc/gd_gamma/ntag/h5 # hdf5 format for loading

# skdetsim_dir=/disk02/usr6/giampaol/mc/srn_ntag/skdetsim # MC truth
# ntag_root_dir=/disk02/usr6/giampaol/mc/srn_ntag/ntag/root # original ntag dset
# ntag_root_dir_flat=/disk02/usr6/giampaol/mc/srn_ntag/ntag/flat # preprocessing
# ntag_hdf5_dir=/disk02/usr6/giampaol/mc/srn_ntag/ntag/h5 # hdf5 format for loading

ntag_root_dir=/disk02/usr6/elhedri/relic_mc/ntag_ibd/1_90
skdetsim_dir=/disk02/lowe8/relic_sk4/dec20/mc/nuebar/skdetsim/1_90
ntag_root_dir_flat=/disk02/usr6/giampaol/mc/srn_ntag_runs/ntag/flat # preprocessing
ntag_hdf5_dir=/disk02/usr6/giampaol/mc/srn_ntag_runs/ntag/h5 # hdf5 format for loading

preprocess_bg=false # If preprocessing pure T2K dummy data
cap_time=-1 # Fixed ncap time. Use skdesim file if -1
# ---------------------------------------

# Training ------------------------------
# training dset locations
export ntag_hdf5_dir_train=/disk02/usr6/giampaol/mc/srn_ntag/ntag/h5
# export ntag_hdf5_dir_train=/disk02/usr6/giampaol/mc/gd_gamma/ntag/h5/p
# export ntag_hdf5_dir_train=/disk02/usr6/giampaol/mc/gd_ibd/ntag/h5
export ntag_hdf5_dir_train=/disk02/usr6/giampaol/mc/srn_ntag_runs/ntag/h5
export ntag_hdf5_dir_train_t2k=/disk02/usr6/giampaol/data/t2k/h5
# model directory
export models_dir=/disk02/usr6/giampaol/ntag-mva/models # parent directory for models
# single-model params
export model_name=sk4_full_1500
export dset=1.0 # Proportion of available signal entries to use
export dset_t2k=1.0 # Proportion of available bg entries
export pos_weight=1.0 # Reweight signal entries during training
export ntest=0.25 # Test and validation set fractions
export n10th=6 # N10 >= n10th
export ntrees=1500 # Max number of trees in forest
export early_stop=50 # Stop training early after x iterations without improvement
export lrate=0.025219 # Learning rate
export max_depth=5 # Maximum tree depth
export subsample=0.97 # Take less than 100% of the trees in gradient calculation
export tree_method=auto # options: auto, exact, approx, hist, gpu_hist
export nthreads=6 # Multithreading
# ---------------------------------------

# Evaluation ----------------------------
export eval_model=bdt22_n10thr6_test_gdmix_gamma

# Application ---------------------------
export apply_model=sk4_full_1500
export apply_label=Gdmix
export apply_signal=/disk02/usr6/giampaol/mc/gd_gamma/ntag/h5/gdmix_EGLO
export apply_t2k=''
export apply_n10th=6
export apply_ps_eff=0.725
export apply_ps_bg=11.4
# ---------------------------------------
