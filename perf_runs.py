"""
Save performance of current $eval_model BDT as csv for quick access.

Output files placed in model directory:
  roc_test.csv roc_train.csv    ROCs effs and bg rates (for given cuts)
  bdt_test.csv bdt_train.csv    BDT response hisograms

Format of csv columns:
  BDT cut, efficiency, mistag rate, eff. uncertainty, mistag uncertainty
"""
from __future__ import print_function
from os import environ

import numpy as np

from load_ntag import load_dset, load_model


MODEL_NAME = environ['eval_model']
MODELS_DIR = environ['models_dir']
N10TH = int(environ['n10th_eval'])
DSET_FRAC = float(environ['dset'])
DSET_FRAC_BG = float(environ['dset_t2k'])
DSET_FRAC_EXTRA = float(environ['extra_dset'])
DSET_FRAC_BG_EXTRA = float(environ['extra_dset_t2k'])
TEST_FRAC = float(environ['ntest'])
NTREES = int(environ['ntrees_eval'])

base_dir =  MODELS_DIR + '/' + MODEL_NAME + '/'

effile_runs = base_dir + '/' + 'eff_runs.csv'
bgfile_runs = base_dir + '/' + 'bg_runs.csv'

nbins = 1000
cut_sampling = 1 - np.logspace(np.log10(1e-5), np.log10(1.0), nbins)[::-1]
cut_sampling = np.concatenate((cut_sampling, [1.0]))
cuts = cut_sampling[:-1]
cut_widths = cut_sampling[1:] - cut_sampling[:-1]


def eff_err_bayes(k, n):
    '''
    Bayesian error on efficiency.
    Assumes binomially-distributed k:
    P(k|n, eff) = Bi(k|n, eff) and uniform prior P(eff).
    Avoids unexpected edge case behavior.
    '''
    term1 = ((k+1.)*(k+2)) / ((n+2)*(n+3))
    term2 = ((k+1.)*(k+1)) / ((n+2)*(n+2))
    return np.sqrt(term1-term2)


def bdt_perf_runs(x, y):

    def runs_s(ar):
        ''' List of feature arrays grouped run-by-run'''
        def get_run_arr(ar, run_num):
            ''' Get feature array from run number. '''
            return ar[ar[:, -2] == run_num]
        runs = np.unique(ar[:, -2])
        run_arrs = [get_run_arr(ar, r) for r in runs]
        return runs, run_arrs

    def runs_b(ar, num):
        ''' List of feature arrays by run groups'''
        def get_run_arr(ar, run_group):
            ''' Get feature array from list of num numbers. '''
            return ar[np.isin(ar[:, -2], run_group)]
        runs = np.unique(ar[:, -2])
        runs = np.sort(runs)
        run_groups = np.array_split(runs, num)
        run_arrs = [get_run_arr(ar, rg) for rg in run_groups]
        return run_groups, run_arrs

    def perf_s(xlist):
        def sigeff(x_s):
            s_tot = len(x_s)
            pred_s = ntag_model.predict_proba(x_s[:,:22], ntree_limit=NTREES)[:, 1]
            hist_s, _ = np.histogram(pred_s, bins=cut_sampling)

            cumul_s = np.cumsum(hist_s[::-1])
            cumul_s = cumul_s[::-1]
            eff_s = cumul_s / s_tot
            err_s = eff_err_bayes(cumul_s, s_tot)

            return np.vstack((eff_s, err_s)).T
        effs = [sigeff(x) for x in xlist]
        effs = np.hstack(effs)
        cutsT = cuts.reshape((cuts.size, 1))
        return np.hstack((cutsT, effs))

    def perf_b(xlist):
        def mistag(x_b):
            b_tot = len(x_b)
            pred_b = ntag_model.predict_proba(x_b[:,:22], ntree_limit=NTREES)[:, 1]
            hist_b, _ = np.histogram(pred_b, bins=cut_sampling)

            cumul_b = np.cumsum(hist_b[::-1])
            cumul_b = cumul_b[::-1]
            eff_b = cumul_b / b_tot
            err_b = eff_err_bayes(cumul_b, b_tot)

            return np.vstack((eff_b, err_b)).T
        bgs = [mistag(x) for x in xlist]
        bgs = np.hstack(bgs)
        cutsT = cuts.reshape((cuts.size, 1))
        return np.hstack((cutsT, bgs))

    x_s, x_b = x[y == 1], x[y == 0]
    runs_sig, x_s_runs = runs_s(x_s)
    runs_bg, x_b_runs = runs_b(x_b, 5)
    effs_array = perf_s(x_s_runs)
    bgs_array = perf_b(x_b_runs)
    return effs_array, bgs_array, runs_sig, runs_bg


print("Loading model...")
ntag_model = load_model(MODEL_NAME)

print("Loading train and test set...")
dset = load_dset(N10th=N10TH, file_frac_s=DSET_FRAC,
                 file_frac_bg=DSET_FRAC_BG,
                 test_frac=TEST_FRAC, nums=True)
print("Loading unused dataset for additional test samples...")
extra_dset = load_dset(N10th=N10TH, file_frac_s=DSET_FRAC_EXTRA,
                       file_frac_bg=DSET_FRAC_BG_EXTRA,
                       file_start_s=DSET_FRAC,
                       file_start_bg=DSET_FRAC_BG, nums=True)
x_test, _, _, y_test, _, _ = dset
x_test_extra = np.concatenate(extra_dset[:3])
y_test_extra = np.concatenate(extra_dset[3:])
x_test = np.concatenate((x_test, x_test_extra))
y_test = np.concatenate((y_test, y_test_extra))
del extra_dset, dset, x_test_extra, y_test_extra



# x_test_s, x_test_b = x_test[y_test == 1], x_test[y_test == 0]
# runs_sig, x_test_s_runs = runs_s(x_test_s)
# runs_bg, x_test_b_runs = runs_b(x_test_b, 5)
# print("Getting time-dependent performance...")
# effs_array = perf_s(x_test_s_runs)
# bgs_array = perf_b(x_test_b_runs)

runeff, runbg, runlist_s, runlist_b = bdt_perf_runs(x_test, y_test)

print("Saving...")
eff_head = ' | '.join([str(r) for r in runlist_s])
bg_head = ' | '.join([str(rx[0]) + '-' + str(rx[-1]) for rx in runlist_b])
np.savetxt(effile_runs, runeff, delimiter=", ", header=eff_head)
np.savetxt(bgfile_runs, runbg, delimiter=", ", header=bg_head)
