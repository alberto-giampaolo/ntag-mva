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
from sys import argv

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

print(DSET_FRAC, DSET_FRAC_BG, DSET_FRAC_EXTRA, DSET_FRAC_BG_EXTRA)

base_dir =  MODELS_DIR + '/' + MODEL_NAME + '/'

prefix = ''
eval_mode = True
if len(argv)>1:
    prefix = argv[1] + '_'
    eval_mode = False

rocfile_test = '%s/%sroc_test.csv' % (base_dir, prefix)
outfile_test = '%s/%sbdt_test.csv' % (base_dir, prefix)

if eval_mode:
    rocfile_train = '%s/%sroc_train.csv' % (base_dir, prefix)
    outfile_train = '%s/%sbdt_train.csv' % (base_dir, prefix)
    effile_runs = '%s/%seff_runs.csv' % (base_dir, prefix)
    bgfile_runs = '%s/%sbg_runs.csv' % (base_dir, prefix)


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


def bdt_perf(x, y):
    ''' Make arrays with BDT response and efficiencies '''
    def bdt_response(hist_s, hist_b):
        ''' BDT response array '''
        # BDT output and errors
        out_s = (hist_s / s_tot) / cut_widths
        out_b = (hist_b / b_tot) / cut_widths
        err_s = eff_err_bayes(hist_s, s_tot) / cut_widths
        err_b = eff_err_bayes(hist_b, b_tot) / cut_widths

        # BDT output array
        bdt_out = np.empty((nbins, 5))  # Initialize array
        bdt_out[:, 0] = cuts
        bdt_out[:, 1] = out_s
        bdt_out[:, 2] = out_b
        bdt_out[:, 3] = err_s
        bdt_out[:, 4] = err_b
        return bdt_out

    def bdt_roc(hist_s, hist_b):
        ''' BDT efficiency and bg rate array'''
        # Invert histogram arrays and take cumulatives
        cumul_s = np.cumsum(hist_s[::-1])
        cumul_b = np.cumsum(hist_b[::-1])
        # Invert back to get complementary cumulative of BDT reponse
        # (probability to fall above BDT cut point)
        cumul_s = cumul_s[::-1]
        cumul_b = cumul_b[::-1]
        # Calculate efficiencies and errors
        eff_s = cumul_s / s_tot
        eff_b = cumul_b / b_tot
        err_s = eff_err_bayes(cumul_s, s_tot)
        err_b = eff_err_bayes(cumul_b, b_tot)

        # ROC array
        roc = np.empty((nbins, 5))  # Initialize array
        roc[:, 0] = cuts
        roc[:, 1] = eff_s
        roc[:, 2] = eff_b
        roc[:, 3] = err_s
        roc[:, 4] = err_b
        return roc

    # Separate signal and background
    x = x[:, :22]
    x_s, x_b = x[y == 1], x[y == 0]
    s_tot = sum(y)
    b_tot = len(y) - s_tot

    # Get BDT response
    print("Applying BDT...")
    pred_s = ntag_model.predict_proba(x_s, ntree_limit=NTREES)[:, 1]
    pred_b = ntag_model.predict_proba(x_b, ntree_limit=NTREES)[:, 1]

    # Bin BDT response
    hist_s, _ = np.histogram(pred_s, bins=cut_sampling)
    hist_b, _ = np.histogram(pred_b, bins=cut_sampling)
    del pred_s, pred_b

    out = bdt_response(hist_s, hist_b)
    roc = bdt_roc(hist_s, hist_b)
    return out, roc


def bdt_perf_runs(x, y):
    ''' Make array with run-dependent performance. '''
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
x_test, _, x_train, y_test, _, y_train = dset
x_test_extra = np.concatenate(extra_dset[:3])
y_test_extra = np.concatenate(extra_dset[3:])
x_test = np.concatenate((x_test, x_test_extra))
y_test = np.concatenate((y_test, y_test_extra))
del extra_dset, dset, x_test_extra, y_test_extra

print("Getting test performance...")
out_test, roc_test = bdt_perf(x_test, y_test)
if eval_mode:
    print("Getting train performance...")
    out_train, roc_train = bdt_perf(x_train, y_train)
    print("Getting time-dependent performance...")
    runeff, runbg, runlist_s, runlist_b = bdt_perf_runs(x_test, y_test)


print("Saving...")
np.savetxt(outfile_test, out_test, delimiter=", ")
np.savetxt(rocfile_test, roc_test, delimiter=", ")

if eval_mode:
    np.savetxt(outfile_train, out_train, delimiter=", ")
    np.savetxt(rocfile_train, roc_train, delimiter=", ")

    eff_head = ' | '.join([str(r) for r in runlist_s])
    bg_head = ' | '.join([str(rx[0]) + '-' + str(rx[-1]) for rx in runlist_b])
    np.savetxt(effile_runs, runeff, delimiter=", ", header=eff_head)
    np.savetxt(bgfile_runs, runbg, delimiter=", ", header=bg_head)
