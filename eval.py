""" Evaluate BDT and plot various metrics. """
from os import environ
from sys import argv

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

import metrics
from load_ntag import load_model, load_hist


MODEL_NAME = environ['eval_model']
MODELS_DIR = environ['models_dir']
PS_EFF = float(environ['ps_eff'])
PS_BG = float(environ['ps_bg'])


base_dir = MODELS_DIR + '/' + MODEL_NAME

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


def header_list(csv_array):
    with open(csv_array, 'r') as fl:
        header = fl.readline()
        header_cols = header.strip(' #\n').split(' | ')
    return header_cols

print("Loading model and performance tables... ")
ntag_model = load_model(MODEL_NAME)
perf = np.genfromtxt(rocfile_test, delimiter=', ')
bdt_test = np.genfromtxt(outfile_test, delimiter=', ')

if eval_mode:
    perf_train = np.genfromtxt(rocfile_train, delimiter=', ')
    bdt_train = np.genfromtxt(outfile_train, delimiter=', ')
    # eff_runs = np.genfromtxt(base_dir + '/eff_runs.csv', delimiter=', ')
    # bg_runs = np.genfromtxt(base_dir + '/bg_runs.csv', delimiter=', ')
    # ps_effs = np.genfromtxt(base_dir + '/ps_eff.txt')
    # ps_bgs = np.genfromtxt(base_dir + '/ps_bg.txt')
    # efflabels = header_list(base_dir + '/eff_runs.csv')
    # bglabels = header_list(base_dir + '/bg_runs.csv')

print("Plotting...")
print("Response function...")
if eval_mode: metrics.plot_bdt_out(bdt_test, bdt_train=bdt_train)
else: metrics.plot_bdt_out(bdt_test)
plt.xlim(1e-4, 1.3)
plt.savefig('%s/%sbdt.pdf' % (base_dir, prefix))

print("ROC curve...")
_, ax = plt.subplots()
metrics.plot_ROC(perf, PS_EFF, PS_BG, label=MODEL_NAME, errs=True,
                 linewidth=2)
# plt.xlim(0.05,0.8)
plt.ylim(0.000001,10)
ax.text(.3, .5, 'Preliminary', horizontalalignment='left',
        transform=ax.transAxes, weight='bold')
plt.savefig('%s/%sroc.pdf' % (base_dir, prefix))
plt.clf()

if eval_mode:
    print("Efficiency and background rate...")
    _, ax = plt.subplots()
    plt.xlim(1e-4, 1)
    metrics.plot_eff(perf, PS_EFF, label=MODEL_NAME, errs=True)
    ax.text(.3, .5, 'Preliminary', horizontalalignment='left',
            transform=ax.transAxes, weight='bold')
    plt.savefig(base_dir + '/eff.pdf')
    plt.clf()

    _, ax = plt.subplots()
    metrics.plot_bg(perf, PS_BG, label=MODEL_NAME, errs=True)
    plt.xlim(1e-4, 1)
    plt.ylim(1e-6, 1)
    ax.text(.3, .5, 'Preliminary', horizontalalignment='left',
            transform=ax.transAxes, weight='bold')
    plt.savefig(base_dir + '/bg.pdf')
    plt.clf()

    _, ax = plt.subplots()
    metrics.plot_eff_bg(perf, PS_EFF, PS_BG,
                        perf_array_train=perf_train, errs=True)
    plt.xlim(1e-4, 1)
    plt.ylim(1e-6, 1)
    ax.text(.3, .5, 'Preliminary', horizontalalignment='left',
            transform=ax.transAxes, weight='bold')
    plt.savefig(base_dir + '/eff_bg.pdf')
    plt.clf()

    print("Training history...")
    hist = np.array(load_hist(MODEL_NAME)['validation_1']['auc'])
    hist_train = np.array(load_hist(MODEL_NAME)['validation_0']['auc'])
    plt.plot(range(len(hist)), hist, label="Validation")
    plt.plot(range(len(hist_train)), hist_train, label="Training")
    plt.yscale("linear")
    #plt.ylim(0.035,0.04)
    #plt.ylim(0.98, 0.988)
    plt.legend()
    #plt.grid(which='both')
    plt.xlabel("Training iteration")
    plt.ylabel("Area Under ROC Curve (AUC)")
    plt.savefig(base_dir+'/hist_auc.pdf')

    print("Feature importance ranking...")
    metrics.bdt_importance_rank(ntag_model)
    plt.savefig(base_dir+'/rank.pdf')
    plt.clf()


    # print("Run-by-run efficiency...")
    # _, ax = plt.subplots()
    # plt.xlim(1e-4, 1)
    # ax.text(.3, .5, 'Preliminary', horizontalalignment='left',
    #         transform=ax.transAxes, weight='bold')
    # for i in range((eff_runs.shape[1]-1)//2):
    #     eff_array = eff_runs[:,[0, i*2+1, i*2+1, i*2+2, i*2+2]]
    #     metrics.plot_eff(eff_array, ps_effs[i],
    #     label=efflabels[i], errs=False, c=None)
    # plt.savefig(base_dir + '/eff_runs.pdf')
    # plt.clf()

    # print("Run-by-run mistag rate...")
    # _, ax = plt.subplots()
    # ax.text(.3, .5, 'Preliminary', horizontalalignment='left',
    #         transform=ax.transAxes, weight='bold')
    # for i in range((bg_runs.shape[1]-1)//2):
    #     bg_array = bg_runs[:,[0, i*2+1, i*2+1, i*2+2, i*2+2]]
    #     metrics.plot_bg(bg_array, ps_bgs[i], label=bglabels[i], errs=True, c=None)
    # plt.xlim(1e-4, 1)
    # plt.ylim(1e-6, 1)
    # plt.savefig(base_dir + '/bg_runs.pdf')
    # plt.clf()



    # eff_single_runs = [int(r) for r in efflabels]
    # bg_run_ranges = [r.split('-') for r in bglabels]
    # bg_run_ranges = [[int(r1), int(r2)] for r1, r2 in bg_run_ranges]
    # eff_ids, bg_ids = [], []
    # for i, r in enumerate(eff_single_runs):
    #     for j, (r1, r2) in enumerate(bg_run_ranges):
    #         if r1 <= r <= r2:
    #             eff_ids.append(i)
    #             bg_ids.append(j)
    #             continue
    # print("Run-by-run ROC curves...")
    # wd, ht = matplotlib.rcParams["figure.figsize"]
    # _, (ax, ax2) = plt.subplots(2, 1, sharex=False, 
    #                gridspec_kw={'height_ratios': [2, 1]}, figsize=[wd, ht*1.6],
    #                tight_layout=False)
    # plt.axes(ax)
    # for i, j in zip(eff_ids, bg_ids):
    #     run_arr = np.hstack((eff_runs[:,[0, i*2+1]], bg_runs[:, j*2+1][:, None]))
    #     metrics.plot_ROC(run_arr, ps_effs[i], ps_bgs[j],
    #                      label=efflabels[i], errs=False, c=None, linewidth=2)
    # plt.ylim(0.000001,10)
    # plt.xlim(0, 0.5)
    # ax.text(.3, .5, 'Preliminary', horizontalalignment='left',
    #         transform=ax.transAxes, weight='bold')

    # bg_std = np.std(bg_runs[:, np.array(bg_ids)*2+1], axis=1)
    # bg_mean = np.mean(bg_runs[:, np.array(bg_ids)*2+1], axis=1)
    # eff_mean = np.mean(eff_runs[:, np.array(eff_ids)*2+1], axis=1)
    # bg_run_err = bg_std / bg_mean
    # plt.axes(ax2)
    # plt.plot(eff_mean * np.mean(ps_effs), bg_run_err)
    # plt.ylim(0, 0.4)
    # plt.xlim(0, 0.5)
    # plt.xlabel("Signal Efficiency")
    # plt.ylabel("STD(B)/mean(B)")
    # plt.minorticks_on()
    # plt.grid(which='minor')
    # plt.savefig(base_dir + '/roc_runs.pdf')
    # plt.clf()