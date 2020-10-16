""" Functions for BDT evaluation and plotting """
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_importance

from load_ntag import varlist

def smooth_step(x, y):
    # ids = [i for i, _ in enumerate(y[:-1]) if y[i] != y[i+1]]
    # return np.array(x)[ids], np.array(y)[ids]
    return x, y

def plot_ROC(perf_array, ps_eff, ps_bg, errs=False, label='', **kwargs):
    """ Plot ROC curve from csv, given preselection efficiencies. """
    effs = perf_array[:, 1] * ps_eff
    bgs = perf_array[:, 2] * ps_bg
    if errs:
        bg_errs = perf_array[:, 4] * ps_bg
        plt.fill_between(effs, bgs + bg_errs, bgs - bg_errs,
                         alpha=0.5, **kwargs)
    plt.plot(effs, bgs, label=label, **kwargs)
    plt.xlabel('Signal efficiency')
    plt.ylabel('Accidental coincidences/event')
    plt.legend(frameon=False, loc="best")
    plt.yscale('log')
    plt.minorticks_on()
    plt.grid(True, which='minor')
    # ax.tick_params(axis="both", which='both', direction="in")
    plt.title("Neutron tagging BDT ROC curve")


def plot_eff(perf_array, ps_eff, errs=False, label='', c='tab:blue'):
    """ Plot efficiency curve from csv, given preselection efficiency. """
    cuts = perf_array[:, 0]
    effs = perf_array[:, 1] * ps_eff
    if errs:
        eff_errs = perf_array[:, 3] * ps_eff
        errcuts, errhi = smooth_step(1. - cuts, effs + eff_errs)
        _, errlo = smooth_step(1. - cuts, effs - eff_errs)
        plt.fill_between(errcuts, errhi, errlo,
                         color=c, alpha=0.5)
    plt.plot(1. - cuts, effs, label=label, color=c, linewidth=2)
    plt.legend(frameon=False, loc="best")
    plt.xlabel('1 - BDT cut')
    plt.ylabel('Signal efficiency')
    plt.title("Neutron tagging BDT efficiency")
    plt.xscale('log')
    plt.minorticks_on()
    plt.grid(True, which='minor')


def plot_bg(perf_array, ps_bg, errs=False, label='', c='tab:blue'):
    """ Plot bg rate curve from csv, given preselection efficiency. """
    cuts_old = perf_array[:, 0]
    bgs = perf_array[:, 2] * ps_bg
    cuts, bgs = smooth_step(cuts_old, bgs)
    if errs:
        bg_errs = perf_array[:, 4] * ps_bg
        _, bg_errs = smooth_step(cuts_old, bg_errs)
        plt.fill_between(1. - cuts, bgs + bg_errs, bgs - bg_errs, color=c,
                         alpha=0.5)

    # if errs:
    #     bg_errs = perf_array[:, 4] * ps_bg
    #     errcuts, errhi = smooth_step(1. - cuts, bgs + bg_errs)
    #     _, errlo = smooth_step(1. - cuts, bgs - bg_errs)
    #     plt.fill_between(errcuts, errhi, errlo,
    #                      color=c, alpha=0.5)

    plt.plot(1. - cuts, bgs, label=label, color=c, linewidth=2)
    plt.xlabel('1 - BDT cut')
    plt.ylabel('Accidental coincidences / event')
    plt.title("Neutron tagging BDT background rate")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(frameon=False, loc="best")
    plt.minorticks_on()
    plt.grid(True, which='minor')

def plot_eff_bg(perf_array, ps_eff, ps_bg, perf_array_train=None, errs=False):
    """ Efficiency and bg rate on the same plot. """
    cuts = perf_array[:, 0]
    effs = perf_array[:, 1] * ps_eff
    bgs = perf_array[:, 2] * ps_bg
    if errs:
        eff_errs = perf_array[:, 3] * ps_eff
        bg_errs = perf_array[:, 4] * ps_bg
        plt.fill_between(1. - cuts, effs + eff_errs, effs - eff_errs,
                         color='tab:blue', alpha=0.5)
        plt.fill_between(1. - cuts, bgs + bg_errs, bgs - bg_errs,
                         color='tab:red', alpha=0.5)
    if perf_array_train is not None:
        cuts_train = perf_array_train[:, 0]
        effs_train = perf_array_train[:, 1] * ps_eff
        bgs_train = perf_array_train[:, 2] * ps_bg
        if errs:
            eff_errs_train = perf_array_train[:, 3] * ps_eff
            bg_errs_train = perf_array_train[:, 4] * ps_bg
            plt.fill_between(1. - cuts_train, effs_train + eff_errs_train,
                             effs_train - eff_errs_train,
                             color='tab:orange', alpha=0.5)
            plt.fill_between(1. - cuts_train, bgs_train + bg_errs_train,
                             bgs_train - bg_errs_train,
                             color='k', alpha=0.5)
        plt.plot(1. - cuts_train, effs_train, color='tab:orange', linewidth=2,
             label='Signal efficiency (train)')
        plt.plot(1. - cuts_train, bgs_train, color='k', linewidth=2,
             label="Accidental coincidences/event (train)")

    plt.plot(1. - cuts, effs, color='tab:blue', linewidth=2,
             label='Signal efficiency (test)')
    plt.plot(1. - cuts, bgs, color='tab:red', linewidth=2,
             label="Accidental coincidences/event (test)")
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('1 - BDT cut')
    plt.title("Neutron tagging BDT background rate")
    plt.legend(frameon=False, loc="best")
    plt.minorticks_on()
    plt.grid(True, which='minor')


def plot_bdt_out(bdt_test, bdt_train=None, nbins=50):
    '''
    Plot BDT ouput distribution from csv file.
    If given, also plot training distributions for overtraining check.
    '''
    def sum_along(ar, ids):
        ''' Array of sums according to indices'''
        return [sum(ar[ids[i]:ids[i + 1]]) for i in range(len(ids[:-1]))]

    plt.style.use("seaborn")

    cuts = bdt_test[:, 0]
    bdt_hist_s = bdt_test[:, 1]
    bdt_hist_b = bdt_test[:, 2]

    bin_ids = np.arange(nbins) * np.floor(len(cuts) / nbins)
    bin_ids = bin_ids.astype(int)
    bins = np.take(cuts, bin_ids.astype(int), mode='raise')
    bin_widths = np.append(bins[1:], 1.0) - bins
    rebin_s = sum_along(bdt_hist_s, np.append(bin_ids, -1))
    rebin_b = sum_along(bdt_hist_b, np.append(bin_ids, -1))

    plt.bar(1.0 - bins + bin_widths/2., rebin_s, width=bin_widths/2.,
            align='edge', label='Neutron captures (test)')
    plt.bar(1.0 - bins, rebin_b, width=bin_widths/2.,
            align='edge', label='Accidentals (test)')

    if bdt_train is not None:
        cuts_train = bdt_train[:, 0]
        bdt_hist_s_train = bdt_train[:, 1]
        bdt_hist_b_train = bdt_train[:, 2]
        assert (cuts_train == cuts).all()

        rebin_s_train = sum_along(bdt_hist_s_train, np.append(bin_ids, -1))
        rebin_b_train = sum_along(bdt_hist_b_train, np.append(bin_ids, -1))
        plt.scatter(1.0 - bins + bin_widths*3/4., rebin_s_train, marker='x',
                    c='k', label='Neutron captures (train)', zorder=3)
        plt.scatter(1.0 - bins + bin_widths*1/4., rebin_b_train, marker='+',
                    c='k', label='Accidentals (train)', s=50,  zorder=3)

    plt.xlabel('BDT Discriminant')
    plt.ylabel('Events / N')
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('1 - BDT Discriminant')


# def get_bg_overtraining(x_test, y_test, model, x_train, y_train, ntrees=None):
#     '''
#     Get BG overtraining estimate for given test/train set and model.
#     Overtraining calculated as (N_test-N_train)/N_test
#     '''
#     plt.style.use("seaborn")
#     nbins = 20
#     if scale=='log': edges = (np.logspace(np.log10(0.001), np.log10(1.0), num=nbins+1))
#     else: edges = np.linspace(0,1,nbins+1)
#     # width = edges[1]-edges[0]
#     width = edges[1:] - edges[:-1]

#     ntag_pred = model.predict_proba(x_test, ntree_limit=ntrees)[:,1]
#     if scale == 'log': ntag_pred = 1-ntag_pred
#     ntag_pred_s, ntag_pred_b = ntag_pred[y_test==1], ntag_pred[y_test==0]
#     plt.hist([ntag_pred_b, ntag_pred_s], bins=edges, density= True, zorder=2,
#              label=["Accidental coincidences (Test set)", "True neutrons (Test set)"])

#     if x_train is not None and y_train is not None:
#         ntag_pred_tr = model.predict_proba(x_train, ntree_limit=ntrees)[:,1]
#         if scale == 'log': ntag_pred_tr = 1-ntag_pred_tr
#         ntag_pred_tr_s, ntag_pred_tr_b = ntag_pred_tr[y_train==1], ntag_pred_tr[y_train==0]
#         hs, _, = np.histogram(ntag_pred_tr_s, bins=edges, density=True)
#         hb, _  = np.histogram(ntag_pred_tr_b, bins=edges, density=True)
#         plt.scatter(edges[:-1]+ width*0.25, hb, label="Accidental coincidences (Train set)",
#                     edgecolors='k', zorder=3)
#         plt.scatter(edges[:-1]+ width*0.75, hs, label="True neutrons (Train set)",
#                     edgecolors='k', zorder=3)

#     plt.xlabel('BDT Discriminant')
#     plt.ylabel('Events / (N * 0.05)')
#     plt.legend()
#     plt.yscale('log')
#     if scale=='log':
#         plt.xscale('log')
#         plt.xlabel('1 - BDT Discriminant')

    #ax.tick_params(axis="both",which='both', direction="in")

    # if save_loc:
    #     plt.savefig(save_loc)
    #     plt.clf()
    # else: plt.show()

def binary_logistic(y,y_pred):
    ''' Compute mean absolute errors
    for binary logistic errors '''
    errors = -(y*np.log(y_pred) + (y-1)*(np.log(1-y_pred)))
    return np.mean(np.abs(errors))

def plot_loss(bdt_model,ntrees,x_test,y_test):
    '''Compute and plot test set deviance
     as a function of training iteration (BDT models only) '''
    test_deviance = np.zeros((ntrees,), dtype=np.float64)
    for ntree in range(ntrees):
        if ntree % 100:
            print(ntree)
        y_pred = bdt_model.predict_proba(x_test,ntree_limit=ntree+1)[:,1]
        test_deviance[ntree] = binary_logistic(y_test,y_pred)

    plt.plot(range(1,ntrees+1),test_deviance)
    plt.xlabel("Training iteration")
    plt.ylabel("Loss")
    plt.title("Neutron tagging model training")
    plt.show()

def plot_loss_hist(hists, labels, metric):
    '''
    Plot deviance given a history object
    '''
    for h, l in zip(hists, labels):
        plt.plot(h.history[metric], label=l)
    plt.ylabel('Loss')
    plt.xlabel('Training epoch')
    plt.yscale('log')
    plt.legend()
    plt.show()

def bdt_importance(xgb_bdt, bdtvars=varlist):
    '''
    Return dict of mapping each feature to its importance in the bdt.
    Must provide the same feature list used in training, in the same order.
    '''
    importance = xgb_bdt.feature_importances_
    if len(importance) != len(bdtvars):
        raise ValueError("Incompatible variable list")
    else:
        return {vr: imp for vr, imp in zip(bdtvars, importance)}

def bdt_importance_rank(xgb_bdt, bdtvars=varlist, print_rank=False):
    '''
    Print feature ranking given bdt feature importance dict
    Save to plot if plot_location given
    '''
    importance = bdt_importance(xgb_bdt, bdtvars)
    if print_rank:
        sorted_keys = sorted(importance, key=importance.get, reverse=True)
        print('++++++++++++++++++++++++++')
        print('+++ Feature importance +++')
        print('++++++++++++++++++++++++++')
        for i, vr in enumerate(sorted_keys):
            print('(%d) %s: %f' %(i, vr, importance[vr]))

    plot_importance(xgb_bdt, importance_type='gain', show_values=False)
    ylocs, ylabels = plt.yticks()
    ylabels_new = [bdtvars[int(l.get_text().strip('f'))] for l in ylabels]
    plt.yticks(ylocs, ylabels_new)
