import numpy as np
import matplotlib.pyplot as plt
from pickle import load
from sklearn.metrics import roc_curve
from xgboost import plot_importance
from scalings import * # pylint: disable=unused-wildcard-import
from load_ntag import varlist


def plot_ROC(x_test, y_test,
             models_names_n7=[], models_names_n8=[],
             SRN_scale=True, save_to=None, save_name="", TMVA=False, ntrees=None):
    '''
    Plot ROC curves for list of models on the testing dataset x_test,y_test.
    Models described by list models_names_n7 and models_names_n8, (depending on N10 threshold),
    where each element is a tuple (sklearn_model, model_name) where model_name is only used as label.
    If plotting "realistic" efficiencies for SRN analysis, use SRN_scale=True
    '''
    #plt.rcParams.update({'font.size': 20, 'axes.linewidth': 3})
    _, ax = plt.subplots()

    for model, name in models_names_n7:
        pre_eff, bg_ev = (pre_eff6_srn,bg_per_ev6_srn) if SRN_scale else (pre_eff6,bg_per_ev6)
        try:
            ntag_pred = model.predict_proba(x_test, ntree_limit=ntrees)[:,1]
        except IndexError:
            ntag_pred = model.predict(x_test, ntree_limit=ntrees)
        ntag_fp, ntag_tp, _ = roc_curve(y_test,ntag_pred)
        plt.plot(pre_eff*ntag_tp,bg_ev*ntag_fp,label=name, linewidth=1)

    for model, name in models_names_n8:
        pre_eff, bg_ev = (pre_eff7_srn,bg_per_ev7_srn) if SRN_scale else (pre_eff7,bg_per_ev7)
        try:
            ntag_pred = model.predict_proba(x_test, ntree_limit=ntrees)[:,1]
        except IndexError:
            ntag_pred = model.predict(x_test, ntree_limit=ntrees)
        ntag_fp, ntag_tp, _ = roc_curve(y_test,ntag_pred)
        plt.plot(pre_eff*ntag_tp,bg_ev*ntag_fp,label=name, linewidth=1)

    if TMVA:
        pre_eff, bg_ev = (pre_eff6_srn,bg_per_ev6_srn) if SRN_scale else (pre_eff6,bg_per_ev6)
        with open("/home/llr/t2k/giampaolo/srn/ntag-mva/models/tmva/n10thr7_nolr", 'rb') as fl:
            ntag_tp, ntag_tn = load(fl)
            ntag_tp, ntag_tn = np.array(ntag_tp), np.array(ntag_tn)
        # with open("/home/llr/t2k/giampaolo/srn/ntag-mva/models/tmva/n10thr7_opt_nolr", 'rb') as fl:
        #     ntag_opt_tp, ntag_opt_tn = load(fl)
        #     ntag_opt_tp, ntag_opt_tn = np.array(ntag_opt_tp), np.array(ntag_opt_tn)
        plt.plot(pre_eff*ntag_tp,bg_ev*(1-ntag_tn),label='BDT (TMVA)', linewidth=1)
        # plt.plot(pre_eff*ntag_opt_tp,bg_ev*(1-ntag_opt_tn),label='BDT (TMVA, optimized)', linewidth=1)

    plt.xlabel('Signal efficiency')
    plt.ylabel('Accidental coincidences/event')
    plt.legend(frameon=False, loc="best")
    plt.yscale('log')
    plt.xlim(0.05,0.45)
    plt.ylim(0.00001,1.5)
    plt.grid(which='both')
    ax.tick_params(axis="both",which='both', direction="in")
    #ax.set_facecolor('beige')
    plt.title('Neutron tagging performance')
    #ax.text(.05,.92,'Neutron tagging performance',horizontalalignment='left',transform=ax.transAxes, weight='bold')

    if save_to: 
        plt.savefig(save_to+'/'+save_name)
        plt.clf()
    else: plt.show()

def plot_ROC_td(x_test, y_test, 
                models_names_n5=[],models_names_n6=[], models_names_n7=[],models_names_n8=[],
                SRN_scale=True, save_to=None, save_name="", TMVA=False, ntrees=None):
    '''
    Plot ROC curves, for time dependent test set 
    '''
    plt.style.use("seaborn")
    _, ax = plt.subplots()

    for model, name in models_names_n5:
        pre_eff, bg_ev = (pre_eff4_td_srn, bg_per_ev4_td_srn) if SRN_scale else (pre_eff4_td, bg_per_ev4_td)
        
        ntag_pred = model.predict_proba(x_test, ntree_limit=ntrees)[:,1]
        ntag_fp, ntag_tp, _ = roc_curve(y_test,ntag_pred)
        plt.plot(pre_eff*ntag_tp,bg_ev*ntag_fp,label=name, linewidth=1)

    for model, name in models_names_n6:
        pre_eff, bg_ev = (pre_eff5_td_srn, bg_per_ev5_td_srn) if SRN_scale else (pre_eff5_td, bg_per_ev5_td)
        
        ntag_pred = model.predict_proba(x_test, ntree_limit=ntrees)[:,1]
        ntag_fp, ntag_tp, _ = roc_curve(y_test,ntag_pred)
        plt.plot(pre_eff*ntag_tp,bg_ev*ntag_fp,label=name, linewidth=1)

    for model, name in models_names_n7:
        pre_eff, bg_ev = (pre_eff6_td_srn, bg_per_ev6_td_srn) if SRN_scale else (pre_eff6_td, bg_per_ev6_td)
        
        ntag_pred = model.predict_proba(x_test, ntree_limit=ntrees)[:,1]
        ntag_fp, ntag_tp, _ = roc_curve(y_test,ntag_pred)
        plt.plot(pre_eff*ntag_tp,bg_ev*ntag_fp,label=name, linewidth=1)

    for model, name in models_names_n8:
        pre_eff, bg_ev = (pre_eff7_td_srn, bg_per_ev7_td_srn) if SRN_scale else (pre_eff7_td, bg_per_ev7_td)
        
        ntag_pred = model.predict_proba(x_test, ntree_limit=ntrees)[:,1]
        ntag_fp, ntag_tp, _ = roc_curve(y_test,ntag_pred)
        plt.plot(pre_eff*ntag_tp,bg_ev*ntag_fp,label=name, linewidth=1)

    if TMVA:
        pre_eff, bg_ev = (pre_eff6_td_srn, bg_per_ev6_td_srn) if SRN_scale else (pre_eff6_td,bg_per_ev6_td)
        with open("/home/llr/t2k/giampaolo/srn/ntag-mva/models/tmva/n10thr7_nolr_sk4", 'rb') as fl:
            ntag_tp, ntag_tn = load(fl)
            ntag_tp, ntag_tn = np.array(ntag_tp), np.array(ntag_tn)
        plt.plot(pre_eff*ntag_tp,bg_ev*(1-ntag_tn),label='Current BDT (N10 > 6)', linewidth=1)

    plt.xlabel('Signal efficiency')
    plt.ylabel('Accidental coincidences/event')
    plt.legend(frameon=False, loc="best")
    plt.yscale('log')
    plt.xlim(0.05,0.8)
    plt.ylim(0.00001,100)
    plt.grid(which='minor')
    ax.tick_params(axis="both",which='both', direction="in")
    #ax.set_facecolor('beige')
    plt.title('Neutron tagging performance')
    #ax.text(.05,.92,'Neutron tagging performance',horizontalalignment='left',transform=ax.transAxes, weight='bold')

    if save_to: 
        plt.savefig(save_to+'/'+save_name)
        plt.clf()
    else: plt.show()

def plot_ROC_td_dn(x_test, y_test, 
                models_names_n6=[], models_names_n5=[],
                SRN_scale=True, save_to=None, save_name="", TMVA=False, ntrees=None):
    '''
    Plot ROC curves, for time dependent test set with dark noise cut
    '''
    plt.style.use("seaborn")
    _, ax = plt.subplots()

    for model, name in models_names_n6:
        pre_eff, bg_ev = (pre_eff5_td_dn_srn, bg_per_ev5_td_dn_srn) if SRN_scale else (pre_eff5_td_dn, bg_per_ev5_td_dn)
        
        ntag_pred = model.predict_proba(x_test, ntree_limit=ntrees)[:,1]
        ntag_fp, ntag_tp, _ = roc_curve(y_test,ntag_pred)
        plt.plot(pre_eff*ntag_tp,bg_ev*ntag_fp,label=name, linewidth=1)

    for model, name in models_names_n5:
        pre_eff, bg_ev = (pre_eff4_td_dn_srn, bg_per_ev4_td_dn_srn) if SRN_scale else (pre_eff4_td_dn, bg_per_ev4_td_dn)
        
        ntag_pred = model.predict_proba(x_test, ntree_limit=ntrees)[:,1]
        ntag_fp, ntag_tp, _ = roc_curve(y_test,ntag_pred)
        plt.plot(pre_eff*ntag_tp,bg_ev*ntag_fp,label=name, linewidth=1)

    if TMVA:
        pre_eff, bg_ev = (pre_eff6_td_srn, bg_per_ev6_td_srn) if SRN_scale else (pre_eff6_td,bg_per_ev6_td)
        with open("/home/llr/t2k/giampaolo/srn/ntag-mva/models/tmva/n10thr7_nolr_sk4", 'rb') as fl:
            ntag_tp, ntag_tn = load(fl)
            ntag_tp, ntag_tn = np.array(ntag_tp), np.array(ntag_tn)
        plt.plot(pre_eff*ntag_tp,bg_ev*(1-ntag_tn),label='Current BDT (N10 > 6)', linewidth=1)

    plt.xlabel('Signal efficiency')
    plt.ylabel('Accidental coincidences/event')
    plt.legend(frameon=False, loc="best")
    plt.yscale('log')
    plt.xlim(0.05,0.8)
    plt.ylim(0.00001,100)
    plt.grid(which='minor')
    ax.tick_params(axis="both",which='both', direction="in")
    #ax.set_facecolor('beige')
    plt.title('Neutron tagging performance')
    #ax.text(.05,.92,'Neutron tagging performance',horizontalalignment='left',transform=ax.transAxes, weight='bold')

    if save_to: 
        plt.savefig(save_to+'/'+save_name)
        plt.clf()
    else: plt.show()

def plot_ROC_sigle(x_test, y_test, model, name, N10th, time_dep_test=False,SRN_scale=True):
    '''
    Plot ROC curve for a single model on the testing dataset x_test,y_test.
    Model described by an sklearn or Keras model and a name, where the name is only used as label.
    N10th is the N10 threshold of the training dataset.
    If using time-varying test dataset, use time_dep_test=True
    If plotting "realistic" efficiencies for SRN analysis, use SRN_scale=True
    '''
    _, ax = plt.subplots()

    try:
        ntag_pred = model.predict_proba(x_test)[:,1] # sklearn model
    except:
        ntag_pred = model.predict(x_test) # Keras model
    ntag_fp, ntag_tp, _ = roc_curve(y_test,ntag_pred)
    pre_eff, bg_ev = 1, 1
    if N10th==7:
        pre_eff, bg_ev = (pre_eff6_srn,bg_per_ev6_srn) if SRN_scale else (pre_eff6,bg_per_ev6)
    elif N10th==8:
        pre_eff, bg_ev = (pre_eff7_srn,bg_per_ev7_srn) if SRN_scale else (pre_eff7,bg_per_ev7)
    else: pass # haven't calculated other scalings
    plt.plot(pre_eff*ntag_tp,bg_ev*ntag_fp,label=name, linewidth=1)


    plt.xlabel('Signal efficiency')
    plt.ylabel('Accidental coincidences/event')
    plt.legend(frameon=False, loc="lower right")
    #plt.yscale('log')
    #plt.xlim(0.06,0.45)
    #plt.ylim(0.00001,1.5)
    plt.grid(which='both')
    ax.tick_params(axis="both",which='both', direction="in")
    ax.set_facecolor('beige')
    ax.text(.05,.92,'Neutron tagging performance',horizontalalignment='left',transform=ax.transAxes, weight='bold')
    plt.show()

def plot_ROC_sigle_gen(x_test_generator,y_test, model, name, N10th, time_dep_test=False,SRN_scale=True):
    '''
    Plot ROC curve for a single model on a dataset generator (for large datasets).
    Model described by a Keras model and a name, where the name is only used as label.
    N10th is the N10 threshold of the training dataset.
    If using time-varying test dataset, use time_dep_test=True
    If plotting "realistic" efficiencies for SRN analysis, use SRN_scale=True
    '''
    _, ax = plt.subplots()


    ntag_pred = model.predict_generator(x_test_generator)
    ntag_fp, ntag_tp, _ = roc_curve(y_test,ntag_pred)
    pre_eff, bg_ev = 1, 1
    if N10th==7:
        pre_eff, bg_ev = (pre_eff6_srn,bg_per_ev6_srn) if SRN_scale else (pre_eff6,bg_per_ev6)
    elif N10th==8:
        pre_eff, bg_ev = (pre_eff7_srn,bg_per_ev7_srn) if SRN_scale else (pre_eff7,bg_per_ev7)
    else: pass # haven't calculated other scalings
    plt.plot(pre_eff*ntag_tp,bg_ev*ntag_fp,label=name, linewidth=1)


    plt.xlabel('Signal efficiency')
    plt.ylabel('Accidental coincidences/event')
    plt.legend(frameon=False, loc="lower right")
    #plt.yscale('log')
    #plt.xlim(0.06,0.45)
    #plt.ylim(0.00001,1.5)
    plt.grid(which='both')
    ax.tick_params(axis="both",which='both', direction="in")
    ax.set_facecolor('beige')
    ax.text(.05,.92,'Neutron tagging performance',horizontalalignment='left',transform=ax.transAxes, weight='bold')
    plt.show()

def plot_bdt_out(x_test, y_test, model, x_train=None, y_train=None, save_loc='', scale='linear', ntrees=None):
    '''
    Plot BDT ouput distribution for model on given test sample.
    If given, also plot training distributions for overtraining check.
    '''
    plt.style.use("seaborn")
    nbins = 20
    if scale=='log': edges = (np.logspace(np.log10(0.001), np.log10(1.0), num=nbins+1))
    else: edges = np.linspace(0,1,nbins+1)
    # width = edges[1]-edges[0]
    width = edges[1:] - edges[:-1]

    ntag_pred = model.predict_proba(x_test, ntree_limit=ntrees)[:,1] 
    if scale == 'log': ntag_pred = 1-ntag_pred
    ntag_pred_s, ntag_pred_b = ntag_pred[y_test==1], ntag_pred[y_test==0]
    plt.hist([ntag_pred_b, ntag_pred_s], bins=edges, density= True, zorder=2,
             label=["Accidental coincidences (Test set)", "True neutrons (Test set)"])

    if x_train is not None and y_train is not None:
        ntag_pred_tr = model.predict_proba(x_train, ntree_limit=ntrees)[:,1]
        if scale == 'log': ntag_pred_tr = 1-ntag_pred_tr
        ntag_pred_tr_s, ntag_pred_tr_b = ntag_pred_tr[y_train==1], ntag_pred_tr[y_train==0]
        hs, _, = np.histogram(ntag_pred_tr_s, bins=edges, density=True)
        hb, _  = np.histogram(ntag_pred_tr_b, bins=edges, density=True)
        plt.scatter(edges[:-1]+ width*0.25, hb, label="Accidental coincidences (Train set)", 
                    edgecolors='k', zorder=3)
        plt.scatter(edges[:-1]+ width*0.75, hs, label="True neutrons (Train set)", 
                    edgecolors='k', zorder=3)

    plt.xlabel('BDT Discriminant')
    plt.ylabel('Events / (N * 0.05)')
    plt.legend()
    plt.yscale('log')
    if scale=='log': 
        plt.xscale('log')
        plt.xlabel('1 - BDT Discriminant')

    #ax.tick_params(axis="both",which='both', direction="in")

    if save_loc: 
        plt.savefig(save_loc)
        plt.clf()
    else: plt.show()

def get_bg_overtraining(x_test, y_test, model, x_train, y_train, ntrees=None):
    '''
    Get BG overtraining estimate for given test/train set and model.
    Overtraining calculated as (N_test-N_train)/N_test
    '''
    plt.style.use("seaborn")
    nbins = 20
    if scale=='log': edges = (np.logspace(np.log10(0.001), np.log10(1.0), num=nbins+1))
    else: edges = np.linspace(0,1,nbins+1)
    # width = edges[1]-edges[0]
    width = edges[1:] - edges[:-1]

    ntag_pred = model.predict_proba(x_test, ntree_limit=ntrees)[:,1] 
    if scale == 'log': ntag_pred = 1-ntag_pred
    ntag_pred_s, ntag_pred_b = ntag_pred[y_test==1], ntag_pred[y_test==0]
    plt.hist([ntag_pred_b, ntag_pred_s], bins=edges, density= True, zorder=2,
             label=["Accidental coincidences (Test set)", "True neutrons (Test set)"])

    if x_train is not None and y_train is not None:
        ntag_pred_tr = model.predict_proba(x_train, ntree_limit=ntrees)[:,1]
        if scale == 'log': ntag_pred_tr = 1-ntag_pred_tr
        ntag_pred_tr_s, ntag_pred_tr_b = ntag_pred_tr[y_train==1], ntag_pred_tr[y_train==0]
        hs, _, = np.histogram(ntag_pred_tr_s, bins=edges, density=True)
        hb, _  = np.histogram(ntag_pred_tr_b, bins=edges, density=True)
        plt.scatter(edges[:-1]+ width*0.25, hb, label="Accidental coincidences (Train set)", 
                    edgecolors='k', zorder=3)
        plt.scatter(edges[:-1]+ width*0.75, hs, label="True neutrons (Train set)", 
                    edgecolors='k', zorder=3)

    plt.xlabel('BDT Discriminant')
    plt.ylabel('Events / (N * 0.05)')
    plt.legend()
    plt.yscale('log')
    if scale=='log': 
        plt.xscale('log')
        plt.xlabel('1 - BDT Discriminant')

    #ax.tick_params(axis="both",which='both', direction="in")

    if save_loc: 
        plt.savefig(save_loc)
        plt.clf()
    else: plt.show()

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
        if ntree%100: print(ntree) 
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

def bdt_importance(xgb_bdt, varlist=varlist):
    '''
    Return dict of mapping each feature to its importance in the bdt.
    Must provide the same feature list used in training, in the same order.
    '''
    importance = xgb_bdt.feature_importances_
    if len(importance) != len(varlist): raise ValueError("Incompatible variable list")
    else: return {vr: imp for vr, imp in zip(varlist, importance)}

def bdt_importance_rank(xgb_bdt, varlist=varlist, plot_location=''):
    '''
    Print feature ranking given bdt feature importance dict
    Save to plot if plot_location given
    '''
    importance = bdt_importance(xgb_bdt, varlist)
    sorted_keys = sorted(importance, key=importance.get, reverse=True)
    print('++++++++++++++++++++++++++')
    print('+++ Feature importance +++')
    print('++++++++++++++++++++++++++')
    for i, vr in enumerate(sorted_keys):
        print('(%d) %s: %f' %(i, vr, importance[vr]))

    if plot_location:
        plot_importance(xgb_bdt, importance_type='gain', show_values=False)
        ylocs, ylabels = plt.yticks()
        ylabels_new = [varlist[int(l.get_text().strip('f'))] for l in ylabels]
        plt.yticks(ylocs, ylabels_new)
        plt.savefig(plot_location)


    
