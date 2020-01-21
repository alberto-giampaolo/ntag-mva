import numpy as np
import matplotlib.pyplot as plt
from pickle import load
from sklearn.metrics import roc_curve
from xgboost import plot_importance
from scalings import pre_eff6_srn,bg_per_ev6_srn,pre_eff7_srn,bg_per_ev7_srn,pre_eff6,bg_per_ev6,pre_eff7,bg_per_ev7
from load_ntag import varlist


def plot_ROC(x_test, y_test, models_names_n7=[], models_names_n8=[],
             time_dep_test=False,SRN_scale=True,
             save_to=None, save_name="",
             TMVA=False):
    '''
    Plot ROC curves for list of models on the testing dataset x_test,y_test.
    Models described by list models_names_n7 and models_names_n8, (depending on N10 threshold),
    where each element is a tuple (sklearn_model, model_name) where model_name is only used as label.
    If using time-verying test dataset, use time_dep_test=True
    If plotting "realistic" efficiencies for SRN analysis, use SRN_scale=True
    '''
    #plt.rcParams.update({'font.size': 20, 'axes.linewidth': 3})
    _, ax = plt.subplots()

    for model, name in models_names_n7:
        pre_eff, bg_ev = (pre_eff6_srn,bg_per_ev6_srn) if SRN_scale else (pre_eff6,bg_per_ev6)
        try:
            ntag_pred = model.predict_proba(x_test)[:,1]
        except IndexError:
            ntag_pred = model.predict(x_test)
        ntag_fp, ntag_tp, _ = roc_curve(y_test,ntag_pred)
        plt.plot(pre_eff*ntag_tp,bg_ev*ntag_fp,label=name, linewidth=1)

    for model, name in models_names_n8:
        pre_eff, bg_ev = (pre_eff7_srn,bg_per_ev7_srn) if SRN_scale else (pre_eff7,bg_per_ev7)
        try:
            ntag_pred = model.predict_proba(x_test)[:,1]
        except IndexError:
            ntag_pred = model.predict(x_test)
        ntag_fp, ntag_tp, _ = roc_curve(y_test,ntag_pred)
        plt.plot(pre_eff*ntag_tp,bg_ev*ntag_fp,label=name, linewidth=1)

    if TMVA:
        pre_eff, bg_ev = (pre_eff6_srn,bg_per_ev6_srn) if SRN_scale else (pre_eff6,bg_per_ev6)
        with open("/home/llr/t2k/giampaolo/srn/ntag-mva/models/tmva/n10thr7_nolr", 'rb') as fl:
            ntag_tp, ntag_tn = load(fl)
            ntag_tp, ntag_tn = np.array(ntag_tp), np.array(ntag_tn)
        with open("/home/llr/t2k/giampaolo/srn/ntag-mva/models/tmva/n10thr7_opt_nolr", 'rb') as fl:
            ntag_opt_tp, ntag_opt_tn = load(fl)
            ntag_opt_tp, ntag_opt_tn = np.array(ntag_opt_tp), np.array(ntag_opt_tn)
        plt.plot(pre_eff*ntag_tp,bg_ev*(1-ntag_tn),label='BDT (TMVA)', linewidth=1)
        plt.plot(pre_eff*ntag_opt_tp,bg_ev*(1-ntag_opt_tn),label='BDT (TMVA, optimized)', linewidth=1)
    

    plt.xlabel('Signal efficiency')
    plt.ylabel('Accidental coincidences/event')
    plt.legend(frameon=False, loc="best")
    plt.yscale('log')
    plt.xlim(0.05,0.45)
    plt.ylim(0.00001,1.5)
    plt.grid(which='both')
    ax.tick_params(axis="both",which='both', direction="in")
    ax.set_facecolor('beige')
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


    
