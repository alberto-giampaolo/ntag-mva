import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scalings import pre_eff6_srn,bg_per_ev6_srn,pre_eff7_srn,bg_per_ev7_srn,pre_eff6,bg_per_ev6,pre_eff7,bg_per_ev7


def plot_ROC(x_test, y_test, models_names_n7=[], models_names_n8=[], time_dep_test=False,SRN_scale=True):
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
        ntag_pred = model.predict_proba(x_test)[:,1]
        ntag_fp, ntag_tp, _ = roc_curve(y_test,ntag_pred)
        plt.plot(pre_eff*ntag_tp,bg_ev*ntag_fp,label=name, linewidth=1)
    for model, name in models_names_n8:
        pre_eff, bg_ev = (pre_eff7_srn,bg_per_ev7_srn) if SRN_scale else (pre_eff7,bg_per_ev7)
        try:
            ntag_pred = model.predict_proba(x_test)[:,1]
        except:
            ntag_pred = model.predict(x_test)
        ntag_fp, ntag_tp, _ = roc_curve(y_test,ntag_pred)
        plt.plot(pre_eff*ntag_tp,bg_ev*ntag_fp,label=name, linewidth=1)

    plt.xlabel('Signal efficiency')
    plt.ylabel('Accidental coincidences/event')
    plt.legend(frameon=False, loc="best")
    plt.yscale('log')
    #plt.xlim(0.06,0.45)
    #plt.ylim(0.00001,1.5)
    plt.grid(which='both')
    ax.tick_params(axis="both",which='both', direction="in")
    ax.set_facecolor('beige')
    plt.title('Neutron tagging performance')
    #ax.text(.05,.92,'Neutron tagging performance',horizontalalignment='left',transform=ax.transAxes, weight='bold')
    plt.show()

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

