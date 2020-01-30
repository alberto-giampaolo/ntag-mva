import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from joblib import load as jload
from pickle import load as pload
from load_ntag import load_hist
from keras.metrics import AUC

models_dir = "/home/llr/t2k/giampaolo/srn/ntag-mva/models/"
grid_subdir = "BDT/grid_0_td/"
grid_dir = models_dir + grid_subdir
params_dir = grid_dir + "params/"


print("Loading models...")
model_files = glob(grid_dir + "models/[0-9][0-9][0-9].joblib")
models_all = [jload(m) for m in model_files]
mnames = [m.split('/')[-1].split('.')[-2] for m in model_files]
param_files = [params_dir + '/' + mn + '.p' for mn in mnames]
print("Loading parameter files...")
models, params, modelID = [], [], []
for pf, m in zip(param_files, models_all):
    try:
        with open(pf, "rb") as fl:
            params += [pload(fl)]
            models += [m]
            modelID += [pf.split('/')[-1].split('.')[-2]]
    except FileNotFoundError:
        continue

print("Sorting models...")
# Testing set evaluation result of final iteration
mae = [m.evals_result()['validation_1']['mae'][-1] for m in models]
auc = [m.evals_result()['validation_1']['auc'][-1] for m in models]
# Sort with best models first
models_sorted = sorted(models,  key = lambda x: auc[models.index(x)], reverse=True)
params_sorted = sorted(params,  key = lambda x: auc[params.index(x)], reverse=True)
modelID_sorted = sorted(modelID,key = lambda x: auc[modelID.index(x)], reverse=True)
mae_sorted = sorted(mae,        key = lambda x: auc[mae.index(x)], reverse=True)
auc_sorted = sorted(auc, reverse=True)


def rank():
    n = 1
    print("***********************")
    print("**** Model Ranking ****")
    print("***********************")
    for mID, p, m, a in zip(modelID_sorted, params_sorted, mae_sorted, auc_sorted):
        if n > 200: break
        print("")
        print("(%d) [ID = %s] AUC = %f | MAE = %f"% (n,mID,a,m))
        print(p)
        n +=1

def plot_param(par):
    parlist = [p[par] for p in params]

    plt.style.use('seaborn')
    plt.scatter(parlist, 1-np.array(auc))

    plt.yscale("log")
    plt.ylabel("1-AUC")
    plt.xlabel(par)

    for i, txt in enumerate(modelID):
        if txt in modelID_sorted[:10]:
            plt.annotate(txt, (parlist[i], (1-np.array(auc))[i]))

    plt.savefig(grid_dir+'/plots/'+par+'.pdf')
    plt.clf()

def plot2params(par1, par2):
    parlist1 = [p[par1] for p in params]
    parlist2 = [p[par2] for p in params]

    plt.style.use('seaborn')
    plt.scatter(parlist1, parlist2)

    plt.ylabel(par2)
    plt.xlabel(par1)

    for i, txt in enumerate(modelID):
        if txt in modelID_sorted[:10]:
            plt.scatter(parlist1[i], parlist2[i], c='tab:red')

    plt.savefig(grid_dir+'/plots/'+par1+'_'+par2+'.pdf')
    plt.clf()

def plot_bdt_mae():
    for i, m in enumerate(models_sorted[:10]):
        maes = m.evals_result()['validation_1']['mae']
        plt.plot(range(len(maes)), maes, label=modelID_sorted[:10][i])

    plt.style.use("seaborn")
    plt.yscale("linear")
    plt.ylim(0.032,0.04)
    plt.legend()
    plt.xlabel("Training iteration")
    plt.ylabel('Mean Absolute Error (Test dataset)')
    plt.savefig(grid_dir+'/plots/top_hist_mae.pdf')
    plt.clf()

def plot_bdt_auc():
    for i, m in enumerate(models_sorted[:10]):
        aucs = m.evals_result()['validation_1']['auc']
        plt.plot(range(len(aucs)), np.array(aucs), label=modelID_sorted[:10][i])

    plt.style.use("seaborn")
    plt.yscale("linear")
    plt.ylim(0.9775, 0.9850)
    plt.legend()
    plt.xlabel("Training iteration")
    plt.ylabel('AUC (Test dataset)')
    plt.savefig(grid_dir+'/plots/top_hist_auc.pdf')
    plt.clf()

def plot_bdt_auc_bottom():
    for i, m in enumerate(models_sorted[-10:]):
        aucs = m.evals_result()['validation_1']['auc']
        plt.plot(range(len(aucs)), 1.-np.array(aucs), label=modelID_sorted[:10][i])

    plt.style.use("seaborn")
    plt.yscale("log")
    plt.ylim(0.015, 0.0175)
    plt.legend()
    plt.xlabel("Training iteration")
    plt.ylabel('AUC (Test dataset)')
    plt.savefig(grid_dir+'/plots/top_hist_auc.pdf')
    plt.clf()

# Print model ranking
rank()

# Plot parameter performance
params_to_plot = ['n_estimators', 'learning_rate', 'maximum_depth', 'subsample']
for pr in params_to_plot:
    plot_param(pr)

# Plot 2D parameter performance
for i, p1 in enumerate(params_to_plot):
    try:
        for p2 in params_to_plot[i+1:]:
            plot2params(p1,p2)
    except IndexError: break

# Plot loss history of top 10 models
plot_bdt_mae()
plot_bdt_auc()
