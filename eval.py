from os import environ, devnull
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress avalanche of tensorflow info outputs

from numpy import array
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt 
from load_ntag import load_dset, load_model, load_hist
from metrics import plot_ROC, plot_ROC_td, plot_ROC_td_dn, plot_loss, plot_loss_hist, bdt_importance, bdt_importance_rank, plot_bdt_out

model_name = 'BDT/n10thr6_051_500k_dn2/051' # For single model plotting
model_names = [model_name] # For plotting more than one ROC at the same time
save_to = '/home/llr/t2k/giampaolo/srn/ntag-mva/models/BDT/n10thr6_051_500k_dn2'
fname = model_name.split('/')[-1]
fnames = [mn.split('/')[-1] for mn in model_names]



evaluate_on_test = True # Whether to plot ROC and output distributions on test dataset
N10TH = 6 # N10 Threshold
NFILES = 200 # Number of datafiles to use (50,000 events/file, maximum 200 files)
NFILES_TRAIN = 10

# Load models
print("Loading model(s)")
ntag_model, ntag_models = load_model(model_name), [load_model(mn) for mn in model_names]

if evaluate_on_test:
    # Load test datasets
    print("Loading test set")
    x_test, _, y_test, _ = load_dset(N10TH,file_frac=NFILES/200.)
    _, x_train, _, y_train = load_dset(N10TH, file_frac=NFILES_TRAIN/200.)

    # Plot ROC curve
    print("Plotting performance (ROC)")
    plot_ROC_td_dn(x_test,y_test,models_names_n6=zip(ntag_models,['BDT (N10>5, dark noise cut)']), 
            save_to=save_to, save_name='_'.join(fnames)+'.pdf', TMVA=True)

    # Plot BDT output distribution
    print("Plotting BDT discriminant")
    plot_bdt_out(x_test, y_test, ntag_model ,x_train=x_train, y_train=y_train,save_loc=save_to+'/'+fname+'_bdt.pdf')

# Plot loss function change
print("Plotting loss history")
hist = array(load_hist(model_name)['validation_1']['auc'])
hist_train = array(load_hist(model_name)['validation_0']['auc'])
plt.style.use("seaborn")
plt.plot(range(len(hist)), hist, label="Testing")
plt.plot(range(len(hist_train)), hist_train, label="Training")
plt.yscale("linear")
#plt.ylim(0.035,0.04)
#plt.ylim(0.98, 0.988)
plt.legend()
#plt.grid(which='both')
plt.xlabel("Training iteration")
plt.ylabel("Area Under ROC Curve (AUC)")
plt.savefig(save_to+'/'+fname+'_hist_auc.pdf')

print("Plotting feature importance ranking")
bdt_importance_rank(ntag_model, plot_location=save_to+'/'+fname+'_rank.pdf')


#ntag_hists = [load_hist(model_name) for model_name in model_names]
#plot_loss_hist(ntag_hists, model_names, 'loss')
#plot_loss_hist(ntag_hists, model_names, 'val_loss')

