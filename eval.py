from os import environ, devnull
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress avalanche of tensorflow info outputs

from numpy import array
import matplotlib.pyplot as plt 
from load_ntag import load_dset, load_model, load_hist
from metrics import plot_ROC, plot_loss, plot_loss_hist, bdt_importance, bdt_importance_rank

model_names = ['BDT/grid_0_extend/models/069']
save_to = '/home/llr/t2k/giampaolo/srn/ntag-mva/models/BDT/grid_0_extend/plots'

N10TH = 7 # N10 Threshold
NFILES = 50 # Number of datafiles to use (50,000 events/file, maximum 200 files)

# Load test datasets
print("Loading test set")
x_test, _, y_test, _ = load_dset(N10TH,file_frac=NFILES/200.)

# Load models
print("Loading model(s)")
ntag_models = [load_model(model_name) for model_name in model_names]

# Plot ROC curve
print("Plotting performance (ROC)")
plot_ROC(x_test,y_test,models_names_n7=zip(ntag_models,model_names), save_to=save_to, save_name='069_only.pdf', TMVA=False)

# # Plot loss function change
# print("Plotting loss history")
# hist = array(load_hist(model_names[0])['validation_1']['auc'])
# hist_train = array(load_hist(model_names[0])['validation_0']['auc'])
# plt.style.use("seaborn")
# plt.plot(range(len(hist)), hist, label="Testing")
# plt.plot(range(len(hist_train)), hist_train, label="Training")
# plt.yscale("linear")
# #plt.ylim(0.035,0.04)
# plt.ylim(0.98, 0.988)
# plt.legend()
# #plt.grid(which='both')
# plt.xlabel("Training iteration")
# #plt.ylabel("Mean Absolute Error")
# plt.savefig(save_to+'/069_hist_auc.pdf')

#print(load_hist(model_names[0])['validation_1']['auc'][-1])

#bdt_importance_rank(ntag_models[0], plot_location=save_to+'/069_rank.pdf')


#ntag_hists = [load_hist(model_name) for model_name in model_names]
#plot_loss_hist(ntag_hists, model_names, 'loss')
#plot_loss_hist(ntag_hists, model_names, 'val_loss')

