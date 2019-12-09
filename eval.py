from os import environ, devnull
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress avalanche of tensorflow info outputs

from load_ntag import load_dset, load_model, load_hist
from roc import plot_ROC
from loss import plot_loss, plot_loss_hist

model_names = ['NN\\dset_size\\NN22_n7_10epoch_256batch_2x50_5',
'NN\\dset_size\\NN22_n7_10epoch_256batch_2x50_10',
'NN\\dset_size\\NN22_n7_10epoch_256batch_2x50_20',
'NN\\dset_size\\NN22_n7_10epoch_256batch_2x50_50']

N10TH = 7 # N10 Threshold
NFILES = 10 # Number of datafiles to use (50,000 events/file, maximum 200 files)

# Load test datasets
x_test, _, y_test, _ = load_dset(N10TH,NFILES)

# Load models
ntag_models = [load_model(model_name) for model_name in model_names]
print("Loaded models from disk.")

# Plot ROC curve
print("Plotting performance (ROC)")
plot_ROC(x_test,y_test,models_names_n7=zip(ntag_models,model_names))

# Plot loss function change
print("Plotting loss history")
ntag_hists = [load_hist(model_name) for model_name in model_names]
plot_loss_hist(ntag_hists, model_names, 'loss')
plot_loss_hist(ntag_hists, model_names, 'val_loss')