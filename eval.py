from load_ntag import load_dset, load_model
from roc import plot_ROC
from loss import plot_loss

model_names = ['BDT22_n8_XGB',
               'BDT22_n8_850tree_0.100lrate_0.50ss_XGB_1',
               'BDT22_n8_850tree_0.100lrate_0.50ss_XGB_10' ]
N10TH = 8 # N10 Threshold
NFILES = 1 # Number of datafiles to use (50,000 events/file, maximum 200 files)

# Load test datasets
x_test, _, y_test, _ = load_dset(N10TH,NFILES)

# Load models
ntag_models = [load_model(model_name) for model_name in model_names]
print("Loaded models from disk.")

# Plot ROC curve
#print("Plotting performance (ROC)")
plot_ROC(x_test,y_test,models_names_n8=zip(ntag_models,model_names))

# Plot loss function change
#plot_loss(ntag_models[0],850,x_test,y_test)