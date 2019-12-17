import time
import sys
import warnings   
from pickle import load
from xgboost import XGBClassifier
with warnings.catch_warnings():  # Clean up output
    warnings.filterwarnings("ignore",category=FutureWarning)
    from load_ntag import load_dset, save_model
print('Ready') 


N10TH = 7
NFILES = 10
NTHREADS = 4


try:
    param_file = sys.argv[1]
    model_name = sys.argv[2]
except IndexError:
    raise ValueError("Usage: python3 param_search_bdt paramfile modelname")


# Load dataset
print("Loading dataset...")
x_test, x_train, y_test, y_train = load_dset(N10TH,NFILES,0.25)

print("Training BDT")
# Define neutron tagging model and start training
with open(param_file, 'rb') as fl:
    params = load(fl)
ntag_model = XGBClassifier(learning_rate=params['learning_rate'],
                          n_estimators=int(params['n_estimators']),
                          subsample=params['subsample'],
                          verbosity=1,
                          nthread=NTHREADS) 


start_time = time.time()
ntag_model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
              eval_metric='mae')
end_time = time.time()
training_dt = end_time-start_time
print("Trained model in %f seconds" % training_dt)

# Save model
print("Saving model to disk...")
save_model(ntag_model, model_name)


# Plot ROC curve

print("All done!")