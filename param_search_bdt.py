import time
import sys
import warnings   
from pickle import load
from xgboost import XGBClassifier
with warnings.catch_warnings():  # Clean up output
    warnings.filterwarnings("ignore",category=FutureWarning)
    from load_ntag import load_dset, save_model

N10TH = 6
NFILES = 200. # 50k events / file
NTHREADS = -1 # Automatically make use of max cores

# Directory in which to save models and parameters (should exist inside the "models" directory)
grid_location = 'BDT/n10thr6_051/d4n_nlow/'
try:
    param_file = sys.argv[1]
except IndexError:
    raise ValueError("Usage: python3 param_search_bdt.py paramfile")
model_name = param_file.split("/")[-1].split('.')[-2] # model has same name as param file

# Load dataset
loadstart = time.time()
print("Loading dataset....")
x_test, x_train, y_test, y_train = load_dset(N10TH,file_frac=NFILES/200., test_frac=0.25, shuffle=True)
print("Loaded in", time.time() - loadstart, "seconds")

# Define neutron tagging model
print("Training BDT")
with open(param_file, 'rb') as fl:
    params = load(fl)
    print('Loaded parameter file: '+param_file)
# count_pos = sum(y_train)
# count_neg = len(y_train) - count_pos
# balance_weight = float(count_neg) / count_pos
ntag_model = XGBClassifier(learning_rate=params['learning_rate'],
                          max_depth=int(params['maximum_depth']),
                          n_estimators=2000,#n_estimators=int(params['n_estimators']),
                          subsample=0.95,#   subsample=params['subsample'],#   
                          verbosity=1,
                          nthread=NTHREADS,
                        #   scale_pos_weight=params['positive_weight']
                          )
print('Built model with ID %s with these hyperparameters:' % model_name)
print(params)

# Training
start_time = time.time()
ntag_model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
              eval_metric=['mae','aucpr','auc'],
              early_stopping_rounds=50)
end_time = time.time()
training_dt = end_time-start_time
print("Trained model with ID "+model_name+" in %f seconds" % training_dt)

# Save model
print("Saving model to disk...")
save_model(ntag_model, model_name + '', subloc=grid_location+'')
print("All done!")