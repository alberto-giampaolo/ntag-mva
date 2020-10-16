"""
BDT training script
"""
from __future__ import print_function
import time
from os import environ

from xgboost import XGBClassifier

from load_ntag import load_dset, save_model


N10TH = int(environ['n10th'])
DSET_FRAC = float(environ['dset'])
DSET_FRAC_BG = float(environ['dset_t2k'])
NTREES = int(environ['ntrees'])
EARLY_STOP = int(environ['early_stop'])
LRATE = float(environ['lrate'])
MAXDEPTH = int(environ['max_depth'])
SSAMPLE = float(environ['subsample'])
PWEIGHT = float(environ['pos_weight'])
MODEL_NAME = environ['model_name']
TEST_FRAC = float(environ['ntest'])
NTHREADS = int(environ['nthreads'])

# Load dataset
loadstart = time.time()
print("Dataset location:", environ['ntag_hdf5_dir_train'])
print("Using", DSET_FRAC * 100., "% of available dset, of which",
       TEST_FRAC * 100., "% is set aside")
print("for validation and", TEST_FRAC * 100., "% for testing.")
print("Preselection: N10 >=", N10TH)
print("Loading....")
ntag_dset = load_dset(N10th=N10TH, file_frac_s=DSET_FRAC,
                      file_frac_bg=DSET_FRAC_BG, test_frac=TEST_FRAC,
                      shuffle=True)
_, x_val, x_train, _, y_val, y_train = ntag_dset
print("Loaded in", time.time() - loadstart, "seconds")
print("")

# Define neutron tagging model and start training
ntag_model = XGBClassifier(learning_rate=LRATE,
                           n_estimators=NTREES,
                           max_depth=MAXDEPTH,
                           subsample=SSAMPLE,
                           scale_pos_weight=PWEIGHT,
                           verbosity=1,
                           nthread=NTHREADS)
print("BDT hyperparams:")
print( {'learning_rate': LRATE,
        'n_estimators': NTREES,
        'max_depth': MAXDEPTH,
        'subsample': SSAMPLE,
        'scale_pos_weight': PWEIGHT,} )
print("")

print("Training BDT...")
trainstart = time.time()
ntag_model.fit(x_train,y_train,
               eval_set=[(x_train,y_train),(x_val,y_val)],
               eval_metric=['mae', 'aucpr', 'auc'],
               early_stopping_rounds=EARLY_STOP)
print("Trained model in", time.time() - trainstart, "seconds")

# Save model
save_model(ntag_model, MODEL_NAME)
print("Saved model to disk.")
print("All done!")
