import time
from load_ntag import load_dset, save_model
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from roc import plot_ROC_sigle
from loss import plot_loss


N10TH = 7
NFILES = 2
XGB = True # Whether to use XGBoost rather than regular Grad Boosting
NTREES = 850
LRATE = 0.1
SSAMPLE = 0.5
XGBstr = ('XGB_' if XGB else '')
SSstr = ('%0.2fss_'%SSAMPLE if SSAMPLE!=1 else '')
model_name = "BDT22_n%i_%itree_%0.3flrate_%s%s%i" % (N10TH,NTREES,LRATE,SSstr,XGBstr,NFILES)
NTHREADS = 4

# Load dataset
x_test, x_train, y_test, y_train = load_dset(N10TH,NFILES,0.25)

# Define neutron tagging model and start training
#ntag_model = GradientBoostingClassifier(learning_rate=1.0,max_depth=3,n_estimators=850,min_samples_leaf=0.05,random_state=1,verbose=1)
ntag_model = XGBClassifier(learning_rate=LRATE,n_estimators=NTREES,subsample=SSAMPLE,verbosity=1,nthread=NTHREADS) # XGBoost
start_time = time.time()
ntag_model.fit(x_train,y_train,eval_set=[(x_train,y_train),(x_test,y_test)],eval_metric='mae')
end_time = time.time()
training_dt = end_time-start_time
print("Trained model in %f seconds" % training_dt)

# Save model
save_model(ntag_model, model_name)
print("Saved model to disk.")

# Plot ROC curve
print("Plotting performance (ROC)")
plot_ROC_sigle(x_test,y_test,ntag_model,model_name,N10TH)
# Plot loss function change
#print("Plotting loss evolution")
#plot_loss(ntag_model,NTREES,x_test,y_test)
print("All done!")