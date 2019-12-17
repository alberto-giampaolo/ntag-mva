import numpy as np
from scipy.stats import expon, uniform
from pickle import dump
# Generate random search parameters
# for hyperparameter optimization

# How many times to sample the params
n_samples = 3

out_dir = "/home/llr/t2k/giampaolo/srn/ntag-mva/models/BDT/test_search/"

# BDT
ntrees = np.floor(expon(loc=10,scale=100).rvs(size=n_samples))
lrate = np.power(uniform.rvs(size=n_samples), 2)
subsample = uniform.rvs(size=n_samples)

zipped = zip(ntrees,lrate,subsample)
params = [{'learning_rate': lr,
           'n_estimators': tr,
           'subsample': ss} for tr, lr, ss in zipped]

for pi, pp in enumerate(params):
    with open(out_dir+'params/'+str(pi)+'.p', 'wb') as fl:
        dump(pp, fl)


