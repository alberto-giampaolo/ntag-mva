import numpy as np
from scipy.stats import expon, uniform, randint
from pickle import dump
# Generate random search parameters
# for hyperparameter optimization

n_samples = 150 # How many times to sample the params
out_dir = "/home/llr/t2k/giampaolo/srn/ntag-mva/models/BDT/grid_2/" # parameter dictionaries are stored in out_dir/params
mode = 'BDT' # 'BDT'|'NN'

if mode=='BDT':
    """ ntrees = np.floor(expon(loc=1800,scale=500).rvs(size=n_samples)) 
    lrate = uniform.rvs(loc=0.01, scale=0.3, size=n_samples)
    subsample = uniform.rvs(loc=0.2, scale=0.6, size=n_samples)
    depth = randint.rvs(low=5,high=11, size=n_samples) """

    ntrees = []
    lrate = uniform.rvs(loc=0.01, scale=0.3, size=n_samples)
    subsample = uniform.rvs(loc=0.2, scale=0.6, size=n_samples)
    depth = randint.rvs(low=5,high=11, size=n_samples)


    zipped = zip(ntrees,lrate,subsample, depth)
    params = [{'learning_rate': lr,
            'n_estimators': tr,
            'subsample': ss,
            'maximum_depth': dp} for tr, lr, ss, dp in zipped]

elif mode=='NN':
    epochs = (np.floor(expon(loc=2,scale=3).rvs(size=n_samples))).astype(int) # 20 300
    dropout = uniform.rvs(size=n_samples) 
    depth = (randint.rvs(low=1,high=3, size=n_samples)).astype(int) # 3 11
    width = (np.floor(expon(loc=3,scale=6).rvs(size=n_samples))).astype(int) # 10 40

    optimizers = ['adam', 'sgd', 'nadam', 'adadelta']
    optimizer = [np.choose(randint.rvs(low=0,high=4,size=1), optimizers)[0] for i in range(n_samples)]
    lrate = np.power(uniform.rvs(size=n_samples), 2)

    zipped = zip(epochs,dropout,depth,width,lrate,optimizer)
    params = [{'epochs': ep,
            'dropout': do,
            'depth': dp,
            'width': wd,
            'learning_rate': lr,
            'optimizer': op} for ep, do, dp, wd, lr, op in zipped]


for pi, pp in enumerate(params):
    with open(out_dir+'params/%03i.p'%pi, 'wb') as fl:
        dump(pp, fl)


