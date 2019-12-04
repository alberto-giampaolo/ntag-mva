import numpy as np
import matplotlib.pyplot as plt

def binary_logistic(y,y_pred):
    ''' Compute mean absolute errors
    for binary logistic errors '''
    errors = -(y*np.log(y_pred) + (y-1)*(np.log(1-y_pred)))
    return np.mean(np.abs(errors))

def plot_loss(bdt_model,ntrees,x_test,y_test):
    '''Compute and plot test set deviance 
     as a function of training iteration (BDT models only) '''
    test_deviance = np.zeros((ntrees,), dtype=np.float64)
    for ntree in range(ntrees):
        if ntree%100: print(ntree) 
        y_pred = bdt_model.predict_proba(x_test,ntree_limit=ntree+1)[:,1]
        test_deviance[ntree] = binary_logistic(y_test,y_pred)
    
    plt.plot(range(1,ntrees+1),test_deviance)
    plt.xlabel("Training iteration")
    plt.ylabel("Loss")
    plt.title("Neutron tagging model training")
    plt.show()