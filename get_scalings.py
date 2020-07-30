import numpy as np
import h5py
from glob import glob

def eff_err_bayes(k,n):
    ''' Bayesian error on efficiency k/n assuming a flat prior '''
    var1 = float((k+1)*(k+2)) / ((n+2)*(n+3))
    var2 = float((k+1)*(k+1)) / ((n+2)*(n+2))
    return np.sqrt(var1-var2)

def rate_err_poisson(k,n):
    ''' Poissonian error on rate k/n'''
    err_k = np.sqrt(float(k))
    err_n = np.sqrt(float(n))
    err = (err_k/k)**2 + (err_n/n)**2
    return np.sqrt(err) * float(k)/n

def get_scalings(run=None):
    if run: ntags = glob("/data_CMS/cms/giampaolo/mc/darknoise4/hdf5_flat/*r"+str(int(run))+"*")
    else: ntags = glob("/data_CMS/cms/giampaolo/mc/darknoise4/hdf5_flat/*")

    n10s = []
    n10b = []
    total_events = 0
    for n in ntags:
        if int(n.split('.')[-2]) % 1 == 0:
            with h5py.File(n) as fl:
                dset = fl['sk2p2']
                n10 = dset['N10']
                is_signal = dset['is_signal']
                if len(n10) == 0: continue
                # print(n)
                n10s += list(n10[is_signal == 1])
                n10b += list(n10[is_signal == 0])
                total_events += dset[-1]['event_num'] +1

    n10s = np.array(n10s)
    n10b = np.array(n10b)
    n5_e = len(n10s) / float(total_events)
    err_e5 = eff_err_bayes(len(n10s), total_events)
    n5_bgr = len(n10b) / float(total_events)
    err_bgr5 = rate_err_poisson(len(n10b), total_events)

    n6 = n10s[n10s > 5]
    n6b = n10b[n10b > 5]
    n6_e = len(n6) / float(total_events)
    err_e6 = eff_err_bayes(len(n6), total_events)
    n6_bgr = len(n6b) / float(total_events)
    err_bgr6 = rate_err_poisson(len(n6), total_events)

    n7 = n10s[n10s > 6]
    n7b = n10b[n10b > 6]
    n7_e = len(n7) / float(total_events)
    err_e7 = eff_err_bayes(len(n7), total_events)
    n7_bgr = len(n7b) / float(total_events)
    err_bgr7 = rate_err_poisson(len(n7), total_events)


    print('')
    if run: print("Efficiencies for run", run)
    else: print("*** Overall efficiencies ***")
    print("Total events:", total_events)
    print('')
    print("SIGNAL EVENTS (efficiency):")
    print("N10>4:", len(n10s), "(%0.4f +/- %0.4f)" % (n5_e, err_e5))
    print("N10>5:", len(n6), "(%0.4f +/- %0.4f)" % (n6_e, err_e6))
    print("N10>6:", len(n7), "(%0.4f +/- %0.4f)" % (n7_e, err_e7))
    print('')
    print("BG EVENTS (BG / event):")
    print("N10>4:", len(n10b), "(%0.4f +/- %0.4f)" % (n5_bgr, err_bgr5))
    print("N10>5:", len(n6b), "(%0.4f +/- %0.4f)" % (n6_bgr, err_bgr6))
    print("N10>6:", len(n7b), "(%0.4f +/- %0.4f)" % (n7_bgr, err_bgr7))
    print('')
    print('')

if __name__ == "__main__":
    runs = [63388, 66491, 67977, 69353, 70994, 72012, 73019, 74006, 75013, 76804]
    for r in runs:
        get_scalings(r)
    get_scalings()
