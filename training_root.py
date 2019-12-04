import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve
from joblib import dump
import ROOT


N10TH = 7 # N10 >= N10TH
model_name = 'BDT22'

def get_low_id(r2,z):
    lowid = 5
    if r2 > 80: lowid -= 1
    if r2 > 130: lowid -= 1
    if r2 > 160: lowid -= 1
    if r2 > 190: lowid -= 1
    if r2 > 210: lowid -= 1
    if z > 12: lowid -= 1
    if z > 14: lowid -= 1
    if z > 15: lowid -= 1
    if lowid < 0: lowid = 0
    return lowid

t = ROOT.TChain("sk2p2")
t.Add("/media/alberto/KINGSTON/Data/shift0/000.root")
entries = t.GetEntries()

x_test, x_train = [], []
y_test, y_train = [], []
for entry in range(entries):
    t.GetEntry(entry)
    if entry%10000==0: print("%d/%d entries read"%(entry,entries))
    
    
    for p in range(t.np):
        if t.N10[p]<N10TH: continue
        pvx,pvy,pvz = t.pvx[p], t.pvy[p], t.pvz[p]
        r2 = pvx**2 + pvy**2
        nlows = [t.Nlow1[p],t.Nlow2[p],t.Nlow3[p],t.Nlow4[p],t.Nlow5[p],t.Nlow6[p],t.Nlow7[p],t.Nlow8[p],t.Nlow9[p]]
        nlowid = get_low_id(r2,pvz)
        # Discriminating variables
        varlist = [t.N10[p], t.N10d[p], t.Nc[p], nlows[nlowid],
                   t.NLowtheta[p],t.Nback[p],t.NhighQ[p],t.N300[p],
                   t.thetam[p], t.thetarms[p],t.phirms[p],t.bwall[p],t.fwall[p],
                   t.bpdist[p],t.bse[p],t.fpdist[p],t.Qrms[p],t.Qmean[p],t.trmsdiff[p],
                   t.trms[p],t.mintrms_6[p],t.mintrms_3[p]]
        is_signal = np.abs(t.dt[p] - 200.9e3) < 200
        if entry%4==0:
            x_test += [varlist]
            y_test += [int(is_signal)]
        else:
            x_train += [varlist]
            y_train += [int(is_signal)]

x_test, x_train = np.array(x_test), np.array(x_train)
y_test, y_train = np.array(y_test), np.array(y_train)

print(x_test.shape, y_test.shape, x_train.shape, y_train.shape)




# # Compare with previous results:
# # Unoptimized, optimized TMVA:
# # learning_rate = 1.0, 0.1
# # n_estimators = 850, 2000
# # max_depth= 3 , 7
# # min_samples_leaf = 150 , 150

print("Begin training...")

ntag_model = GradientBoostingClassifier(learning_rate=1.0,max_depth=3,n_estimators=850,
    min_samples_leaf=0.05,random_state=0,verbose=1, loss='deviance',subsample=1.0)
#ntag_model = GradientBoostingClassifier(learning_rate=0.1,max_depth=7,n_estimators=2000,
#    min_samples_leaf=150,random_state=0,verbose=1) # optimized
ntag_model.fit(x_train,y_train)
print("Saving model to disk...")
dump(ntag_model,"/home/alberto/SK19/ntag_algo/models/%s.joblib"%model_name)

ntag_pred = ntag_model.decision_function(x_test)
ntag_fp, ntag_tp, ntag_cuts = roc_curve(y_test,ntag_pred)
plt.plot(ntag_fp,ntag_tp,label='BDT')
plt.show()