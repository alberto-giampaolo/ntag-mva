
""" Plot ROC curves from csv """
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import interpolate

import metrics


rocfiles_gd = ['/disk02/usr6/giampaol/ntag-mva/models/bdt22_n10thr6_test_gd_gamma/roc_test.csv',
'/disk02/usr6/giampaol/ntag-mva/models/bdt22_n10thr6_test_gdmix_gamma/pureGd_roc_test.csv',
'/disk02/usr6/giampaol/ntag-mva/models/sk4_full_1500/pureGd_roc_test.csv']
labels_gd = ['Trained on pure Gd', 'Trained on 0.01%% Gd', 'Trained on pure water']

rocfiles_water = ['/disk02/usr6/giampaol/ntag-mva/models/bdt22_n10thr6_test_gd_gamma/pureWater_roc_test.csv',
'/disk02/usr6/giampaol/ntag-mva/models/bdt22_n10thr6_test_gdmix_gamma/pureWater_roc_test.csv',
'/disk02/usr6/giampaol/ntag-mva/models/sk4_full_1500/roc_test.csv']
labels_water = ['Trained on pure Gd', 'Trained on 0.01%% Gd', 'Trained on pure water']

rocfiles_mix = ['/disk02/usr6/giampaol/ntag-mva/models/bdt22_n10thr6_test_gd_gamma/Gdmix_roc_test.csv',
'/disk02/usr6/giampaol/ntag-mva/models/bdt22_n10thr6_test_gdmix_gamma/roc_test.csv',
'/disk02/usr6/giampaol/ntag-mva/models/sk4_full_1500/Gdmix_roc_test.csv']
labels_mix = ['Trained on pure Gd', 'Trained on 0.01%% Gd', 'Trained on pure water']

ps_gd = 0.912
ps_mix = 0.725
ps_water = 0.515
ps_bg = 11.4


plt.style.use("seaborn")
_, ax = plt.subplots()
for roc, lb in zip(rocfiles_gd, labels_gd):
    perf = np.genfromtxt(roc, delimiter=', ')
    metrics.plot_ROC(perf, ps_gd, ps_bg, label=lb, errs=True,
                    linewidth=2, color=None)
plt.ylim(0.000001,10)
ax.text(.3, .5, 'Performance on pure Gd (EGLO model n caupture)', horizontalalignment='left',
        transform=ax.transAxes, weight='bold')
plt.savefig('plots/gd_roc.pdf')
plt.clf()


_, ax = plt.subplots()
for roc, lb in zip(rocfiles_water, labels_water):
    perf = np.genfromtxt(roc, delimiter=', ')
    metrics.plot_ROC(perf, ps_water, ps_bg, label=lb, errs=True,
                    linewidth=2, color=None)
plt.ylim(0.000001,10)
ax.text(.5, .3, 'Performance on pure water', horizontalalignment='left',
        transform=ax.transAxes, weight='bold')
plt.savefig('plots/water_roc.pdf')
plt.clf()


_, ax = plt.subplots()
for roc, lb in zip(rocfiles_mix, labels_mix):
    perf = np.genfromtxt(roc, delimiter=', ')
    metrics.plot_ROC(perf, ps_mix, ps_bg, label=lb, errs=True,
                    linewidth=2, color=None)

gd_perf = np.genfromtxt(rocfiles_gd[0], delimiter=', ')
water_perf = np.genfromtxt(rocfiles_water[2], delimiter=', ')
gd_effs = gd_perf[:, 1] * ps_gd
gd_bgs = gd_perf[:, 2] * ps_bg
gd_bgerrs = gd_perf[: 4] * ps_bg

water_effs = water_perf[:, 1] * ps_water
water_bgs = water_perf[:, 2] * ps_bg
water_bgerrs = water_perf[: 4] * ps_bg
water_eff_interp = interpolate.interp1d(water_bgs, water_effs)
water_effs_new = water_eff_interp(gd_bgs)
gd_eff_interp = interpolate.interp1d(gd_bgs, gd_effs)
gd_effs_new = gd_eff_interp(water_bgs)

combined_eff_mix = (gd_effs_new + water_effs) / 2.0
combined_eff_mix /= (ps_water + ps_gd) / 2.0
combined_eff_mix *= ps_mix
plt.plot(combined_eff_mix, water_bgs, label="Use two separately-trained BDTs",
        linestyle='dashed', color='k')
plt.legend(frameon=False, loc='best')
plt.ylim(0.000001,10)
ax.text(.1, .5, 'Performance on 0.01%% Gd (EGLO model n capture)', horizontalalignment='left',
        transform=ax.transAxes, weight='bold')
plt.savefig('plots/gdmix_roc.pdf')
plt.clf()
