# Scalings for plotting ROC curves
# of ntag models for different
# MC samples and preselections
import numpy as np

n_tau = 204.8 # neutron capture characteristic time, microsecs

# Time window scaling
lin_w = 1 # Linyan's sample, lcombined sample
td_w = np.exp(-18./n_tau)-np.exp(-535./n_tau) # time-dependent sample
srn_w = np.exp(-2./n_tau)-np.exp(-535./n_tau) # SRN analysis window

# Preselection efficiency ----------------------------------------------------
pre_eff7 = 0.354     # N10>7, Linayn MC, lcombined MC
pre_eff6 = 0.474     # N10>6, Linyan MC, lcombined MC
pre_eff5 = 0.616     # N10>5, Linyan MC, lcombined MC
pre_eff7_srn = pre_eff7 * srn_w # N10>7, Linayn MC, lcombined MC, scale by SRN window
pre_eff6_srn = pre_eff6 * srn_w # N10>6, Linayn MC, lcombined MC, scale by SRN window

pre_eff4_td = 0.670 # N10>4, time dependent sample
pre_eff4_td_all = pre_eff4_td / td_w
pre_eff4_td_srn = pre_eff4_td_all * srn_w

pre_eff5_td = 0.518 # N10>5, time dependent sample
pre_eff5_td_all = pre_eff5_td / td_w
pre_eff5_td_srn = pre_eff5_td_all * srn_w

pre_eff6_td = 0.4016  # N10>6, time dependent sample
pre_eff6_td_all = pre_eff6_td / td_w # N10>6, time dependent sample. assume all events in window
pre_eff6_td_srn = pre_eff6_td_all * srn_w # Scale by SRN window size

pre_eff7_td = 0.3004 # N10>7, time dependent sample
pre_eff7_td_all = pre_eff7_td / td_w # N10>7, time dependent sample. assume all events in window
pre_eff7_td_srn = pre_eff7_td_all * srn_w # Scale by SRN window size

pre_eff5_td_dn = 0.4735 # N10>5, time dependent sample and dark noise cut
pre_eff5_td_dn_all = pre_eff5_td_dn / td_w
pre_eff5_td_dn_srn = pre_eff5_td_dn_all * srn_w
#-----------------------------------------------------------------------------

# Background rate ------------------------------------------------------------
bg_factor6 =  3.06   # N10>6 background rate compared with N10>7 cut bg rate, Linyan MC
bg_factor5 = 11.36   # N10>5 background rate compared with N10>7 cut bg rate, Linyan MC

bg_per_ev7 = 1.7766  # BG peaks/event, N10>7, Linyan MC
bg_per_ev6 = 5.4562  # BG peaks/event, N10>6, Linyan MC
bg_per_ev7_srn = bg_per_ev7 * (535.-2.)/(535.-18.) # BG peaks/ev N10>7 Linyan MC, SRN window
bg_per_ev6_srn = bg_per_ev6 * (535.-2.)/(535.-18.) # BG peaks/ev N10>6 Linyan MC, SRN window

bg_per_ev6_td = 4.9815 # BG peaks/event, N10>6, time-dependent sample
bg_per_ev7_td = 1.6377 # BG peaks/event, N10>7, time-dependent sample
bg_per_ev6_td_srn = bg_per_ev6_td * (535.-2.)/(535.-18.) # BG peaks/ev N10>6 time-dependent, SRN window
bg_per_ev7_td_srn = bg_per_ev7_td * (535.-2.)/(535.-18.) # BG peaks/ev N10>6 time-dependent, SRN window

bg_per_ev4_td = 79.44   # BG peaks/event, N10>4, time-dependent sample
bg_per_ev5_td = 17.559  # BG peaks/event, N10>5, time-dependent sample
bg_per_ev4_td_srn = bg_per_ev4_td * (535.-2.)/(535.-18.) # BG peaks/ev N10>6 time-dependent, SRN window
bg_per_ev5_td_srn = bg_per_ev5_td * (535.-2.)/(535.-18.) # BG peaks/ev N10>6 time-dependent, SRN window

bg_per_ev5_td_dn = 11.081  # BG peaks/event, N10>5, time-dependent sample with dark noise cut
bg_per_ev5_td_dn_srn = bg_per_ev5_td_dn * (535.-2.)/(535.-18.) # SRN window
# ----------------------------------------------------------------------------