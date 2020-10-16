
from __future__ import print_function
import argparse
from glob import glob
from os import environ
from math import exp

from ROOT import TChain, gSystem #pylint: disable=no-name-in-module

# Load SKOFL libraries
libs = glob(environ["skofl"] + "/lib/*.so")
for ll in libs:
    gSystem.Load(ll)

relic_mc_dir = '/disk02/usr6/giampaol/mc/srn_ntag/ntag/root'
t2k_dir = '/disk02/usr6/giampaol/data/t2k/root'
skdetsim_dir = '/disk02/usr6/giampaol/mc/srn_ntag/skdetsim'
tree_name = 'sk2p2'
tree_name2 = 'skmc'

relic_mc_window = [18, 535]
t2k_window = [-500, 500]
final_window = [5, 535]
tau = 204.8 # n cap time on H

def get_ps_eff(run_num):
    ''' Get preselection efficiency from relic MC run number. '''

    def skdfile(mcfile):
        prefix = "ntag"
        prefix_new = "skdetsim"
        fname = mcfile.split('/')[-1]
        fname_new = prefix_new + fname[len(prefix):]
        return skdetsim_dir + '/' + fname_new

    files = glob('%s/*%d*.root' % (relic_mc_dir, run_num))[:10]
    files_skdetsim = [skdfile(fl) for fl in files]
    chain = TChain(tree_name)
    chain2 = TChain(tree_name2)
    for fl, fl2 in zip(files, files_skdetsim):
        chain.Add(fl)
        chain2.Add(fl2)
    chain.AddFriend(chain2)

    tot_events = chain.GetEntries()
    ps = "N10 > 5"
    signal = "abs(dt - timeCap[0] - 900) < 200"
    n200m = "N200M < 50"
    ncaps = chain.GetEntries(" && ".join([signal, ps, n200m]))
    eff = float(ncaps) / tot_events
    eff *= exp(-final_window[0] / tau) - exp(-final_window[1] / tau)
    eff /= exp(-relic_mc_window[0] / tau) - exp(-relic_mc_window[1] / tau)
    return eff

def get_ps_bg(run_start, run_end):
    ''' Get preselection bg rate from T2K dummy
    trigger run number range (inclusive)
    '''
    def in_run_range(mcfile):
        fname = mcfile.split('/')[-1]
        nrun = fname.split('.')[2]
        nrun = int(nrun)
        return run_start <= nrun <= run_end

    files = glob('%s/*.root' % t2k_dir)
    files = [fl for fl in files if in_run_range(fl)]
    chain = TChain(tree_name)
    for fl in files:
        chain.Add(fl)

    tot_peaks = 0
    tot_events = chain.GetEntries()
    for event in range(tot_events):
        if event % 100000 == 0:
            print("Processed %d/%d" % (event, tot_events))
        chain.GetEntry(event)
        if chain.N10 < 6: continue
        if chain.N200M > 50: continue
        tot_peaks += chain.np
    bg = float(tot_peaks) / tot_events
    bg *= (final_window[1] - final_window[0])
    bg /= (t2k_window[1] - t2k_window[0])
    return bg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-b", nargs=2,
        help="T2K dummy run number range. Get preselection background rate")
    group.add_argument("-s",
        help="Relic MC run number. Get preselection efficiency.")
    args = parser.parse_args()
    bg_run_range = args.b
    s_run = args.s

    if bg_run_range:
        bg_run_range = [int(n) for n in bg_run_range]
        bg_per_event = get_ps_bg(bg_run_range[0], bg_run_range[1])
        print(bg_per_event)

    if s_run:
        s_run = int(s_run)
        pseff = get_ps_eff(s_run)
        print(pseff)
