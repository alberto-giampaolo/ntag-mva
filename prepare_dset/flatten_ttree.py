"""
Script to flatten ntag root file for BDT training and calculate
additional variables. Needs ncap timing information to determing
whether candidate is a true signal.
Needs original skdetsim file in case
of variable ncap time, or the ncap time as argument if fixed. If neither is
provided (i.e. for pure bg samples), label all candidates as background.

Usage: python2 flatten_ttree.py [-h] [-mc MC | -ct CT] input_file output_file

positional arguments:
  input_file   ntagged ROOT file
  output_file  flattened ROOT file

optional arguments:
  -h, --help   show this help message and exit
  -mc MC       skdetsim file with capture time info
  -ct CT       fixed capture time
"""
from __future__ import print_function
import argparse
from os import environ
from glob import glob
from array import array
from math import isnan

from ROOT import TFile, TTree, TChain, gSystem #pylint: disable=no-name-in-module


# Load SKOFL libraries
libs = glob(environ["skofl"] + "/lib/*.so")
for ll in libs:
    gSystem.Load(ll)

tree_name = "sk2p2"

# SKdetsim tree with truth info on same events
tree_name2 = "skmc"

def merge_dicts(dicts):
    ''' Merge list of dicts into one dict. '''
    merged = dicts[0].copy()
    for d in dicts[1:]:
        merged.update(d)
    return merged

def flatten(rootfile_in, rootfile_out, rootfile_in2=None, time_cap=None):
    '''
    If rootfile_in2==None (default),
    label all entries as accidental background.
    '''
    t_in = TChain(tree_name) # Input tree
    t_in.Add(rootfile_in)
    print("rootfile_in:", rootfile_in)
    # Run num, CHANGE DEPENDING ON FILE NAMING SCHEME
    if rootfile_in2:
        run_no = int(rootfile_in.split('.')[-3][1:])
    elif time_cap is not None:
        run_no = 0
    else:
        run_no = int(rootfile_in.split('.')[-3])
    entries = t_in.GetEntries()

    if rootfile_in2:
        t_in.AddFriend(tree_name2, rootfile_in2)

    f = TFile( rootfile_out, 'recreate') # Output .root
    t = TTree( tree_name, 'Flattened tree with timing peaks') # Output tree

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

    var_types_to_flatten = {
        # Dict of variables to flatten from input tree,
        # with their types ('i' for int, 'f' for float)
        'N10':'i', 'N10d':'i',
        'Nc':'i', 'Nback':'i', 'N300':'i',
        'trms':'f', 'trmsdiff':'f', 'fpdist':'f',
        'bpdist':'f', 'fwall':'f', 'bwall':'f',
        'bse':'f', 'mintrms_3':'f', 'mintrms_6':'f',
        'Qrms':'f', 'Qmean':'f', 'thetarms':'f',
        'phirms':'f', 'thetam':'f', 'dt':'f',

        'NhighQ':'i', 'NLowtheta':'i', 'Nlow1':'i', 'Nlow2':'i',
        'Nlow3':'i', 'Nlow4':'i', 'Nlow5':'i', 'Nlow6':'i',
        'Nlow7':'i', 'Nlow8':'i', 'Nlow9':'i',

        'goodness_neutron':'f', 'goodness_prompt':'f',
        'goodness_window':'f', 'goodness_combined':'f'
    }

    var_types_to_copy = {
        # Variables to copy without flattening (that are already flat)
        'np':'i', 'N200M':'i',
    }

    var_types_to_calc = {
        # Variables to calculate, not present in input tree
        'event_num': 'i','run_num':'i','is_signal': 'i','Nlow':'i',
    }

    # Merge dicts
    var_types = merge_dicts([var_types_to_calc,
                             var_types_to_flatten,
                             var_types_to_copy
                             ])

    # Arrays to be filled with single entry as tree is read ROOT-style
    var_arrays = {vr: array(ty,[0]) for vr, ty in var_types.items()}

    # Book new TTree branches,
    # i.e. t.Branch( 'event_num', event_num, 'event_num/I')
    for vr in var_types:
        t.Branch( vr, var_arrays[vr], vr + '/' + var_types[vr].capitalize())

    var_arrays['run_num'][0] = run_no
    for entry in range(entries):
        #if entry % 10000 == 0: print("Processed %d/%d" % (entry, entries))
        t_in.GetEntry(entry)
        lowe = t_in.LOWE

        # Record primary event number to retain peak-grouping information
        var_arrays['event_num'][0] = entry

        # Copy already flat variables
        for vr in var_types_to_copy:
            var_arrays[vr][0] = getattr(t_in, vr)

        # Flatten peak-level variables
        for p in range(var_arrays['np'][0]):
            for vr in var_types_to_flatten:
                in_val = getattr(t_in, vr)[p]
                if vr=='N10d' and isnan(in_val):
                    in_val = 0 # Replace NaN N10d with 0s
                if var_types[vr] == 'i' :
                    in_val = int(in_val)

                var_arrays[vr][0] = in_val

            # Nlow
            pvx = lowe.bsvertex[0]
            pvy = lowe.bsvertex[1]
            pvz = lowe.bsvertex[2]
            r2 = pvx**2 + pvy**2
            pvz /= 100.
            r2 /= 10000.
            nlows = [var_arrays['Nlow%d'%i][0] for i in range(1,10)]
            nlowid = get_low_id(r2, pvz)
            # print(nlowid)
            var_arrays['Nlow'][0] = nlows[nlowid]

            if rootfile_in2:
                # Signal truth variable from MC
                time = var_arrays['dt'][0]
                try:
                    cap_time = t_in.timeCap[0] # True n catpture time
                except IndexError:
                    # In absence of n capture on H,
                    # candidate peak is accidental
                    is_sig = False
                else:
                    is_sig = abs(time - cap_time - 900.) < 200
                var_arrays['is_signal'][0] = is_sig
            elif time_cap is not None:
                # Preset cap time
                time = var_arrays['dt'][0]
                is_sig = abs(time - time_cap - 900.) < 200
                var_arrays['is_signal'][0] = is_sig
            else:
                # For pure-background samples
                var_arrays['is_signal'][0] = False

            t.Fill()
    f.Write()
    f.Close()
    print("Flattened file: ", rootfile_in)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="ntagged ROOT file")
    parser.add_argument("output_file", help="flattened ROOT file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-mc", help="skdetsim file with capture time info")
    group.add_argument("-ct", help="fixed capture time")

    args = parser.parse_args()
    rin = args.input_file
    rout = args.output_file
    rin2 = args.mc
    capT = args.ct
    if capT:
        capT = float(capT)

    flatten(rin, rout, rin2, capT)
