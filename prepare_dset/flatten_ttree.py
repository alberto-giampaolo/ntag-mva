import threading as thr
from sys import argv
from glob import glob
from queue import Queue
from ROOT import TFile, TTree, TChain, gSystem
from array import array
from math import isnan


multithread=False
max_threads = 3

# Load SKOFL libraries
libs = glob("/home/llr/t2k/giampaolo/skroot/*.so")
for ll in libs:
    gSystem.Load(ll)

dir_in  = "/data_CMS/cms/giampaolo/td-ntag-dset/root/"
dir_out = "/data_CMS/cms/giampaolo/td-ntag-dset/root_flat/"

rootfile_ins = glob(dir_in + '*6338*')[102:]
rootfile_outs = [dir_out + fl.split('/')[-1] for fl in rootfile_ins]

#start, stop = 103, 105 # Process files numbers [start, stop)
tree_name = "data"

def flatten(rootfile_in, rootfile_out):
    t_in = TChain(tree_name) # Input tree
    t_in.Add(rootfile_in)
    entries = t_in.GetEntries()
    run_no = int(rootfile_in.split('.')[-2])

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
        'Nc':'i', 'Nback':'i', 'N300':'i','trms':'f', 'trmsdiff':'f', 'fpdist':'f',
        'bpdist':'f', 'fwall':'f','bwall':'f',
        'bse':'f', 'mintrms_3':'f', 'mintrms_6':'f','Qrms':'f', 'Qmean':'f',
        'thetarms':'f','phirms':'f','thetam':'f','dt':'f',

        'NhighQ':'i','NLowtheta':'i','Nlow1':'i','Nlow2':'i',
        'Nlow3':'i','Nlow4':'i','Nlow5':'i','Nlow6':'i','Nlow7':'i','Nlow8':'i',
        'Nlow9':'i'
    }

    var_types_to_copy = { 
        # Variables to copy without flattening (that are already flat)
        'np':'i',
    }

    var_types_to_calc = { 
        # Variables to calculate, not present in input tree
        'event_num': 'i','run_num':'i','is_signal': 'i','Nlow':'i',
    }

    # Merge dicts
    var_types = {**var_types_to_calc, **var_types_to_flatten, **var_types_to_copy}

    # Arrays to be filled with single entry as tree is read ROOT-style
    var_arrays = {vr: array(ty,[0]) for vr, ty in var_types.items()}

    # Book new TTree branches, i.e. t.Branch( 'event_num', event_num, 'event_num/I')
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
        for vr in var_types_to_copy: var_arrays[vr][0] = getattr(t_in, vr)

        # Flatten peak-level variables
        for p in range(var_arrays['np'][0]):
            for vr in var_types_to_flatten: 
                in_val = getattr(t_in, vr)[p]

                if vr=='N10d' and isnan(in_val): in_val = 0 # Replace NaN N10d with 0s
                if var_types[vr] == 'i' : in_val = int(in_val)
                var_arrays[vr][0] = in_val


            # Calculate Nlow
            pvx, pvy, pvz = lowe.bsvertex[0], lowe.bsvertex[1], lowe.bsvertex[2]
            r2 = pvx**2 + pvy**2 
            nlows = [var_arrays['Nlow%d'%i][0] for i in range(1,10)]
            nlowid = get_low_id(r2, pvz)
            var_arrays['Nlow'][0] = nlows[nlowid]

            # Signal truth variable
            var_arrays['is_signal'][0] = abs(var_arrays['dt'][0] - 200.9e3) < 200

            t.Fill()

    f.Write()
    f.Close()
    print("Flattened file: ", rootfile_in)


if __name__=='__main__':

    #try:
    #  start, stop = int(argv[1]), int(argv[2])
    #except IndexError:
    #    pass
    #print("Flattening files from number %i to %i" % (start, stop-1))
    

    #rootfile_ins = [dir_in + "%03i.root"%i for i in range(start,stop)]
    #rootfile_outs = [dir_out + "%03i.root"%i for i in range(start,stop)]

    if len(argv) == 2: 
        rootfile_ins = [argv[1]]
    rootfile_outs = [dir_out + fl.split('/')[-1] for fl in rootfile_ins]

    for rin,rout in zip(rootfile_ins,rootfile_outs):
        flatten(rin, rout)