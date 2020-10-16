"""
Merge hdf5 files into larged files for faster access.
"""
import  sys
from os import mkdir
from glob import glob

import numpy as np
import h5py

if len(sys.argv) != 2:
    raise TypeError("Expected 1 arg, got %d" % len(sys.argv))
h5_dir = sys.argv[1]

tree_name = 'sk2p2'
files_to_merge = glob(h5_dir + "/*.h5")
try:
    mkdir(h5_dir + "/merged",)
except FileExistsError:
    pass
min_size = 2.0 # min GB to make one file


output = []
out_i = 1
def check_size(out):
    ''' Check if minimum size reached '''
    size = sum([sys.getsizeof(ar) for ar in out])
    size = size / 1024. / 1024. / 1024. # GB
    return size > min_size

num_fl = len(files_to_merge)
for i_fl, fl in enumerate(files_to_merge):
    if i_fl % (num_fl/100) == 0:
        print(f"{float(i_fl)/num_fl*100:.1f} %", end="\r")
        sys.stdout.flush()
    with h5py.File(fl, 'r') as hfl:
        output.append(hfl[tree_name][:])

    if check_size(output):
        output = np.concatenate(output)
        with h5py.File(h5_dir + "/merged/ntag%i.h5" % out_i, 'w') as merged:
            merged[tree_name] = output
        output = []
        out_i += 1
if len(output) > 0:
    output = np.concatenate(output)
    with h5py.File(h5_dir + "/merged/ntag%i.h5" % out_i, 'w') as merged:
        merged[tree_name] = output
