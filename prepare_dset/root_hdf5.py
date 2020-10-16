"""
Convert a flat-structure root file to hdf5 format
Usage: python root_hdf5.py input_file.root output_file.h5
"""
from __future__ import print_function
from sys import argv

from rootpy.root2hdf5 import root2hdf5


if __name__=='__main__':
    if len(argv) != 3:
        raise ValueError("Usage: python root_hdf5.py root_in h5_out")

    root_in = argv[1]
    h5_out = argv[2]
    root2hdf5(root_in,h5_out,entries=10000, show_progress=True)
    print('Converted file:', root_in, '-->', h5_out)
