from rootpy.root2hdf5 import root2hdf5
from glob import glob
from sys import argv

if __name__=='__main__':
    if len(argv) > 1: 
        rootfile_ins = [argv[1]]
    else:
        rootfile_ins = glob("/data_CMS/cms/giampaolo/mc/darknoise4_new/root_flat_nlow/*")

    hdf5file_outs = ["/data_CMS/cms/giampaolo/mc/darknoise4_new/hdf5_flat_nlow/"
                    + '.'.join(fl.split("/")[-1].split(".")[:-1]) 
                    + '.hdf5' for fl in rootfile_ins]
    for rin, hout in zip(rootfile_ins, hdf5file_outs):
        root2hdf5(rin,hout,entries=10000, show_progress=True)
        print('Converted file:', rin, '-->', hout)