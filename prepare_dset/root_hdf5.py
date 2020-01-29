from rootpy.root2hdf5 import root2hdf5
from glob import glob

#start, stop = 100, 200
#rootfile_ins = ["/data_CMS/cms/giampaolo/ntag-dset/root_flat/shift0/%03i.root"%i for i in range(start,stop)]
#hdf5file_outs = ["/data_CMS/cms/giampaolo/ntag-dset/hdf5_flat/shift0/%03i.hdf5"%i for i in range(start,stop)]

rootfile_ins = glob("/data_CMS/cms/giampaolo/td-ntag-dset/root_flat/*")
hdf5file_outs = ["/data_CMS/cms/giampaolo/td-ntag-dset/hdf5_flat/"
                 + '.'.join(fl.split("/")[-1].split(".")[:-1]) 
                 + '.hdf5' for fl in rootfile_ins]

for rin, hout in zip(rootfile_ins, hdf5file_outs):
    root2hdf5(rin,hout,entries=10000, show_progress=True)
    print('Converted file:', rin, '-->', hout)