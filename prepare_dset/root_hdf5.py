from rootpy.root2hdf5 import root2hdf5

start, stop = 100, 200

rootfile_ins = ["/data_CMS/cms/giampaolo/ntag-dset/root_flat/shift0/%03i.root"%i for i in range(start,stop)]
hdf5file_outs = ["/data_CMS/cms/giampaolo/ntag-dset/hdf5_flat/shift0/%03i.hdf5"%i for i in range(start,stop)]

for rin, hout in zip(rootfile_ins, hdf5file_outs):
    root2hdf5(rin,hout,entries=10000, show_progress=True)