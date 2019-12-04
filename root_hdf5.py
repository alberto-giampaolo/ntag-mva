from rootpy.root2hdf5 import root2hdf5

rootfile_ins = ["/media/alberto/KINGSTON/Data/flat/shift0/%03i.root"%i for i in range(20,50)]
hdf5file_outs = ["/media/alberto/KINGSTON/Data/hdf5_flat/shift0/%03i.hdf5"%i for i in range(20,50)]

for rin, hout in zip(rootfile_ins, hdf5file_outs):
    root2hdf5(rin,hout,entries=10000, show_progress=True)