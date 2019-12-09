from rootpy.root2hdf5 import root2hdf5

start, stop = 50, 100

rootfile_ins = ["/media/alberto/KINGSTON/Data/flat/shift0/%03i.root"%i for i in range(start,stop)]
hdf5file_outs = ["/media/alberto/KINGSTON/Data/hdf5_flat/shift0/%03i.hdf5"%i for i in range(start,stop)]

for rin, hout in zip(rootfile_ins, hdf5file_outs):
    root2hdf5(rin,hout,entries=10000, show_progress=True)