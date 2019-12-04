import numpy as np
import h5py
from numpy.lib.recfunctions import structured_to_unstructured as unstruc
from joblib import load, dump
from keras.utils import Sequence

# Linux
# dset_location = "/media/alberto/KINGSTON/Data/hdf5_flat/shift0/" # Directory of flattened (1 peak/entry) hdf5 files
# model_location = "/home/alberto/SK19/ntag_algo/models/" # Directory of trained ntag models
# Windows
dset_location = "F:\\Data\\hdf5_flat\\shift0\\"
model_location = "D:\\Alberto\\University\\X_HS_2018_2019\\DSNB_SK\\Neutron_tagging\\ntag_local\\models\\"

varlist = '''N10 N10d Nc Nback N300 trms trmsdiff fpdist bpdist 
            fwall bwall bse mintrms_3 mintrms_6 Qrms Qmean 
            thetarms NLowtheta phirms thetam NhighQ Nlow '''.split()


def load_dset(N10th=7, num_files=1, test_frac=0.25, mode='xy', start_file=0):
    '''
    Load neutron tagging training and testing datasets into memory
    Mode must be chosen from: 'xy','yx','x','y' to return either
    features, targets, or both, in a tuple (x_test, x_train, y_test, y_train).
    Omitting features can be useful to limit memory usage.
    '''
    if mode not in ['xy','yx','x','y']: raise ValueError("mode must be chosen from: 'xy','yx','x','y'")
    files = [dset_location+'%03i.hdf5'%i for i in range(start_file, start_file+num_files)]
    x_test, x_train, y_test, y_train = [], [], [], []
    for fname in files:
        # Load data
        f = h5py.File(fname,'r')
        dset = f['sk2p2']

        # Choose variables
        myvars = tuple(varlist)

        # Preselection cut
        presel = dset["N10"]>=N10th
        # Build training and testing samples
        frac_num = int(1./test_frac)
        test, train = dset["event_num"]%frac_num==0, dset["event_num"]%frac_num!=0
        test, train = test & presel, train & presel
        if 'x' in mode:
            x = dset[myvars]
            x_itest, x_itrain = x[test], x[train]
            # Remove ndarry structure, converting to unstructured float ndarray
            x_itest, x_itrain = unstruc(x_itest), unstruc(x_itrain)
            x_test += [x_itest]
            x_train += [x_itrain]
        if 'y' in mode:
            y = dset["is_signal"]
            y_itest, y_itrain = y[test], y[train]
            y_test += [y_itest]
            y_train += [y_itrain]

    if 'x' in mode: x_test, x_train = np.concatenate(x_test), np.concatenate(x_train)
    if 'y' in mode: y_test, y_train = np.concatenate(y_test), np.concatenate(y_train)

    if 'x' in mode and 'y' in mode: return x_test, x_train, y_test, y_train
    elif mode == 'x': return x_test, x_train
    else: return y_test, y_train

def load_model(model_name): return load(model_location + "%s.joblib" % model_name)

def save_model(model, model_name, hist=None):
    ''' Save a model (and, optionally, its history)'''
    dump(model, model_location + "%s.joblib" % model_name)
    if hist: dump(hist, model_location + "%s_hist.joblib" % model_name)

def is_invalid(array):
    isn = np.isnan(array)
    isi = np.isinf(array)
    return np.any(np.logical_or(isn,isi))

class ntagGenerator(Sequence):
    '''
    Generate training or testing set in batches
    to speed up GPU performance and handle 
    larger-than-memory datasets
    for training / testing (train=True/False)
    Mode:     'xy'= each batch contains (features, targets)
              'x' = features only
              'y' = targets only
    '''
    def __init__(self, N10th=7, num_files=1, test_frac=0.25, batch_size=32, train=True, start_file=0, mode='xy'):
        '''
        Initialize generator
        '''
        if mode not in ['xy','yx','x','y']: raise ValueError("mode must be chosen from: 'xy','yx','x','y'")
        self.N10th = N10th
        self.num_files = num_files
        self.test_frac = test_frac
        self.frac_num = int(1./self.test_frac)
        self.batch_size = batch_size
        self.train = train
        self.files = [dset_location+'%03i.hdf5'%i for i in range(start_file, start_file+num_files)]
        self.mode = mode
        self.lengths = [] # Number of entries for each data file
        self.indices = [] # Indices corresponding to valid entries after filtering, for each file
        self.__filter()

    def __filter(self):
        '''
        Apply cuts and select training/testing set
        '''
        for fname in self.files:
            f = h5py.File(fname,'r') # Load data
            dset = f['sk2p2']
            dset_indices = np.array(range(len(dset))) # Get indices

            presel = dset["N10"]>=self.N10th  # Preselection cut
            frac = (dset["event_num"]%self.frac_num!=0) if self.train else (dset["event_num"]%self.frac_num==0) # Select testing or training fraction
            dset_indices = dset_indices[frac & presel] # Filtered indices, file-specific

            self.lengths += [len(dset_indices)]  # number of entries in current file
            self.indices += [dset_indices]

            f.close()
        self.l_tot = np.sum(self.lengths) # Total number of entries


    def __len__(self):
        '''Number of batches'''
        return int( np.ceil( self.l_tot / float(self.batch_size) ) )

    def __generate_batch(self, ifile, istart, istop):
        ''' Generate 1 batch of data'''
        f = h5py.File(self.files[ifile],'r') # Load data file

        batch_indices = list(self.indices[ifile][istart:istop])
        batch = f['sk2p2'][batch_indices] # Only load batch into memory
        
        x_batch, y_batch = batch[varlist], batch["is_signal"]
        x_batch = unstruc(x_batch) # Remove ndarry structure
        if is_invalid(x_batch): raise ValueError("Invalid value found in a batch from file %d"%ifile)

        if 'x' in self.mode and 'y' in self.mode: return x_batch, y_batch
        elif self.mode == 'x': return x_batch
        else: return y_batch

    def __getitem__(self, idx):
        '''Get batch of data number idx'''
        istart, istop = idx*self.batch_size, (idx+1)*self.batch_size
        if istop > self.l_tot: istop = self.l_tot # Resize the last batch

        ifile = 0
        multifile = False # Whether a batch includes entries from two files
        for i,l in enumerate(self.lengths):
            if istart >= l and not multifile: # Batch not in current file
                istart -= l
                istop -= l
            elif istop > l: # Batch starts in current file, ends in next
                multifile = True
                istop -= l
            else: # Batch starts and ends in current file
                ifile = i
                break
    
        if multifile:
            istop0 = self.lengths[ifile-1]
            batch0 = self.__generate_batch(ifile-1, istart, istop0)
            batch1 = self.__generate_batch(ifile, 0, istop)
            if 'x' in self.mode and 'y' in self.mode:
                return np.concatenate((batch0[0], batch1[0])), np.concatenate((batch0[1], batch1[1]))
            else: return np.concatenate((batch0, batch1))
        else:
            return self.__generate_batch(ifile, istart, istop)
