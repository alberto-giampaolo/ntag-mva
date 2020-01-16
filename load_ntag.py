import numpy as np
import h5py
from glob import glob
from joblib import load, dump
from keras.utils import Sequence


# polui
dset_location = "/data_CMS/cms/giampaolo/ntag-dset/hdf5_flat/shift0/"
model_location = "/home/llr/t2k/giampaolo/srn/ntag-mva/models/"

# local Linux
#dset_location = "/media/alberto/KINGSTON/Data/hdf5_flat/shift0/" # Directory of flattened (1 peak/entry) hdf5 files
#model_location = "/home/alberto/SK19/ntag_algo/models/" # Directory of trained ntag models
# local Windows
#dset_location = "F:\\Data\\hdf5_flat\\shift0\\"
#model_location = "D:\\Alberto\\University\\X_HS_2018_2019\\DSNB_SK\\Neutron_tagging\\ntag_local\\models\\"

varlist = '''N10 N10d Nc Nback N300 trms trmsdiff fpdist bpdist 
            fwall bwall bse mintrms_3 mintrms_6 Qrms Qmean 
            thetarms NLowtheta phirms thetam NhighQ Nlow '''.split()


def get_fields_and_offsets(dt, offset=0):
    """
    Returns a flat list of (dtype, count, offset) tuples of all the
    scalar fields in the dtype "dt", including nested fields, in left
    to right order.
    """
    fields = []
    for name in dt.names:
        field = dt.fields[name]
        if field[0].names is None:
            count = 1
            for size in field[0].shape:
                count *= size
            fields.append((field[0], count, field[1] + offset))
        else:
            fields.extend(get_fields_and_offsets(field[0], field[1] + offset))
    return fields

def unstructure(arr, dtype=None, copy=False, casting='unsafe'):
    """
    Converts and n-D structured array into an (n+1)-D unstructured array.
    The new array will have a new last dimension equal in size to the
    number of field-elements of the input array. If not supplied, the output
    datatype is determined from the numpy type promotion rules applied to all
    the field datatypes.
    Nested fields, as well as each element of any subarray fields, all count
    as a single field-elements.
    Parameters
    ----------
    arr : ndarray
       Structured array or dtype to convert. Cannot contain object datatype.
    dtype : dtype, optional
       The dtype of the output unstructured array.
    copy : bool, optional
        See copy argument to `ndarray.astype`. If true, always return a copy.
        If false, and `dtype` requirements are satisfied, a view is returned.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        See casting argument of `ndarray.astype`. Controls what kind of data
        casting may occur.
    Returns
    -------
    unstructured : ndarray
       Unstructured array with one more dimension.
    """
    if arr.dtype.names is None:
        raise ValueError('arr must be a structured array')

    fields = get_fields_and_offsets(arr.dtype)
    n_fields = len(fields)
    dts, counts, offsets = zip(*fields)
    names = ['f{}'.format(n) for n in range(n_fields)]

    if dtype is None:
        out_dtype = np.result_type(*[dt.base for dt in dts])
    else:
        out_dtype = dtype

    # Use a series of views and casts to convert to an unstructured array:

    # first view using flattened fields (doesn't work for object arrays)
    # Note: dts may include a shape for subarrays
    flattened_fields = np.dtype({'names': names,
                                 'formats': dts,
                                 'offsets': offsets,
                                 'itemsize': arr.dtype.itemsize})
    
    arr = arr.view(flattened_fields)

    # next cast to a packed format with all fields converted to new dtype
    packed_fields = np.dtype({'names': names,
                              'formats': [(out_dtype, dt.shape) for dt in dts]})
    arr = arr.astype(packed_fields, copy=copy, casting=casting)

    # finally is it safe to view the packed fields as the unstructured type
    return arr.view((out_dtype, (sum(counts),)))

def load_dset(N10th=7, file_frac=0.005, test_frac=0.25, mode='xy'):
    '''
    Load neutron tagging training and testing datasets into memory
    file_frac controls the fraction of the file to use, in [0,1]
    test_frac controls how much of the dataset should be used for testing
    Mode must be chosen from: 'xy','yx','x','y' to return either
    features, targets, or both, in a tuple (x_test, x_train, y_test, y_train).
    Omitting features can be useful to limit memory usage.
    '''
    if mode not in ['xy','yx','x','y']: raise ValueError("mode must be chosen from: 'xy','yx','x','y'")
    #files = [dset_location+'%03i.hdf5'%i for i in range(start_file, start_file+num_files)]
    files = glob(dset_location + '*')
    x_test, x_train, y_test, y_train = [], [], [], []
    for fname in files:
        # Load data
        f = h5py.File(fname,'r')
        dset = f['sk2p2']
        tot_entries = len(dset)
        n_entries = int(np.ceil(file_frac * tot_entries))
        dset = dset[:n_entries]

        # Preselection cut
        presel = dset["N10"]>=N10th
        # Build training and testing samples
        test_frac_num = int(1./test_frac)
        test, train = dset["event_num"]%test_frac_num==0, dset["event_num"]%test_frac_num!=0
        test, train = test & presel, train & presel
        if 'x' in mode:
            x = dset[varlist]
            x_itest, x_itrain = x[test], x[train]
            # Remove ndarry structure, converting to unstructured float ndarray
            x_itest, x_itrain = unstructure(x_itest), unstructure(x_itrain)
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
def load_hist(model_name):
    try:
        return load(model_location + "%s_hist.joblib" % model_name)
    except FileNotFoundError:
        try:
            return load_model(model_name).evals_result()
        except:
            raise FileNotFoundError("No training history found")
    

def save_model(model, model_name, hist=None, subloc=""):
    ''' Save a model (and, optionally, its history)'''
    dump(model, model_location+subloc + "%s.joblib" % model_name)
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
    def __init__(self, N10th=7, file_frac=0.005, test_frac=0.25, batch_size=32, train=True, mode='xy'):
        '''
        Initialize generator
        '''
        if mode not in ['xy','yx','x','y']: raise ValueError("mode must be chosen from: 'xy','yx','x','y'")
        self.N10th = N10th
        self.file_frac = file_frac
        #self.num_files = num_files
        self.test_frac = test_frac
        self.test_frac_num = int(1./self.test_frac)
        self.batch_size = batch_size
        self.train = train
        #self.files = [dset_location+'%03i.hdf5'%i for i in range(start_file, start_file+num_files)]
        self.files = glob(dset_location + '*')
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
            tot_entries = len(dset)
            dset_indices = np.array(range(tot_entries)) # Get indices
            n_entries = np.ceil(tot_entries * self.file_frac)
            
            select = dset_indices < n_entries # Only use file_frac of total file
            presel = dset["N10"]>=self.N10th  # Preselection cut
            frac = (dset["event_num"]%self.test_frac_num!=0) if self.train else (dset["event_num"]%self.test_frac_num==0) # Select testing or training fraction
            dset_indices = dset_indices[select & frac & presel] # Filtered indices, file-specific

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
        x_batch = unstructure(x_batch) # Remove ndarry structure
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
