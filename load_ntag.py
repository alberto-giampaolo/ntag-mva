"""
Dataset and model loading functions
"""
from __future__ import print_function
import sys
from os import environ
from glob import glob

import numpy as np
import h5py
from joblib import load, dump

H5_DIR = environ['ntag_hdf5_dir_train']
H5_DIR_T2K = environ['ntag_hdf5_dir_train_t2k']
MODELS_DIR = environ['models_dir']

tree_name = "sk2p2"
varlist = '''N10 N10d Nc Nback N300 trms trmsdiff fpdist bpdist
            fwall bwall bse mintrms_3 mintrms_6 Qrms Qmean
            thetarms NLowtheta phirms thetam NhighQ Nlow'''.split()


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
                              'formats': [(out_dtype,
                                           dt.shape) for dt in dts]})
    arr = arr.astype(packed_fields, copy=copy, casting=casting)

    # finally is it safe to view the packed fields as the unstructured type
    return arr.view((out_dtype, (sum(counts),)))


def load_dset(N10th=7, file_frac_s=0.005, file_frac_bg=0.005, test_frac=0.25,
              file_start_s=0.0, file_start_bg=0.0, val_id=0, mode='xy',
              shuffle=False, dtype='float32', nums=False):
    '''
    Load neutron tagging training and testing datasets into memory.
    file_frac_s : fraction of the file topr use, in [0,1].
    file_frac_bg : fraction of background files to use.
    test_frac : how much of the dataset should be used for testing/validation.
        test : validate : train = test_frac : test_frac : (1 - 2*test_frac)
    file_start_s : file fraction at which to start loading signal entries.
    file_start_bg : file fraction at which to start loading background entries.
    val_id : cross-validation index (defaul=0 for no cross-validation)
    mode : to be chosen from: 'x', 'y', 'xy' to return either
        features, targets, or both, in a tuple.
        Omitting features can be useful to limit memory usage.

    shuffle : shuffle entries after loading.
    dtype : Numpy dtype
    nums : return run number and event number along  with features
    Returns: (x_test, x_validation, x_train, y_test, y_validation, y_train)
    '''
    if mode not in ['xy', 'x', 'y']:
        raise ValueError("mode must be chosen from: 'xy','x','y'")
    # files_s = glob(H5_DIR + '/*62447*')
    files_s = glob(H5_DIR + '/*ntag*')
    files_bg = glob(H5_DIR_T2K + '/*ntag*')
    x_test, x_val, x_train, y_test, y_val, y_train = [], [], [], [], [], []
    test_frac_num = int(1./test_frac)
    val_id = val_id % (test_frac_num-1)
    if nums:
        load_vars = varlist + ['run_num', 'event_num']
    else:
        load_vars = varlist
    numvars = len(load_vars)

    def load_subset(files, file_frac, file_start, mc=False):
        """
        Load from given file list and file fractions.
        mc : take only signal entries
        """
        if file_frac == 0.0:
            return
        ntot = len(files)
        n = 0
        sys.stdout.flush()
        for fname in files:
            print(f"{float(n)/ntot*100:.1f} %", end="\r")
            n += 1
            with h5py.File(fname, 'r') as f:
                try:
                    dset = f[tree_name]
                except KeyError:  # Corrupted/crashed files
                    continue
                tot_entries = len(dset)
                if tot_entries == 0:  # Corrupted/crashed files
                    continue

                n_entries = int(np.ceil(file_frac * tot_entries))
                n_start = int(np.floor(file_start * tot_entries))
                dset = dset[n_start: n_start + n_entries]

                # Preselection cut
                presel = dset["N10"] >= N10th
                # print(presel)
                n200mcut = dset["N200M"] < 50
                presel = presel & n200mcut

                # Avoid double-counting accidentals
                if mc:
                    presel = presel & dset["is_signal"] == 1
                # Build training, testing, and validation samples
                test = ((dset["event_num"] % test_frac_num) == 0)
                vali = ((dset["event_num"] % test_frac_num) == (val_id+1))
                train = (~ (test | vali))
                test, vali = test & presel, vali & presel
                train = train & presel

                if 'x' in mode:
                    x = dset[load_vars]
                    x_itest, x_ival, x_itrain = x[test], x[vali], x[train]
                    # Remove ndarray structure,
                    # converting to unstructured float ndarray
                    x_itest = unstructure(x_itest)
                    x_ival = unstructure(x_ival)
                    x_itrain = unstructure(x_itrain)
                    x_test.append(x_itest.astype(dtype))
                    x_val.append(x_ival.astype(dtype))
                    x_train.append(x_itrain.astype(dtype))
                if 'y' in mode:
                    y = dset["is_signal"]
                    y_itest, y_ival, y_itrain = y[test], y[vali], y[train]
                    y_test.append(y_itest)
                    y_val.append(y_ival)
                    y_train.append(y_itrain)
    if file_frac_s == 0.0 and file_frac_bg == 0.0:
        x_test.append(np.empty((0, numvars)))
        x_val.append(np.empty((0, numvars)))
        x_train.append(np.empty((0, numvars)))
        y_test.append(np.empty((0,)))
        y_val.append(np.empty((0,)))
        y_train.append(np.empty((0,)))
    if file_frac_s > 0.0:
        print("Loading signal events...")
        load_subset(files_s, file_frac_s, file_start_s, mc=True)
    if file_frac_bg > 0.0:
        print("Loading background events...")
        load_subset(files_bg, file_frac_bg, file_start_bg)

    # Concat and shuffle
    rng_state = np.random.get_state()
    if 'x' in mode:
        print("Preparing features...")
        x_test = np.concatenate(x_test)
        x_val = np.concatenate(x_val)
        x_train = np.concatenate(x_train)
        if shuffle:
            np.random.set_state(rng_state)
            np.random.shuffle(x_test)
            np.random.set_state(rng_state)
            np.random.shuffle(x_val)
            np.random.set_state(rng_state)
            np.random.shuffle(x_train)
    if 'y' in mode:
        print("Preparing targets...")
        y_test = np.concatenate(y_test)
        y_val = np.concatenate(y_val)
        y_train = np.concatenate(y_train)
        if shuffle:
            np.random.set_state(rng_state)
            np.random.shuffle(y_test)
            np.random.set_state(rng_state)
            np.random.shuffle(y_val)
            np.random.set_state(rng_state)
            np.random.shuffle(y_train)

    print("Shapes of test, validation, train datasets:",
          x_test.shape, x_val.shape, x_train.shape)

    if 'x' in mode and 'y' in mode:
        return x_test, x_val, x_train, y_test, y_val, y_train
    elif mode == 'x':
        return x_test, x_val, x_train
    else:
        return y_test, y_val, y_train


def load_model(model_name):
    ''' Load BDT model for application. '''
    return load("%s/%s/%s.joblib" % (MODELS_DIR, model_name, model_name))


def load_hist(model_name):
    ''' Load BDT evaluation metrics as function of training iterations. '''
    return load_model(model_name).evals_result()  # xgboost


def save_model(model, model_name):
    ''' Save BDT to disk. '''
    dump(model, "%s/%s/%s.joblib" % (MODELS_DIR, model_name, model_name))
