import time
import sys
from joblib import load
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress tensorflow info outputs
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam, Nadam, Adadelta
from keras import metrics
from load_ntag import load_dset, save_model, ntagGenerator

N10TH = 7
NFILES = 50
BATCH_SIZE = 256

# Directory in which to save models and parameters (should exist inside the "models" directory)
grid_location = 'NN/test_search/'

try:
    param_file = sys.argv[1]
except IndexError:
    raise ValueError("Usage: python3 param_search_nn.py paramfile")

model_name = param_file.split("/")[-1].split('.')[-2] # model has same name as param file


if __name__ == '__main__':

    # Initialize model according to parameter file
    with open(param_file, 'rb') as fl:
        params = load(fl)
        print('Loaded parameter file: '+param_file)

    ntag_model = Sequential()
    ntag_model.add(Dense(params['width'], activation='relu', input_dim=22))
    ntag_model.add(Dropout(params['dropout']))
    for layer in range(params['depth']-1):
        ntag_model.add(Dense(params['width'], activation='relu'))
        ntag_model.add(Dropout(params['dropout']))
    ntag_model.add(Dense(1, activation='sigmoid'))

    if params['optimizer']=='sgd':
        opt = SGD(learning_rate=params['learning_rate'], decay=1e-6, momentum=0.9, nesterov=False)
    elif params['optimizer']=='adam':
        opt = Adam(learning_rate=params['learning_rate'], beta_1=0.9, beta_2=0.999, amsgrad=False) # lr = 0.001*BATCH_SIZE/64.
    elif params['optimizer']=='nadam':
        opt = Nadam(learning_rate=params['learning_rate'], beta_1=0.9, beta_2=0.999)
    elif params['optimizer']=='adadelta':
        opt = Adadelta() # Keep Adadelta with default settings
    else:
        raise ValueError("Invalid optimizer parameter: " + params['optimizer'])
    ntag_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy',metrics.AUC()])
    print('Built NN with these hyperparameters:')
    print(params)

    #x_test, x_train, y_test, y_train = load_dset(N10TH, NFILES)
    print("Initializing dataset generators...")
    train_gen = ntagGenerator(N10th=N10TH, file_frac=NFILES/200., test_frac=0.25, batch_size=BATCH_SIZE, train=True)
    test_gen  = ntagGenerator(N10th=N10TH, file_frac=NFILES/200., test_frac=0.25, batch_size=BATCH_SIZE, train=False)
    print("Generators initialized")

    # Train
    print("Starting training...")
    start_time = time.time()
    #ntag_model.fit(x_train,y_train, epochs=5, verbose=1)
    ntag_history = ntag_model.fit_generator(train_gen, validation_data=test_gen,epochs=params['epochs'],verbose=2,use_multiprocessing=False, workers=0)
    end_time = time.time()
    training_dt = end_time-start_time
    print("Trained model with ID "+model_name+" in %0.2f seconds" % training_dt)

    # Save model and history object
    save_model(ntag_model, model_name, hist=ntag_history)
    print("Saved model and training history to disk.")
    print("All done!")
