from os import environ, devnull
environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress avalanche of tensorflow info outputs
import sys
stderr = sys.stderr
sys.stderr = open(devnull, 'w') # Suppress Keras info outputs
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
sys.stderr = stderr
import numpy as np
import matplotlib.pyplot as plt
import time
from load_ntag import load_dset, save_model, ntagGenerator
from roc import plot_ROC_sigle, plot_ROC_sigle_gen


N10TH = 7
NFILES = 1
STARTFILE = 1
NEPOCHS = 10
BATCH_SIZE = 32
model_name = "NN\\batch_size\\NN22_n%i_%iepoch_%ibatch_%i"%(N10TH, NEPOCHS,BATCH_SIZE, NFILES)

if __name__ == '__main__':
    #x_test, x_train, y_test, y_train = load_dset(N10TH, NFILES)
    train_gen = ntagGenerator(N10th=N10TH, num_files=NFILES, test_frac=0.25, batch_size=BATCH_SIZE, train=True, start_file=STARTFILE)
    test_gen  = ntagGenerator(N10th=N10TH, num_files=NFILES, test_frac=0.25, batch_size=BATCH_SIZE, train=False, start_file=STARTFILE)

    # Define neutron tagging model
    ntag_model = Sequential()
    ntag_model.add(Dense(22, activation='relu', input_dim=22))
    ntag_model.add(Dropout(0.5))
    ntag_model.add(Dense(22, activation='relu'))
    #ntag_model.add(Dropout(0.5))
    ntag_model.add(Dense(1, activation='sigmoid'))
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    ntag_model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=['accuracy'])

    # Train
    start_time = time.time()
    #ntag_model.fit(x_train,y_train, epochs=5, verbose=1)
    ntag_history = ntag_model.fit_generator(train_gen, validation_data=test_gen,epochs=NEPOCHS,verbose=1,use_multiprocessing=True, workers=3)
    end_time = time.time()
    training_dt = end_time-start_time
    print("Trained model in %f seconds" % training_dt)

    # Save model and history object
    save_model(ntag_model, model_name, hist=ntag_history)
    print("Saved model to disk.")

    # Plot ROC curve
    print("Plotting performance (ROC)")
    y_test, _ = load_dset(N10TH, NFILES, mode='y', start_file=STARTFILE)
    x_test_gen= ntagGenerator(N10th=N10TH, num_files=NFILES, test_frac=0.25, batch_size=BATCH_SIZE, train=False, mode='x', start_file=STARTFILE)
    plot_ROC_sigle_gen(x_test_gen, y_test, ntag_model, model_name, N10TH)
    print("All done!")

    # Plot loss vs epoch
    print("Plotting loss evolution")
    plt.plot(ntag_history.history['loss'], label='Training')
    plt.plot(ntag_history.history['val_loss'], label='Testing', color='tab:red')
    plt.ylabel('Loss')
    plt.xlabel('Training epoch')
    plt.yscale('log')
    plt.legend()
    plt.show()