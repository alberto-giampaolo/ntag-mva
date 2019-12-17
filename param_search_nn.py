
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import use as mpl_use
from load_ntag import load_dset, save_model, ntagGenerator
from roc import plot_ROC_sigle, plot_ROC_sigle_gen


N10TH = 7
NFILES = 10
STARTFILE = 0
NEPOCHS = 2
BATCH_SIZE = 256
NLAYERS = 1
NNODES = 50
model_name = "polui_tests/NN22_n%i_%iepoch_%ibatch_%ix%i_%i"%(N10TH, NEPOCHS,BATCH_SIZE, NLAYERS,NNODES,NFILES)

if __name__ == '__main__':
    #x_test, x_train, y_test, y_train = load_dset(N10TH, NFILES)
    train_gen = ntagGenerator(N10th=N10TH, num_files=NFILES, test_frac=0.25, batch_size=BATCH_SIZE, train=True, start_file=STARTFILE)
    test_gen  = ntagGenerator(N10th=N10TH, num_files=NFILES, test_frac=0.25, batch_size=BATCH_SIZE, train=False, start_file=STARTFILE)

    # Define neutron tagging model
    ntag_model = Sequential()
    ntag_model.add(Dense(NNODES, activation='relu', input_dim=22))
    ntag_model.add(Dropout(0.5))
    for layer in range(NLAYERS-1):
        ntag_model.add(Dense(NNODES, activation='relu'))
        ntag_model.add(Dropout(0.5))
    ntag_model.add(Dense(1, activation='sigmoid'))
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    adam = Adam(learning_rate=0.001*BATCH_SIZE/64., beta_1=0.9, beta_2=0.999, amsgrad=False)
    ntag_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train
    print("Starting training...")
    start_time = time.time()
    #ntag_model.fit(x_train,y_train, epochs=5, verbose=1)
    ntag_history = ntag_model.fit_generator(train_gen, validation_data=test_gen,epochs=NEPOCHS,verbose=1,use_multiprocessing=False, workers=0)
    end_time = time.time()
    training_dt = end_time-start_time
    print("Trained model in %0.2f seconds" % training_dt)

    # Save model and history object
    save_model(ntag_model, model_name, hist=ntag_history)
    print("Saved model to disk.")

    # Plot ROC curve
    print("Plotting performance (ROC)")
    y_test, _ = load_dset(N10TH, NFILES, mode='y', start_file=STARTFILE)
    x_test_gen= ntagGenerator(N10th=N10TH, num_files=NFILES, test_frac=0.25, batch_size=BATCH_SIZE, train=False, mode='x', start_file=STARTFILE)
    plot_ROC_sigle_gen(x_test_gen, y_test, ntag_model, model_name, N10TH)

    # Plot loss vs epoch
    print("Plotting loss history")
    plt.plot(ntag_history.history['loss'], label='Training')
    plt.plot(ntag_history.history['val_loss'], label='Testing', color='tab:red')
    plt.ylabel('Loss')
    plt.xlabel('Training epoch')
    plt.yscale('log')
    plt.legend()
    plt.show()

    print("All done!")