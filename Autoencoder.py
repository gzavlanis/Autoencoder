from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import numpy as np
import tensorflow as tf
from matplotlib import pyplot

# create and plot the random signals X for input training data and y for output training data
X= np.array(np.random.standard_normal(250000))
y= X
X= X.reshape(250000,1)
y= y.reshape(250000,1)
pyplot.plot(X)
pyplot.title('Random signal', fontsize=18)
pyplot.xlabel('Time (Nanoseconds)', fontsize=18)
pyplot.ylabel('Amplitude (a.u.)', fontsize=18)
pyplot.axis([5000, 6000, -5, 5])
pyplot.grid(True)
pyplot.show()

pyplot.plot(y)
pyplot.title('Random signal', fontsize=18)
pyplot.xlabel('Time (Nanoseconds)', fontsize=18)
pyplot.ylabel('Amplitude (a.u.)', fontsize=18)
pyplot.axis([5000, 6000, -5, 5])
pyplot.grid(True)
pyplot.show()

# convert data to one-hot vectors
X= tf.one_hot(X, depth=64, dtype=tf.int32)
y= tf.one_hot(y, depth=64, dtype=tf.int32)
print(X)
print(y)

#define the encoder
def get_encoder(n_inputs= 64):
     model= Sequential()
     model.add(Dense(256, activation= 'relu', input_shape=(None, n_inputs)))
     model.add(Dense(256, activation= 'relu'))
     model.add(Dense(48, activation= 'sigmoid'))
     print(model.summary())
     plot_model(model, to_file='encoder.png', show_shapes=True, show_layer_names=True)
     return model

#define the decoder
def get_decoder():
    model= Sequential()
    model.add(Dense(256, activation= 'relu', input_shape=(None, 48)))
    model.add(Dense(256, activation= 'relu'))
    model.add(Dense(64, activation= 'linear'))
    model.add(Dense(64, activation= 'softmax'))
    print(model.summary())
    plot_model(model, to_file='decoder.png', show_shapes=True, show_layer_names=True)
    return model

#define the autoencoder
def get_autoencoder(Transmitter, Receiver):
    model= Sequential()
    model.add(Transmitter)
    model.add(Receiver)
    model.compile(loss= 'mse' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    plot_model(model, to_file='autoencoder.png', show_shapes=True, show_layer_names=True)
    return model

# connect Encoder and Decoder to create the system
Transmitter= get_encoder()
Receiver= get_decoder()
Autoencoder= get_autoencoder(Transmitter, Receiver)

#train the autoencoder and plot the loss
hist= Autoencoder.fit(X, y, epochs=3, batch_size=250)
pyplot.plot(hist.history['loss'])
pyplot.title('Train loss of Autoencoder', fontsize=18)
pyplot.ylabel('Loss', fontsize=18)
pyplot.xlabel('Epochs', fontsize=18)
pyplot.locator_params(axis="x", nbins=3)
pyplot.grid(True)
pyplot.show()

# evaluate model
loss, acc = Autoencoder.evaluate(X, y, verbose=0)
print('Loss: %f, Accuracy: %f' % (loss, acc*100)) #the accuracy (x100%)

#make predictions for X after training
yhat= Autoencoder.predict(X, verbose=0) #data after autoencoder
print(yhat)

# convert tensor objects to array-type
yhat= np.array(yhat)
y= np.array(y)
print(yhat)
print(y.shape)
print(yhat.shape)

#convert predicted data to one-hot
yhat[:,:,:]= np.where(yhat[:,:,:]==np.amax(yhat[:,:,:], axis=0), 1, 0)
print(yhat)
print(yhat.shape)

#find errors and probability of errors
errors=0
if (yhat[:,:]!= y[:,:]).any():
     errors= errors+1
print(errors)
Pb= (errors/250000)*100
print(Pb)
