#!/usr/bin/env python

from utils import *

import pandas as pd
import matplotlib.pylab as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector, AveragePooling1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *

#import seaborn as sns
#sns.despine()


#data_original = pd.read_csv('./data/AAPL1216.csv')[::-1]
#
#openp = data_original.ix[:, 'Open'].tolist()
#highp = data_original.ix[:, 'High'].tolist()
#lowp = data_original.ix[:, 'Low'].tolist()
#closep = data_original.ix[:, 'Adj Close'].tolist()
#volumep = data_original.ix[:, 'Volume'].tolist()

data_original = pd.read_csv('./data/ETHBTC.csv')[::-1]

openp = data_original.ix[:, 'open'].tolist()
highp = data_original.ix[:, 'high'].tolist()
lowp = data_original.ix[:, 'low'].tolist()
closep = data_original.ix[:, 'close'].tolist()
volumep = data_original.ix[:, 'volume'].tolist()
quoteVolumep = data_original.ix[:, 'quoteVolume'].tolist()
weightedAveragep = data_original.ix[:, 'weightedAverage'].tolist()

# data_chng = data_original.ix[:, 'Adj Close'].pct_change().dropna().tolist()

#data_original = pd.read_csv('./data/BtcEthLtc.csv')[::-1]
#
#openp = data_original.ix[:, 'Open'].tolist()
#highp = data_original.ix[:, 'High'].tolist()
#lowp = data_original.ix[:, 'Low'].tolist()
#closep = data_original.ix[:, 'Close'].tolist()
#volumep = data_original.ix[:, 'Volume'].tolist()
#quoteVolumep = data_original.ix[:, 'Market Cap'].tolist()
#
#openpeth = data_original.ix[:, 'ethOpen'].tolist()
#highpeth = data_original.ix[:, 'ethHigh'].tolist()
#lowpeth = data_original.ix[:, 'ethLow'].tolist()
#closepeth = data_original.ix[:, 'ethClose'].tolist()
#volumepeth = data_original.ix[:, 'ethVolume'].tolist()
#quoteVolumepeth = data_original.ix[:, 'ethMarket Cap'].tolist()
#
#openpltc = data_original.ix[:, 'ltcOpen'].tolist()
#highpltc = data_original.ix[:, 'ltcHigh'].tolist()
#lowpltc = data_original.ix[:, 'ltcLow'].tolist()
#closepltc = data_original.ix[:, 'ltcClose'].tolist()
#volumepltc = data_original.ix[:, 'ltcVolume'].tolist()
#quoteVolumepltc = data_original.ix[:, 'ltcMarket Cap'].tolist()



WINDOW = 8
EMB_SIZE = 7
STEP = 1
FORECAST = 1

X, Y = [], []
for i in range(0, len(data_original), STEP): 
    try:
        o = openp[i:i+WINDOW]
        h = highp[i:i+WINDOW]
        l = lowp[i:i+WINDOW]
        c = closep[i:i+WINDOW]
        v = volumep[i:i+WINDOW]
        qv = quoteVolumep[i:i+WINDOW]
        wa = weightedAveragep[i:i+WINDOW]
        
        o = (np.array(o) - np.mean(o)) / np.std(o)
        h = (np.array(h) - np.mean(h)) / np.std(h)
        l = (np.array(l) - np.mean(l)) / np.std(l)
        c = (np.array(c) - np.mean(c)) / np.std(c)
        v = (np.array(v) - np.mean(v)) / np.std(v)
        qv = (np.array(qv) - np.mean(qv)) / np.std(qv) 
        wa = (np.array(wa) - np.mean(wa)) / np.std(wa)
        
        #o = openp[i:i+WINDOW]
        #h = highp[i:i+WINDOW]
        #l = lowp[i:i+WINDOW]
        #c = closep[i:i+WINDOW]
        #v = volumep[i:i+WINDOW]
        #qv = quoteVolumep[i:i+WINDOW]
        #
        #
        #o = (np.array(o) - np.mean(o)) / np.std(o)
        #h = (np.array(h) - np.mean(h)) / np.std(h)
        #l = (np.array(l) - np.mean(l)) / np.std(l)
        #c = (np.array(c) - np.mean(c)) / np.std(c)
        #v = (np.array(v) - np.mean(v)) / np.std(v)
        #qv = (np.array(qv) - np.mean(qv)) / np.std(qv) 
        #
        #oeth = openpeth[i:i+WINDOW]
        #heth = highpeth[i:i+WINDOW]
        #leth = lowpeth[i:i+WINDOW]
        #ceth = closepeth[i:i+WINDOW]
        #veth = volumepeth[i:i+WINDOW]
        #qveth = quoteVolumepeth[i:i+WINDOW]
        #
        #
        #oeth = (np.array(oeth) - np.mean(oeth)) / np.std(oeth)
        #heth = (np.array(heth) - np.mean(heth)) / np.std(heth)
        #leth = (np.array(leth) - np.mean(leth)) / np.std(leth)
        #ceth = (np.array(ceth) - np.mean(ceth)) / np.std(ceth)
        #veth = (np.array(veth) - np.mean(veth)) / np.std(veth)
        #qveth = (np.array(qveth) - np.mean(qveth)) / np.std(qveth) 
        
        x_i = closep[i:i+WINDOW]
        y_i = closep[i+WINDOW+FORECAST]  

        #x_i = closepeth[i:i+WINDOW]
        #y_i = closepeth[i+WINDOW+FORECAST]
        
        last_close = x_i[-1]
        next_close = y_i

        if last_close < next_close:
            y_i = [1, 0]
        else:
            y_i = [0, 1] 
        x_i = np.column_stack((o, h, l, c, v, qv, wa))
        #x_i = np.column_stack((o, h, l, c, v, qv,oeth, heth, leth, ceth, veth, qveth))

    except Exception as e:
        break

    X.append(x_i)
    Y.append(y_i)

X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = create_Xt_Yt(X, Y)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))


model = Sequential()
model.add(Convolution1D(input_shape = (WINDOW, EMB_SIZE),
                        nb_filter=8,
                        filter_length=2,padding = "same"))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Convolution1D(nb_filter=16,
                        filter_length=4,padding = "same"))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(AveragePooling1D())
model.add(Dropout(0.5))
model.add(Convolution1D(nb_filter=32,
                        filter_length=4,padding = "same"))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Flatten())
#model.add(Bidirectional(LSTM(10,return_sequences = True,activation = 'sigmoid',recurrent_dropout = 0.25,recurrent_activation = 'sigmoid')))
#model.add(BatchNormalization())
#model.add(LeakyReLU())
#model.add(AveragePooling1D())
#model.add(Dropout(0.25))

#model.add(Bidirectional(LSTM(16,return_sequences = False,activation = None,recurrent_dropout = 0.25, recurrent_activation = 'sigmoid')))
#model.add(BatchNormalization())
#model.add(LeakyReLU())
#model.add(Dropout(0.5))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(LeakyReLU())


model.add(Dense(2))
model.add(Activation('softmax'))

opt = Nadam(lr=0.004)

reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath="lol.hdf5", verbose=1, save_best_only=True)


model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, Y_train, 
          nb_epoch = 10000, 
          batch_size = 32, 
          verbose=1, 
          validation_data=(X_test, Y_test),
          callbacks=[reduce_lr, checkpointer],
          shuffle=True)

model.load_weights("lol.hdf5")
pred = model.predict(np.array(X_test))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
C = confusion_matrix([np.argmax(y) for y in Y_test], [np.argmax(y) for y in pred])

print (C / C.astype(np.float).sum(axis=1))

# Classification
# [[ 0.75510204  0.24489796]
#  [ 0.46938776  0.53061224]]


# for i in range(len(pred)):
#     print Y_test[i], pred[i]


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()