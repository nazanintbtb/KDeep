# -*- coding: utf-8 -*-
import numpy as np
import h5py
import scipy.io
from sklearn import metrics

#from keras.layers import Embedding
#from keras.models import Sequential
#from keras.models import Model
#from keras.layers import Dense, Dropout, Activation, Flatten, Layer, merge, Input, Concatenate, Reshape, concatenate,Lambda,multiply,Permute,Reshape,RepeatVector
#from keras.layers.convolutional import Conv1D, MaxPooling1D
#from keras.layers.pooling import GlobalMaxPooling1D
#from keras.layers.recurrent import LSTM
#from keras.layers.wrappers import Bidirectional, TimeDistributed
#from keras.models import load_model
#from keras.callbacks import ModelCheckpoint, EarlyStopping
#from keras import optimizers
#from keras import backend as K
#from keras import regularizers
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import roc_auc_score
#from keras.models import load_model
from pyfaidx import Fasta

def parse_function(example_proto):
    dics = {
        'x': tf.io.FixedLenFeature([998,64],tf.int64),
        'y': tf.io.FixedLenFeature([919], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example_proto, dics)
    x = tf.reshape(parsed_example['x'], [998,64])
    y = tf.reshape(parsed_example['y'], [919])
  
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return (x, y)

def fasta_process(dir):
  x = Fasta(dir)
  f = open(dir, "r")
  lines=f.readlines()
  y=[]
  for i in lines:
    if("class" in i):
      y.append([int(i[-2])])
  y=np.array(y)

  return y
def get_train_data(batch_size):
    filenames = ['./data/traindata-00.tfrecord','./data/traindata-01.tfrecord','./data/traindata-02.tfrecord','./data/traindata-03.tfrecord','./data/traindata-04.tfrecord','./data/traindata-05.tfrecord'
                ,'./data/traindata-06.tfrecord','./data/traindata-07.tfrecord','./data/traindata-08.tfrecord','./data/traindata-09.tfrecord','./data/traindata-10.tfrecord','./data/traindata-11.tfrecord'
                ,'./data/traindata-12.tfrecord','./data/traindata-13.tfrecord','./data/traindata-14.tfrecord','./data/traindata-15.tfrecord']
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=100000)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset # 4400000/64 = 68750


def get_valid_data(batch_size):
    filenames = ['./data/validdata.tfrecord']
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=10000)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset # 455024/64 = 7109.75 = 7110

def get_test_data(batch_size):
    filenames = ['./data/testdata-00.tfrecord','./data/testdata-01.tfrecord','./data/testdata-02.tfrecord','./data/testdata-03.tfrecord']
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=10000)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset # 455024/64 = 7109.75 = 7110

#----------------------------------------------------------------------------------------------
def train_model():
  #np.random.seed(420)
  #tf.random.set_seed(420)
  #earlystop=keras.callbacks.EarlyStopping(monitor="val_loss",patience=6,verbose=1, mode='min',restore_best_weights=True )
  #checkpoint = keras.callbacks.ModelCheckpoint("fck4%s.h5"%num,monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  #callbacks_list=[checkpoint,earlystop]
#-----------------------------------------------
  sequence_input = keras.layers.Input(shape=(998,64))


# Convolutional Layer
  output = keras.layers.Conv1D(320,kernel_size=26,padding="valid",activation="relu")(sequence_input)
  output = keras.layers.MaxPooling1D(pool_size=13, strides=13)(output)
  output = keras.layers.Dropout(0.2)(output)

  #Attention Layer
#  attention = keras.layers.Dense(1)(output)
#  attention = keras.layers.Permute((2, 1))(attention)
#  attention = keras.layers.Activation('softmax')(attention)
#  attention = keras.layers.Permute((2, 1))(attention)
#  attention = keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=2), name='attention',output_shape=(75,))(attention)
#  attention = keras.layers.RepeatVector(320)(attention)
#  attention = keras.layers.Permute((2,1))(attention)
#  output = keras.layers.multiply([output, attention])
  
  #BiLSTM Layer
  output = keras.layers.Bidirectional(keras.layers.LSTM(320,return_sequences=True))(output)
  output = keras.layers.Dropout(0.5)(output)

  flat_output = keras.layers.Flatten()(output)

  #FC Layer
  FC_output = keras.layers.Dense(925)(flat_output)
  FC_output = keras.layers.Activation('relu')(FC_output)

  #Output Layer
  output = keras.layers.Dense(919)(FC_output)
  output = keras.layers.Activation('sigmoid')(output)

  model = keras.models.Model(inputs=sequence_input, outputs=output)

  print('compiling model')
  model.compile(loss='binary_crossentropy', optimizer='adam')

  print('model summary')
  model.summary()

  checkpointer = keras.callbacks.ModelCheckpoint(filepath="./model/KDeep.h5",monitor='val_loss', verbose=1, save_best_only=True)
  earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, verbose=1)




  model.fit(get_train_data(100), steps_per_epoch=int(np.ceil(4400000/100)), epochs=300,
                        validation_data=get_valid_data(100), validation_steps=int(np.ceil(8000/100)), callbacks=[checkpointer,earlystopper])

  model.save('./model/KDeep.h5')
#-----------------------------------------------------------------------

 

  

if __name__ == '__main__':
   
    
    train_model()