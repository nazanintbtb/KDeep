# -*- coding: utf-8 -*-

from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
import numpy as np

def parse_function(example_proto):
    dics = {
        'x': tf.io.FixedLenFeature([lenght_seq,64],tf.float32),
        'y': tf.io.FixedLenFeature([1], tf.float32),
    }
    parsed_example = tf.io.parse_single_example(example_proto, dics)
    x = tf.reshape(parsed_example['x'], [lenght_seq,64])
    y = tf.reshape(parsed_example['y'], [1])
  
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    return (x, y)

def get_train_data(batch_size,lenght_seq):
    filenames = ['./data/traindata.tfrecord']
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=100000)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    return dataset # 4400000/64 = 68750


def get_valid_data(batch_size,lenght_seq):
    filenames = ['./data/validdata.tfrecord']
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=10000)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset # 455024/64 = 7109.75 = 7110

#----------------------------------------------------------------------------------------------
def train_model(seed,train_number,valid_number,batch_size,lenght_seq):
  np.random.seed(seed)
  tf.random.set_seed(seed)
  earlystop=keras.callbacks.EarlyStopping(monitor="val_loss",patience=15,verbose=1, mode='min',restore_best_weights=True )
  checkpoint = keras.callbacks.ModelCheckpoint("KDeep.h5",monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  callbacks_list=[checkpoint,earlystop]
  model = keras.Sequential()

  conv_layer =keras.layers.Conv1D(filters=64,
                      kernel_size=12,
                      strides=1,
                      padding='same',
                      activation='relu',
                      input_shape=(lenght_seq,64))
  

  model.add(conv_layer)
  
  model.add(keras.layers.MaxPooling1D(pool_size=2,
                         strides=1))
  model.add(keras.layers.Dropout(0.2))
  model.add(keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True)))

  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(925,
                  activation='relu'))
  
  model.add(keras.layers.Dense(1,
                  activation='sigmoid'))

  loss1 = keras.losses.BinaryCrossentropy()
  opt1 = keras.optimizers.Adam()
  model.compile(loss=loss1,
                optimizer=opt1 ,
                metrics=['accuracy'])

#-----------------------------------------------------------------------

  history = model.fit(get_train_data(batch_size,lenght_seq), steps_per_epoch=int(np.ceil(train_number/batch_size)), epochs=60,
                        validation_data=get_valid_data(batch_size,lenght_seq), validation_steps=int(np.ceil(valid_number/batch_size)),callbacks=callbacks_list)

  np.save('my_history.npy',history.history)

if __name__ == '__main__':
    #Write the train data and test data to .tfrecord file.
    seed = input("Enter appropriate seed for learning : ")
    train_number = input("Enter train number : ")
    valid_number= input("Enter valid number : ")
    batch_size= input("Enter batch_size : ")
    lenght_seq= input("Enter lenght of sequence : 375 or 101? ")
    lenght_seq=int(lenght_seq)-2
    train_model(int(seed),int(train_number),int(valid_number),int(batch_size),int(lenght_seq))