# -*- coding: utf-8 -*-
import h5py
import numpy as np
import tensorflow as tf
import scipy.io as sio
from tqdm import tqdm
import math
def make_zeros(row,col):
      return[[0]*col for _ in range(row)]

def CGRblock (dna, resolution, chr):
      
      loc_A = chr.find('A')
      loc_C = chr.find('C')
      loc_G = chr.find('G')
      loc_T = chr.find('T')
  
      x = math.ceil(resolution/2)
      xpre = 0
      y = math.ceil(resolution/2)
      ypre = 0
  
      p = 0.5
      screen = make_zeros(resolution,resolution)
      #screen = zeros(resolution,resolution);
      for i in range(len(dna)):
        if(dna[i]=="A"):
          xpre, ypre = FCGR_next_step(loc_A, x, y, p, resolution)
          screen[ypre][xpre]= screen[ypre][xpre] + 1
        elif(dna[i]=="C"):
          xpre, ypre = FCGR_next_step(loc_C, x, y, p, resolution)
          screen[ypre][xpre]= screen[ypre][xpre] + 1
        elif(dna[i]=="G"):
          xpre, ypre = FCGR_next_step(loc_G, x, y, p, resolution)
          screen[ypre][xpre]= screen[ypre][xpre] + 1
        elif(dna[i]=="T"):
          xpre, ypre = FCGR_next_step(loc_T, x, y, p, resolution)
          screen[ypre][xpre]= screen[ypre][xpre] + 1
        x = xpre
        y = ypre
      return screen
  

def FCGR_next_step(loc, x, y, p, resolution):
      
      if (loc == 0):
        xpre = math.floor((x)*p)
        ypre = math.floor((y)*p)
      elif(loc == 1):
        xpre = math.floor((x + resolution)*p)
        ypre = math.floor((y)*p)
      elif(loc == 2):
        xpre = math.floor((x + resolution)*p)
        ypre = math.floor((y + resolution)*p)
      elif(loc == 3):
        xpre = math.floor((x)*p)
        ypre = math.floor((y + resolution)*p)
      return (xpre,ypre)


def Gen_Words(sequences,kmer_len,s):
      kmer_list1=[]
      kmer=[]
      for j in range(0,(len(sequences)-kmer_len)+1,s):

             kmer_list1.append(sequences[j:j+kmer_len])
      for i in range(len(kmer_list1)):
        a=FC[kmer_list1[i]]
        kmer.append(a)

    
      return kmer  
  
def ret_sequence(strand):
      strandor=""
      for i in range(len(strand)):
        result = np.where(strand[i] == 1)
        if(result[0]==0):
          strandor+="A"
        elif(result[0]==1):
          strandor+="G"
        elif(result[0]==2):
          strandor+="C"
        elif(result[0]==3):
          strandor+="T"
        else:
          strandor+="N"
  
      return strandor

permutation1=[]
for roll in product(["A", "C", "G", "T","N"], repeat = 3):
    #print(roll)
    sub=""
    for s in roll:
      sub=sub + s
    permutation1.append(sub)


FC =	{}
FCreverse = {}
for s in permutation1:
  a=CGRblock(s,8,"ACGT")
  a=np.array(a)
  a= np.reshape(a, (8*8))
  FC[s] = a
  FCreverse[str(a)]=s

FC_file = open('FCdicFile', 'wb')
pickle.dump(FC, FC_file)
FC_file.close()

ReverseFC_file = open('ReverseFCdicFile', 'wb')
pickle.dump(FCreverse, ReverseFC_file)
ReverseFC_file.close()


def serialize_example(x, y):
      # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    
      example = {
        'x': tf.train.Feature(int64_list=tf.train.Int64List(value=x.flatten())),
        'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y.flatten()))}

      # Create a Features message using tf.train.Example.
      example = tf.train.Features(feature=example)
      example = tf.train.Example(features=example)
      serialized_example = example.SerializeToString()
      return serialized_example



def traindata_to_tfrecord():
      print('Loading data')
      filename = './DanQ-Data/train.mat'
      with h5py.File(filename, 'r') as file:
        X_train = file['trainxdata'] # shape = (1000, 4, 4400000)
        y_train = file['traindata'] # shape = (919, 4400000)
        X_train = np.transpose(X_train, (2, 0, 1)) # shape = (4400000, 1000, 4)
        y_train = np.transpose(y_train, (1, 0)) # shape = (4400000, 919)
  
  
   
 
      for file_num in range(16):
          with tf.io.TFRecordWriter('./DanQ-Data/traindata-%.2d.tfrecord' % file_num) as writer:
              for i in range(file_num*275000, (file_num+1)*275000):
                  strand=Gen_Words(ret_sequence(X_train[i]),3,1)
                  strand=np.array(strand)
                  example_proto = serialize_example(strand,y_train[i])
                  writer.write(example_proto)

def testdata_to_tfrecord():
      filename = './DanQ-Data/test.mat'
      data = sio.loadmat(filename)
      X_test = data['testxdata'] # shape = (455024, 4, 1000)
      y_test = data['testdata'] # shape = (455024, 919)
      X_test = np.transpose(X_test, (0, 2, 1)) # shape = (455024, 1000, 4)
      y_test = np.transpose(y_test, (0, 1))
 
      for file_num in range(4):
          with tf.io.TFRecordWriter('./DanQ-Data/testdata-%.2d.tfrecord' % file_num) as writer:
              for i in range(file_num*113756, (file_num+1)*113756):
                  strand=Gen_Words(ret_sequence(X_test[i]),3,1)
                  strand=np.array(strand)
                  example_proto = serialize_example(strand,y_test[i])
                  writer.write(example_proto)
 

def validdata_to_tfrecord():
            
      data = sio.loadmat('./DanQ-Data/valid.mat')
      X_valid = data['validxdata']  # shape = (8000, 4, 1000)
      y_valid = data['validdata']  # shape = (8000, 919)
      X_valid = np.transpose(X_valid, (0, 2, 1)).astype(dtype=np.float32)  # shape = (8000, 1000, 4)
      y_valid = np.transpose(y_valid, (0, 1)).astype(dtype=np.int32)  # shape = (8000, 919)
      with tf.io.TFRecordWriter('./DanQ-Data/validdata.tfrecord') as writer:
        for i in range(len(y_valid)):
              strand=Gen_Words(ret_sequence(X_valid[i]),3,1)
              strand=np.array(strand)
              example_proto = serialize_example(strand, y_valid[i])
              writer.write(example_proto)

if __name__ == '__main__':
      
      # Write the train data and test data to .tfrecord file.
      validdata_to_tfrecord()
      print("valid finish")
      testdata_to_tfrecord()
      print("test finish")
      traindata_to_tfrecord()
      print("train finish")

