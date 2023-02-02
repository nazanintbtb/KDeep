# -*- coding: utf-8 -*-
from pyfaidx import Fasta
import os
import tensorflow as tf
from tqdm import tqdm
from itertools import product
import numpy as np
import pickle

def make_zeros(n_rows: int, n_columns: int):
    return [[0] * n_columns for _ in range(n_rows)]

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



def fasta_process(dir):
  x = Fasta(dir)
  f = open(dir, "r")
  lines=f.readlines()
  y=[]
  for i in lines:
    if("class" in i):
      y.append([int(i[-2])])
  y=np.array(y)

  return x, y
  
def text_process(dir):
  y=[]
  x=[]
  f = open(dir, "r")
  lines=f.readlines()
  
  lines=lines[1:len(lines)]
  for i in lines:
    y.append([int(i[-2])])
    x.append(i[0:375])
  y=np.array(y)
  return x, y

def Gen_Words(sequences,kmer_len,s):
   
    kmer_list1=[]
    kmer=[]
    for j in range(0,(len(sequences)-kmer_len)+1,s):

           kmer_list1.append(sequences[j:j+kmer_len])
    for i in range(len(kmer_list1)):
      a=FC[kmer_list1[i]]
      kmer.append(a)

    
    return kmer  
  



def serialize_example(x, y):
    # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
    
    example = {
         
        'x': tf.train.Feature(float_list=tf.train.FloatList(value=x.flatten())),
        'y': tf.train.Feature(float_list=tf.train.FloatList(value=y.flatten()))}

    # Create a Features message using tf.train.Example.
    example = tf.train.Features(feature=example)
    example = tf.train.Example(features=example)
    serialized_example = example.SerializeToString()
    return serialized_example



def traindata_to_tfrecord(dir,type_data):
  if(type_data=="text"):
    trainx,trainy=text_process(dir)

  if(type_data=="fasta"):
    trainx,trainy=fasta_process(dir)
  with tf.io.TFRecordWriter('./data/traindata.tfrecord') as writer:
    for i in tqdm(range(len(trainy)), desc="Processing Train Data", ascii=True):
      aa=str(trainx[i]).replace("U", "T")
      strand=Gen_Words(aa,3,1)
      strand=np.array(strand)
      example_proto = serialize_example(strand,trainy[i])
      writer.write(example_proto)


def testdata_to_tfrecord(dir,type_data):
  if(type_data=="text"):
    testx,testy=text_process(dir)
  if(type_data=="fasta"):
    testx,testy=fasta_process(dir)
    #testy=testy[0:9832]
  with tf.io.TFRecordWriter('./data/testdata.tfrecord') as writer:
    for i in tqdm(range(len(testy)), desc="Processing Test Data", ascii=True):
      aa=str(testx[i]).replace("U", "T")
      strand=Gen_Words(aa,3,1)
      strand=np.array(strand)
      example_proto = serialize_example(strand,testy[i])
      writer.write(example_proto)


def validdata_to_tfrecord(dir,type_data):
  if(type_data=="text"):
    validx,validy=text_process(dir)

  if(type_data=="fasta"):
    validx,validy=fasta_process(dir)  
  with tf.io.TFRecordWriter('./data/validdata.tfrecord') as writer:
      for i in tqdm(range(len(validy)), desc="Processing valid Data", ascii=True):
          aa=str(validx[i]).replace("U", "T")
          strand=Gen_Words(aa,3,1)
          strand=np.array(strand)
          example_proto = serialize_example(strand, validy[i])
          writer.write(example_proto)

if __name__ == '__main__':
    #Write the train data and test data to .tfrecord file.
    dir_train = input("Enter your direction of experience_train : ")
    dir_test = input("Enter your direction of experience_test : ")
    type_data= input("Enter type of your data: type fasta or text? ")
    
    
    path="data"
    if not os.path.exists(path):
      os.makedirs(path, exist_ok=False)

    testdata_to_tfrecord(dir_test,type_data)
    traindata_to_tfrecord(dir_train,type_data)
    validdata_to_tfrecord(dir_test,type_data)