import tensorflow as tf
from tensorflow import keras
import scipy.io as sio
import numpy as np
import h5py
import scipy.io
import os
import csv
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm
from keras.models import Model
#-----------------------------------------
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from pyfaidx import Fasta
import pickle

with open('FCdicFile', 'rb') as handle:
	data = handle.read()
dicFC = pickle.loads(data)

with open('ReverseFCdicFile', 'rb') as handle:
	data = handle.read()
FCreverse = pickle.loads(data)

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

import math

def Gen_Words(sequences,kmer_len,s):
   
    kmer_list1=[]
    kmer=[]
    for j in range(0,(len(sequences)-kmer_len)+1,s):

           kmer_list1.append(sequences[j:j+kmer_len])
    for i in range(len(kmer_list1)):
      a=dicFC[kmer_list1[i]]
      kmer.append(a)

    
    return kmer  

def one_hott(dade):
 # print(dade)
  onehot_dnas=[]
  for i in range(len(dade)):
   
    if(dade[i]=="A"):
      onehot_dnas.append([1,0,0,0])
    elif(dade[i]=="G"):
      onehot_dnas.append([0,0,1,0])
    elif(dade[i]=="C"):
      onehot_dnas.append([0,1,0,0])
    elif(dade[i]=="T"):
      onehot_dnas.append([0,0,0,1])
    elif(dade[i]=="N"):
      onehot_dnas.append([0,0,0,0])

  onehot_dnas=np.array(onehot_dnas)
  return (onehot_dnas)



def reach_seq(words):# retriev sequence
  seq=""
  for i in range(len(words)):
    w=words[i]
    if(i!=(len(words)-1)):
      seq+=w[0]
    else:
      seq+=w
  return seq


if __name__ == '__main__':
    dir = input("Enter your direction of test_experience to extract motifs : ")
    type_data= input("Enter type of your experience_data: type fasta or text? ")
    batch= input("Enter your batch-size in trainin section? ")
    batch=int(batch)
    if(type_data=="text"):
      testx,testy=text_process(dir)
    if(type_data=="fasta"):
      testx,testy=fasta_process(dir)

    path="motif"
    if not os.path.exists(path):
      os.makedirs(path, exist_ok=False)
    testX=[]
    
    for i in tqdm(range(len(testy)), desc="Processing Test Data", ascii=True):
      aa=str(testx[i]).replace("U", "T")
      strand=Gen_Words(aa,3,1)
      strand=np.array(strand)
      testX.append(strand) 
    
    testX=np.array(testX)

    model = load_model("KDeep.h5")
    conv_output=model.get_layer(model.layers[0].name).get_output_at(0)
    f = keras.backend.function([model.input], [keras.backend.argmax(conv_output, axis=1), keras.backend.max(conv_output, axis=1)])
    motifs = np.zeros((64, 14,4))
    nsites = np.zeros(64)
    y_test_NRSF = testy[:, [0]].sum(axis=1) > 0
    X_test_NRSF = testX[y_test_NRSF]
  #----------------------------------------------------------------
    for i in range(0, len(X_test_NRSF),batch):
        x = X_test_NRSF[i:i+batch]
        z = f([x])
        max_inds = z[0] # N x M matrix, where M is the number of motifs
    
        max_acts = z[1]
         
        for m in range(64):
            for n in range(len(x)):

                if max_acts[n, m] > 0.5:
              
                    motif_words=[]
                
               
                    a1=x[n]# padding
                    c=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
                    for pad in range(5):
                      a1=np.append(c,a1,axis=0)

                    for pad in range(6):
                      a1=np.append(a1,c,axis=0)
                    motif=a1[max_inds[n,m]:max_inds[n,m] + 12,:]# benevis agar not mokhalefe 24 tulesh bood ezafe kon ta betune jam kone khkhk 0000
      
                    for j in range(len(motif)):
                      oneword=motif[j]
                      oneword=[oneword]
                      listToStr = ' '.join(map(str, oneword))
                      listToStr=listToStr.replace("\n  ","")
                      oneword=FCreverse[listToStr]
                      motif_words.append(oneword)
        
                    motif_org_sequence=reach_seq(motif_words)
                
                    #for i in range(14-len(motif_org_sequence)):
                     # motif_org_sequence=motif_org_sequence+"N"
                    if(len(motif_org_sequence)==14):
                      mot_one=one_hott(motif_org_sequence)
                      mot_one=np.array(mot_one)
                      motifs[m] += mot_one[:,:]#imp
                      nsites[m] += 1

    print('Making motifs')

    motifs = motifs[:, :, [0, 1, 2, 3]]

    motifs_file = open('motif/motifsfcgr.txt', 'w')
    motifs_file.write('MEME version 4.9.0\n\n'
                      'ALPHABET= ACGU\n\n'
                      'strands: + -\n\n'
                      'Background letter frequencies (from uniform background):\n'
                      'A 0.25000 C 0.25000 G 0.25000 U 0.25000\n\n')

    for m in range(64):
        if nsites[m] == 0:
            continue
        motifs_file.write('MOTIF M%i O%i\n' % (m, m))
        motifs_file.write("letter-probability matrix: alength= 4 w= %i nsites= %i E= 1337.0e-6\n" % (14, nsites[m]))
        for j in range(14):
            motifs_file.write("%f %f %f %f\n" % tuple(1.0 * motifs[m, j, 0:4] / np.sum(motifs[m, j, 0:4])))
        motifs_file.write('\n')

    motifs_file.close()
#------------------------------------------------------------------------------