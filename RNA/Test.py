import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.models import Model
#-----------------------------------------
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import csv
#from pyfaidx import Fasta

def fasta_process(dir):

  f = open(dir, "r")
  lines=f.readlines()
  y=[]
  for i in lines:
    if("class" in i):
      y.append([int(i[-2])])
  y=np.array(y)

  return y
  


def text_process(dir):
  y=[]
  f = open(dir, "r")
  lines=f.readlines()
  
  lines=lines[1:len(lines)]
  for i in lines:
    y.append([int(i[-2])])
    
  y=np.array(y)
  return y

def parse_function(example_proto):
    dics = {
        'x': tf.io.FixedLenFeature([lenght_seq,64],tf.float32),
        'y': tf.io.FixedLenFeature([1], tf.float32),
    }
    parsed_example = tf.io.parse_single_example(example_proto, dics)
    x = tf.reshape(parsed_example['x'], [lenght_seq,64])
    y = tf.reshape(parsed_example['y'], [1])
  
    #x = tf.cast(x, tf.float32)
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    return (x, y)

def get_test_data(batch_size):
    filenames = ['./data/testdata.tfrecord']
    dataset = tf.data.TFRecordDataset(filenames, buffer_size=10000)
    dataset = dataset.map(map_func=parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset # 455024/64 = 7109.75 = 7110

def create_dirs(dirs):
    """
    Create dirs. (recurrent)
    :param dirs: a list directory path.
    :return: None
    """
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=False)

def write2txt(content, file_path):
    """
    Write array to .txt file.
    :param content: array.
    :param file_path: destination file path.
    :return: None.
    """
    try:
        file_name = file_path.split('/')[-1]
        dir_path = file_path.replace(file_name, '')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(file_path, 'w+') as f:
            for item in content:
                f.write(' '.join([str(i) for i in item]) + '\n')

        print("write over!")
    except IOError:
        print("fail to open file!")

def write2csv(content, file_path):
    """
    Write array to .csv file.
    :param content: array.
    :param file_path: destination file path.
    :return: None.
    """
    try:
        temp = file_path.split('/')[-1]
        temp = file_path.replace(temp, '')
        if not os.path.exists(temp):
            os.makedirs(temp)

        with open(file_path, 'w+', newline='') as f:
            csv_writer = csv.writer(f, dialect='excel')
            for item in content:
                csv_writer.writerow(item)

        print("write over!")
    except IOError:
        print("fail to open file!")

def calculate_auroc(predictions, labels):
    if np.max(labels) ==1 and np.min(labels)==0:
        fpr_list, tpr_list, _ = metrics.roc_curve(y_true=labels, y_score=predictions, drop_intermediate=True)
        auroc = metrics.roc_auc_score(labels, predictions)
    else:
        fpr_list, tpr_list = [], []
        auroc = np.nan

    return fpr_list, tpr_list, auroc

def calculate_aupr(predictions, labels):
    if np.max(labels) == 1 and np.min(labels) == 0:
        precision_list, recall_list, _ = metrics.precision_recall_curve(y_true=labels, probas_pred=predictions)
        aupr = metrics.auc(recall_list, precision_list)
    else:
        precision_list, recall_list = [], []
        aupr = np.nan
    return precision_list, recall_list, aupr

def plot_loss_curve(epoch, train_loss, val_loss, file_path):

    plt.figure()
    plt.plot(epoch, train_loss, lw=1, label = 'Train Loss')
    plt.plot(epoch, val_loss, lw=1, label = 'Valid Loss')
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.savefig(file_path)


def plot_roc_curve(fpr_list, tpr_list, file_path):
    plt.figure()
    #for i in range(125, 815):
    for i in range(1):
        plt.plot(fpr_list[i], tpr_list[i], lw=0.2, linestyle='-')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"Transcription factors (ROC)")
    plt.savefig(os.path.join(file_path, 'ROC_Curve_TF.jpg'))

    plt.figure()
 


def plot_pr_curve(precision_list, recall_list, file_path):
 
    plt.figure()
    for i in range(1):
        plt.plot(recall_list[i], precision_list[i], lw=0.2, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(u"Transcription factors (PR)")
    plt.savefig(os.path.join(file_path, 'PR_Curve_TFBinding.jpg'))

if __name__ == '__main__':
   
    dir = input("Enter your direction of experience to test: ")
    type_data= input("Enter type of your data: type fasta or text? ")
    lenght_seq= input("Enter lenght of sequence : 375 or 101? ")
    lenght_seq=int(lenght_seq)-2
    if(type_data=="text"):
      testy=text_process(dir)
    if(type_data=="fasta"):
      testy=fasta_process(dir)

    path="result"
    if not os.path.exists(path):
      os.makedirs(path, exist_ok=False)

    model = load_model("KDeep.h5")
    result = model.predict(get_test_data(100))
    label = testy
    result_shape = np.shape(result)

    fpr_list, tpr_list, auroc_list = [], [], []
    precision_list, recall_list, aupr_list = [], [], []
    for i in tqdm(range(result_shape[1]), ascii=True):
        fpr_temp, tpr_temp, auroc_temp  = calculate_auroc(result[:, i], label[:, i])
        precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], label[:, i])

        fpr_list.append(fpr_temp)
        tpr_list.append(tpr_temp)
        precision_list.append(precision_temp)
        recall_list.append(recall_temp)
        auroc_list.append(auroc_temp)
        aupr_list.append(aupr_temp)

    plot_roc_curve(fpr_list, tpr_list, './result/')
    plot_pr_curve(precision_list, recall_list, './result/')

    header = np.array([['auroc', 'aupr']])
    content = np.stack((auroc_list, aupr_list), axis=1)
    content = np.concatenate((header, content), axis=0)
    write2csv(content, './result/result.csv')
    write2txt(content, './result/result.txt')
    avg_auroc = np.nanmean(auroc_list)
    avg_aupr = np.nanmean(aupr_list)
    print('AVG-AUROC:{:.3f}, AVG-AUPR:{:.3f}.\n'.format(avg_auroc, avg_aupr))



    