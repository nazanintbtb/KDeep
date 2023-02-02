# KDeep: a k-mer-based deep learning approach for  predicting DNA/RNA transcription factor binding  sites
# KDeep & KDeep+ WORKFLOW
![Screenshot (4)](https://user-images.githubusercontent.com/88847995/216258822-1f120880-749d-45b4-8fa0-473398a45ce3.png)

![Screenshot (6)](https://user-images.githubusercontent.com/88847995/216259426-c3c339fe-daf2-44d9-8845-f69ccdc6b17e.png)

# ABSTRACT
Based on the importance of DNA/RNA binding proteins in different cellular processes, finding binding sites of them play crucial role in many applications, like designing drug/vaccine, designing protein, and cancer control. Many studies target this issue and try to improve the prediction accuracy with three strategies: complex neural-network structures, various types of inputs, and ML methods to extract input features. But due to the growing volume of sequences, these methods face serious processing challenges. So, this paper presents KDeep, based on CNN-LSTM and the primary form of DNA/RNA sequences as input. As the key feature improving the prediction accuracy, we propose a new encoding method, 2Lk, which includes two levels of k-mer encoding. 2Lk not only increases the prediction accuracy of RNA/DNA binding sites, but also, reduces the encoding memory-consumption by maximum 84%, improves the number of trainable parameters, and increases the interpretability of KDeep by about 79%, compared to the state-of-the-art methods.

# DNA ACCURACY
ROC & PR accuracy
![Screenshot (8)](https://user-images.githubusercontent.com/88847995/216260753-28ad0aec-eb4a-4f67-989e-f4351fee716e.png)

Dnase ROC & PR accuracy
![DNASE](https://user-images.githubusercontent.com/88847995/216276090-2e577602-c2b4-440c-958f-1132290603ae.png)

TF ROC & PR accuracy
![TF](https://user-images.githubusercontent.com/88847995/216276141-aa5344c2-4800-40dd-a05f-81b6229bde52.png)

Histone ROC & PR accuracy
![HISTONE](https://user-images.githubusercontent.com/88847995/216276193-c45e1b77-a8eb-4865-b1cd-92353661c740.png)

# RNA ACCURACY
![RNAPIC](https://user-images.githubusercontent.com/88847995/216283950-3f772f10-880d-4363-a391-9e4040c3cb1e.png)

# USAGE
## Need package
python3.7,  tensorflow==2.8, cuda and cuDNN if you have GPU

## DNA
###  Train model by DNA Dataset and run instruction
**To train the model, download the training, validation and testing sets from DeepSEA dataset (You can download the datasets from [here](http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz))
After you have extracted the contents of the tar.gz file, move the 3 .mat files into the KDeep/data/ or KDeep+/data/ folder.
then run below command:**

 1.python preprocess_FCGR.py.
 
 2.python KDeep.py | KDeep+.py.
 
 3.python test.py.
 
 
### To test the KDeep or KDeep+ model without train model

 #### 1.Test trained model on your system:
 
Skip download data from deepsea link. You need just download test data from [here](https://drive.google.com/file/d/1y_KarPolOGFFzcdeoKOY9w_tg0NG3jYg/view?usp=sharing) and [here](https://drive.google.com/file/d/1fBN1fVCMKRmCLCO4vBiYB3OZYdjUV-ae/view?usp=sharing) then extract files and move to DNA\KDeep\data or DNA\KDeep+\data folder. and download The KDeep model that trained by myself from [here](https://drive.google.com/file/d/150I1vVEpqrPR_m6yZAyEwEGMAGfTzYZa/view?usp=sharing) or KDeep+ model
 from [here](https://drive.google.com/file/d/1xUuL74NiVLXNDtsLI0HjB5lNTrZsgy7x/view?usp=sharing) and move to DNA\KDeep\model or DNA\KDeep+\model folder.


#### 2.Test trained model on colab:
 
If you want just test KDeep without training go to **[colab](https://colab.research.google.com/drive/1bdPTxxkB4Gd_R0GBSVfI_R57bUVTjomv?usp=sharing)**.

If you want just test KDeep+ without training go to **[colab](https://colab.research.google.com/drive/1f4AUlTIwnB_1ezZkbf8L7y0g8C6m_o3S?usp=sharing)**.
 
 ##  **RNA** 
 ### RNA dataset
 Download Datasets from [RNA_31](https://drive.google.com/drive/folders/1zW4cGL2SsfCxscnsCKmRywHbSsKSb_gA?usp=sharing)then move to RNA\RNA_31 folder. AND [RNA_24](https://drive.google.com/drive/folders/1--hAqnWlECTDRA-ILV0IKrFR1Wvw_L2E?usp=sharing) then move to RNA\RNA_24 folder. 
 
 ### Train and test in colab 
 go to [colab](https://colab.research.google.com/drive/1mLV1jp-VIQSu99h51O3mKf5_gExoowrc?usp=sharing) and run codes step by step. 
 
###  Pre_process section 

**For RNA-31:**

python PreProcess.py

+ Enter your direction of experience_train like (RNA_31/train/1/sequences.fa)

+ Enter your direction of experience_test like (RNA_31/test/1/sequences.fa)

+ Enter (fasta) to determine type of your data

**For RNA-24:**

python PreProcess.py 

+ Enter your direction of experience_train like (RNA_24/1/ALKBH5_Baltz2012_train)

+ Enter your direction of experience_test like (RNA_24/1/ALKBH5_Baltz2012_test)

+ Enter (text) to determine type of your data

### Training section 

**For RNA-31:**
pythin Training.py

+ Enter (420) to determine appropriate seed for learning

+ Enter train number =(30000)

+ Enter valid number = (10000)

+ Enter batch_size = (300)

+ Enter 101 to determine sequences lenght of RNA-31

**For RNA-24:**
pythin Training.py

+ Enter (0) to determine appropriate seed for learning

+ Enter train number =(Check output of preprocess section) for experience one 'ALKBH5_Baltz2012' training number is 2410

+ Enter valid number = (Check output of preprocess section). for experience one 'ALKBH5_Baltz2012' valid number is 266

+ Enter batch_size like (300)

+ Enter 375 to determine sequences lenght of RNA-24

Point=If the model fails to train, you should reduce the batch number

### **Test section** 

**For RNA-31:**

python Test.py

+ Enter your direction of experience_test like (RNA_31/test/1/sequences.fa)

+ Enter (fasta) to determine type of your data
 
+ Enter (101) to determine sequences lenght of RNA-31

**For RNA-24:**

python Test.py

+ Enter your direction of experience_test like (RNA_24/1/ALKBH5_Baltz2012_test)

+ Enter (text) to determine type of your data

+ Enter (375) to determine sequences lenght of RNA-24


### **Extracted motif Section** 

**For RNA-31:**

python Training.py

+ Enter your direction of experience_test like (RNA_31/test/1/sequences.fa)

+ Enter (fasta) to determine type of your data

+ Enter batch-size that use in trainin section

**For RNA-24:**

pyhton Training.py

+ Enter your direction of experience_test like (RNA_24/1/ALKBH5_Baltz2012_test)

+ Enter (text) to determine type of your data

+ Enter batch-size that use in trainin section

# CONTACT INFO
Somayyeh Koohi

Department of Computer Engineering

Sharif University of Technology

e-mail: koohi@sharfi.edu

WWW: http://sharif.ir/~koohi/
