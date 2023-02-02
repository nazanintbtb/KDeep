# KDeep: a k-mer-based deep learning approach for  predicting DNA/RNA transcription factor binding  sites
# KDeep & KDeep+ workflow
![Screenshot (4)](https://user-images.githubusercontent.com/88847995/216258822-1f120880-749d-45b4-8fa0-473398a45ce3.png)

![Screenshot (6)](https://user-images.githubusercontent.com/88847995/216259426-c3c339fe-daf2-44d9-8845-f69ccdc6b17e.png)

# Abstract
Based on the importance of DNA/RNA binding proteins in different cellular processes, finding binding sites of them play crucial role in many applications, like designing drug/vaccine, designing protein, and cancer control. Many studies target this issue and try to improve the prediction accuracy with three strategies: complex neural-network structures, various types of inputs, and ML methods to extract input features. But due to the growing volume of sequences, these methods face serious processing challenges. So, this paper presents KDeep, based on CNN-LSTM and the primary form of DNA/RNA sequences as input. As the key feature improving the prediction accuracy, we propose a new encoding method, 2Lk, which includes two levels of k-mer encoding. 2Lk not only increases the prediction accuracy of RNA/DNA binding sites, but also, reduces the encoding memory-consumption by maximum 84%, improves the number of trainable parameters, and increases the interpretability of KDeep by about 79%, compared to the state-of-the-art methods.

# DNA accuracy
ROC & PR accuracy
![Screenshot (8)](https://user-images.githubusercontent.com/88847995/216260753-28ad0aec-eb4a-4f67-989e-f4351fee716e.png)

Dnase ROC & PR accuracy
![DNASE](https://user-images.githubusercontent.com/88847995/216276090-2e577602-c2b4-440c-958f-1132290603ae.png)

TF ROC & PR accuracy
![TF](https://user-images.githubusercontent.com/88847995/216276141-aa5344c2-4800-40dd-a05f-81b6229bde52.png)

Histone ROC & PR accuracy
![HISTONE](https://user-images.githubusercontent.com/88847995/216276193-c45e1b77-a8eb-4865-b1cd-92353661c740.png)

# RNA accuracy
![RNAPIC](https://user-images.githubusercontent.com/88847995/216283950-3f772f10-880d-4363-a391-9e4040c3cb1e.png)



