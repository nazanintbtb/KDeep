# KDeep: a k-mer-based deep learning approach for  predicting DNA/RNA transcription factor binding  sites
# KDeep & KDeep+ workflow
![Screenshot (4)](https://user-images.githubusercontent.com/88847995/216258822-1f120880-749d-45b4-8fa0-473398a45ce3.png)

![Screenshot (6)](https://user-images.githubusercontent.com/88847995/216259426-c3c339fe-daf2-44d9-8845-f69ccdc6b17e.png)

# Abstract
Based on the importance of DNA/RNA binding proteins in different cellular processes, finding binding sites of them play crucial role in many applications, like designing drug/vaccine, designing protein, and cancer control. Many studies target this issue and try to improve the prediction accuracy with three strategies: complex neural-network structures, various types of inputs, and ML methods to extract input features. But due to the growing volume of sequences, these methods face serious processing challenges. So, this paper presents KDeep, based on CNN-LSTM and the primary form of DNA/RNA sequences as input. As the key feature improving the prediction accuracy, we propose a new encoding method, 2Lk, which includes two levels of k-mer encoding. 2Lk not only increases the prediction accuracy of RNA/DNA binding sites, but also, reduces the encoding memory-consumption by maximum 84%, improves the number of trainable parameters, and increases the interpretability of KDeep by about 79%, compared to the state-of-the-art methods.

# DNA
ROC & PR accuracy
![Screenshot (8)](https://user-images.githubusercontent.com/88847995/216260753-28ad0aec-eb4a-4f67-989e-f4351fee716e.png)

Dnase ROC
Method	Q1	Median	Q3	Average
DanQ	0.8890	0.9023	0.9107	0.8982
KDeep	0.8956	0.9134	0.9194	0.9065
KDeep+	0.9100	0.9241	0.9309	0.9183
	Dnase PR
Method	Q1	Median	Q3	Average
DanQ	0.3497	0.3810	0.4124	0.3789
KDeep	0.3657	0.3994	0.4326	0.3960
KDeep+	0.4006	0.4367	0.4728	0.4296



# RNA


