
# Ethereum Account Profiling and De-anonymization via Pseudo-Siamese BERT (IJCAI23 under review)

Code and dataset for the submission "Ethereum Account Profiling and De-anonymization via Pseudo-Siamese BERT".


<div align=center><img width="360" height="250" src="https://github.com/PSBERTAuthor/PSBERT/blob/master/materials/framework.pdf"/></div>


PSBERT mainly consists of two parts: pre-training and fine-tuning, corresponding to /Model/PSBERT and /Model/PSBERT_finetune.

# Getting Start

##  1. Dataset download
Download dataset from Google drive:
``` 
cd Data
tar -xzvf 
tar -xzvf 
``` 

## 2. Pre-training
``` 
cd Model/PSBERT
``` 
#### Step1: Generate transaction sequence
``` 
python gen_seq.py
``` 
#### Step2: Generate pre-training dataset
``` 
python gen_pretrain_data.py
``` 
#### Step3: Pre-train PSBERT via Masked Address Prediction
``` 
python run_pretrain.py
``` 

## 3. Fine-tuning
``` 
cd Model/PSBERT_finetune
``` 
#### Step1: Generate pseudo positive pair
``` 
python gen_pseudo_pair.py
``` 
### Step2: Fine-tune PSBERT in a siamese-network
``` 
python run_finetune.py
``` 
### Step3: Evaluate the result of PSBERT
``` 
python run_test.py
``` 