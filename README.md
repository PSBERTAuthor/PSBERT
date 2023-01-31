
# Ethereum Account Profiling and De-anonymization via Pseudo-Siamese BERT (IJCAI23 under review)

Code and dataset for the submission "Ethereum Account Profiling and De-anonymization via Pseudo-Siamese BERT".

The majority of code has been uploaded. We will complete this repository soon.. (update in Jan. 30)

<div align=center><img width="600" height="270" src="https://github.com/PSBERTAuthor/PSBERT/blob/master/materials/framework.png"/></div>


PSBERT mainly consists of two parts: pre-training and fine-tuning, corresponding to /Model/PSBERT and /Model/PSBERT_finetune.

## Getting Start
### Requirements:
* Python >= 3.6.1
* NumPy >= 1.12.1
* TensorFlow >= 1.4.0


###  1. Preprocess dataset 

#### Step 1. Download dataset from Google drive:
* [Normal Account Transaction](https://drive.google.com/file/d/1-htLUymg1UxDrXcI8tslU9wbn0E1vl9_/view?usp=sharing)

* [ENS Account Transaction](https://drive.google.com/file/d/1Yveis90jCx-nIA6pUL_4SUezMsVJr8dp/view?usp=sharing)

* [Tornado Account Transaction]

#### Step 2. Unzip dataset under the directory of "PSBERT/Data"
``` 
cd Data;
unzip ...;
``` 
The total volume of unzipped dataset is quite huge (about 15GB).

#### Step 3. Generate Transaction sequence
```sh
cd Model/bert4eth;
python gen_seq.py --phisher=True \
                  --deanon=True \ 
                  --mev=True \ 
                  --bizdate=xxx
``` 

### 2. Pre-training

The configuration file is "Model/PSBERT/bert_config.json"
```
{
  "attention_probs_dropout_prob": 0.2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "hidden_size": 64,
  "intermediate_size": 64,
  "initializer_range": 0.02,
  "max_position_embeddings": 50,
  "num_attention_heads": 2,
  "num_hidden_layers": 16,
  "type_vocab_size": 2,
  "vocab_size": 3000000
}

```

#### Step1: Generate pre-training dataset
```sh
python gen_pretrain_data.py --bizdate=xxx \
                            --max_seq_length=50 \
                            --masked_lm_prob=0.8 \
                            --max_predictions_per_seq=40 \
                            --sliding_step=30 \
                            --dupe_factor=10 \
                            --do_eval=False
```
#### Step2: Pre-train PSBERT via Masked Address Prediction
```sh
python run_pretrain.py --bizdate=xxx \
                       --max_seq_length=50 \
                       --max_predictions_per_seq=40 \
                       --masked_lm_prob=0.8 \
                       --epoch=5 \
                       --batch_size=256 \
                       --learning_rate=1e-4 \
                       --num_train_steps=1000000 \
                       --num_warmup_steps=100 \
                       --save_checkpoints_steps=8000 \
                       --neg_strategy=zip \
                       --neg_sample_num=5000 \
                       --do_eval=False \
                       --checkpointDir=xxx \
                       --init_seed=1234 
```

| Parameter                  | Description                                                                        |
|----------------------------|------------------------------------------------------------------------------------|
| `bizdate`                  | The signature for this experiment run.                                             |
| `max_seq_length`           | The maximum length of BERT4ETH.                                                    |
| `max_predictions_per_seq`  | The maximum number of masked addresses in one sequence.                            |
| `masked_lm_prob`           | The probability of masking an address.                                             |
| `epochs`                   | Number of training epochs, default = `5`.                                          |
| `batch_size`               | Batch size, default = `256`.                                                       |
| `learning_rate`            | Learning rate for the optimizer (Adam), default = `1e-4`.                          |
| `num_train_steps`          | The maximum number of training steps, default = `1000000`,                         |
| `num_warmup_steps`         | The step number for warm-up training, default = `100`.                             |
| `save_checkpoints_steps`   | The parameter controlling the step of saving checkpoints, default = `8000`.        |
| `neg_strategy`             | Strategy for negative sampling, default `zip`, options (`uniform`, `zip`, `freq`). |
| `neg_sample_num`           | The negative sampling number for one batch, default = `5000`.                      |
| `do_eval`                  | Whether to do evaluation during training, default = `False`.                       |
| `checkpointDir`            | Specify the directory to save the checkpoints.                                     |
| `init_seed`                | The initial seed, default = `1234`.                                                |
#### Step3: Evaluate the result after pre-training
``` 
python run_test.py
``` 

### 3. Fine-tuning
``` 
cd Model/PSBERT_finetune
``` 
#### Step1: Generate pseudo positive pair
``` 
python gen_pseudo_pair.py
``` 
#### Step2: Fine-tune PSBERT in a siamese-network
``` 
python run_finetune.py
``` 
#### Step3: Evaluate the result of PSBERT
``` 
python run_test.py
``` 