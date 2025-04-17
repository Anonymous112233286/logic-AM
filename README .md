# Learning First-Order Logic Rules for Argumentation Mining
Code for our ACL-2025 paper [Learning First-Order Logic Rules for Argumentation Mining] 

## Environment
- ipdb==0.13.9
- nltk==3.8.1
- numpy==1.21.5
- pandas==1.3.5
- pattern==3.6
- scikit_learn==1.0.2
- tensorboardX==2.5.1
- torch==1.9.1
- torch_geometric==2.3.1
- tqdm==4.66.1
- transformers==4.18.0

The other packages needed is shown in requirement.txt .

### DSG Parser Setup
We follow the instruction in https://github.com/seq-to-mind/DMRST_Parser to get the DSG information with minor modifications.


## Data
We support `AAEC`, and `AbstRCT`.

### Original data
Our preprocessing mainly adapts https://github.com/hitachi-nlp/graph_parser released scripts. We deeply thank the contribution from the authors of the paper.

1. Get original data by https://github.com/hitachi-nlp/graph_parser, and arrange according to the following path:

data
├── AAEC
│     ├── aaec_para_dev.mrp
│     ├── aaec_para_test.mrp
│     └── aaec_para_train.mrp
└── CDCP
      ├── cdcp_dev.mrp
      ├── cdcp_test.mrp
      └── cdcp_train.mrp


2. 
Download the pretrained model longformer-base-4096 and place it in folder `./pretrained_model/`.
Please refer to the script `sh ./data/get_data_CDCP.py` and `sh ./data/get_data_PE.py` for data processing.


## Training and Evaluating
Run `python3 ./src/train_and_eval.py`

### Outputs files
1. You can see the final result in `train.log`.
2. You can get the predicted results of the model in `pred.dev.json` and `pred.test.json`.

## Citation

If you find that the code is useful in your research, please consider citing our paper.

