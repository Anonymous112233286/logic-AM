import json, logging
import torch
from torch.utils.data import Dataset
from collections import namedtuple
import numpy as np
import random

logger = logging.getLogger(__name__)

gen_batch_fields = ['enc_idxs', 'enc_attn', 'glb_attn', 'lbl_idxs_ac', 'lbl_idxs_ar', 'num_ac', 'indices_ac']
GenBatch = namedtuple('GenBatch', field_names=gen_batch_fields, defaults=[None] * len(gen_batch_fields))


def mapping_ARs_token_index_directed(relation_tuple, num_AC):
    
    relation_src = relation_tuple[0]
    relation_tgt = relation_tuple[1]
    if relation_tgt < 0 or relation_tgt >= num_AC or relation_src < 0 or relation_src >= num_AC:
        print("ar pair idx error")
        raise ValueError

    if relation_tgt < relation_src:
        flag_sm2big = False  
        smaller_index = relation_tgt
        bigger_index = relation_src
    elif relation_tgt > relation_src:
        flag_sm2big = True  
        smaller_index = relation_src
        bigger_index = relation_tgt
    else:
        print("ar pair idx equal")
        raise ValueError

    num_AC_ = num_AC - 1
    ARs_token_index = (num_AC_ + num_AC_-smaller_index+1) * smaller_index / 2 + (bigger_index - smaller_index) - 1
    
    return int(ARs_token_index)

def get_meaningful_ARs_turple(num_span):
        
    list_meaningful_turple = []
    for i in range(num_span - 1):
        for j in range(i+1, num_span):
            list_meaningful_turple.append((i, j))

    return list_meaningful_turple


class AMDataset(Dataset):
    def __init__(self, config, tokenizer, data_name, data_list, dataset, unseen_types=[]):
        self.config = config
        self.tokenizer = tokenizer
        self.data_name = data_name
        self.data_list = data_list
        self.data = []
        
        self.dataset = dataset

        if self.dataset == "CDCP":
            self.num_AC_type = 5
            self.num_ARs_type = 3
        elif self.dataset == "AAEC":
            self.num_AC_type = 3
            self.num_ARs_type = 3

        
        self.load_data(unseen_types)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self, unseen_types):
        
        for sample in self.data_list:
            input = sample['para_text'].split()
            para_length = len(input)
            input.insert(0, '<s>')
            input.append('</s>')
            

            label_list_ac = []
            label_list_ar = []


            indices_ac = eval(sample['adu_spans'])
            num_AC = len(indices_ac)

        
            if self.dataset == "CDCP":
                mapping = {'value': 0, 'policy': 1, 'testimony': 2, 'fact': 3, 'reference': 4, "reason": 0, "evidence": 1, "no relation":2}
            elif self.dataset == "AAEC":
                mapping = {'Premise': 0, 'Claim': 1, 'MajorClaim': 2, "Support": 0, "Attack": 1, "no relation":2}

            ACs_label = eval(sample['AC_types'])
            
            ACs_label_mapping = [mapping[item] for item in ACs_label]
            label_list_ac.extend(ACs_label_mapping)
            
            
            ARs_types = eval(sample['AR_types'])
            ARs_pairs = eval(sample['AR_pairs'])
            

            ARs_label = [2] * int((num_AC-1) * num_AC / 2)
            for pair, type in zip(ARs_pairs, ARs_types):
                ARs_index = mapping_ARs_token_index_directed(pair, num_AC)
                ARs_label[ARs_index] = mapping[type]
            
            label_list_ar.extend(ARs_label)

            self.data.append({
                'input': input,
                'target_ac': label_list_ac,
                'target_ar': label_list_ar,
                'num_ac': num_AC,
                'indices_ac': indices_ac
            })
            
        logger.info(f'Loaded {len(self)} instances from {self.data_name}')
        

    def collate_fn(self, batch):
        

        targets_ac = [torch.tensor(x['target_ac']).cuda() for x in batch]
        targets_ar = [torch.tensor(x['target_ar']).cuda() for x in batch]


        input_text = [x['input'] for x in batch]
        max_length_input = max(len(word_list) for word_list in input_text)
        pad_token = self.tokenizer.pad_token
        enc_word_list = [word_list + [pad_token] * (max_length_input - len(word_list)) for word_list in input_text]
        enc_idxs = []
        for word_list in enc_word_list:
            word_tokenized = self.tokenizer.convert_tokens_to_ids(word_list)
            enc_idxs.append(word_tokenized)
        
        
        enc_idxs = torch.tensor(enc_idxs)
        enc_attn = torch.zeros(enc_idxs.size(0), enc_idxs.size(1), dtype=torch.long)
        glb_attn = torch.zeros(enc_idxs.size(0), enc_idxs.size(1), dtype=torch.long)
        for i in range(enc_idxs.size(0)):
            for j in range(enc_idxs.size(1)):
                if enc_idxs[i][j] != 1:
                    enc_attn[i][j] = 1
        
        for i in range(enc_idxs.size(0)):
            glb_attn[i][0] = 1
        
        num_ac = [x['num_ac'] for x in batch]
        indices_ac = [x['indices_ac'] for x in batch]
        
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        glb_attn = glb_attn.cuda()

        return GenBatch(
            enc_idxs=enc_idxs,
            enc_attn=enc_attn,
            glb_attn=glb_attn,
            lbl_idxs_ac=targets_ac,   
            lbl_idxs_ar=targets_ar,                                
            num_ac=num_ac,
            indices_ac=indices_ac
        )
