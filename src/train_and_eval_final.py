import os, json, logging, time, pprint, tqdm
import sys

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, LongformerTokenizer
from torch.optim import AdamW
from model_logic_final_AAEC import Model
from data import  AMDataset, get_meaningful_ARs_turple
from utils import Summarizer, args_metric, eval_component_PE, eval_edge_PE, eval_component_cdcp, eval_edge_cdcp
from argparse import ArgumentParser, Namespace
import time
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = ArgumentParser()
parser.add_argument('-c', '--config', default="./config/CDCP_std.json")
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()
with open(args.config) as fp:
    config = json.load(fp)

config = Namespace(**config)

def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
# set_seed(config.seed)
set_seed(args.seed)

# logger and summarizer
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
output_dir = os.path.join(config.output_dir, timestamp)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
log_path = os.path.join(output_dir, "train.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', datefmt='[%Y-%m-%d %H:%M:%S]', 
                    handlers=[logging.FileHandler(os.path.join(output_dir, "train.log")), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.info(f"\n{pprint.pformat(vars(config), indent=4)}")
summarizer = Summarizer(output_dir)

# set GPU device
torch.cuda.set_device(config.gpu_device)

# output
with open(os.path.join(output_dir, 'config.json'), 'w') as fp:
    json.dump(vars(config), fp, indent=4)
best_model_path = os.path.join(output_dir, 'best_model.mdl')
dev_prediction_path = os.path.join(output_dir, 'pred.dev.json')
test_prediction_path = os.path.join(output_dir, 'pred.test.json')

# tokenizer
tokenizer = LongformerTokenizer.from_pretrained(config.plm_model_path, add_prefix_space=True)

with open(config.path_data_dir + 'data.train.json', 'r', encoding='utf-8') as fp:
    data_train = json.load(fp)
with open(config.path_data_dir + 'data.test.json', 'r', encoding='utf-8') as fp:
    data_test = json.load(fp)
with open(config.path_data_dir + 'data.dev.json', 'r', encoding='utf-8') as fp:
    data_valid = json.load(fp)



train_set = AMDataset(config, tokenizer, "train", data_train, config.dataset)
dev_set = AMDataset(config, tokenizer, "valid", data_valid, config.dataset)
test_set = AMDataset(config, tokenizer, "test", data_test, config.dataset)



train_batch_num = len(train_set) // config.train_batch_size + (len(train_set) % config.train_batch_size != 0)
dev_batch_num = len(dev_set) // config.eval_batch_size + (len(dev_set) % config.eval_batch_size != 0)
test_batch_num = len(test_set) // config.eval_batch_size + (len(test_set) % config.eval_batch_size != 0)

# initialize the model
model = Model(config, tokenizer)
model.cuda(device=config.gpu_device)

param_groups = [{'params': [p for n, p in model.named_parameters() if "PLM_model" in n], 
                'lr': config.lr_plm, 'weight_decay': 1e-5},
                {'params': [p for n, p in model.named_parameters() if all(substring not in n for substring in ['PLM_model'])], 
                'lr': config.lr_other, 'weight_decay': config.weight_decay}]
optimizer = AdamW(params=param_groups)

schedule = get_cosine_schedule_with_warmup(optimizer,
                                        num_warmup_steps=train_batch_num*config.warmup_epoch,
                                        num_training_steps=train_batch_num*config.max_epoch)

# start training
logger.info("Start training ...")
summarizer_step = 0
best_dev_epoch = -1
best_dev_scores = {
    'F1_ACTC': 0.0,
    'macro_ACTC': 0.0,
    'F1_ARI': 0.0,
    'F1_ARTC': 0.0,
    'macro_ARTC': 0.0,
    'acc_ACTC': 0.0,
    'acc_ARI': 0.0,
    'acc_ARTC': 0.0,
}

best_dev_test_scores = {}

cnt_stop = 0

loss_result = {
    'sum':[],
    'ACTC':[],
    'ARTC':[],
    'train_time':[],
    'predict_time_dev':[],
    'predict_time_test':[]
}

if config.dataset == "AAEC":
    num_AC_type = 3
    num_ARs_type = 3
elif config.dataset == "AbstRCT":
    num_AC_type = 3
    num_ARs_type = 4
elif config.dataset == "CDCP":
    num_AC_type = 5
    num_ARs_type = 3
else:
    print("The dataset is doesn't exist!")
    raise ValueError


cnt_stop = 0
for epoch in range(1, config.max_epoch+1):
    logger.info(log_path)
    logger.info(f"Epoch {epoch}")
    # training
    progress = tqdm.tqdm(total=train_batch_num, ncols=75, desc='Train {}'.format(epoch))
    model.train()
    optimizer.zero_grad()
    losses = []
    losses_ACTC = []
    losses_ARTC = []
    if epoch == 50:
        cnt_stop = 0
    elif epoch > 50 and cnt_stop >= config.early_stop: 
        break
    
    label_final_ACI_train = []
    label_final_ARI_train = []
    label_final_ARTC_train = []

    predict_final_ACI_train = []
    predict_final_ARI_train = []
    predict_final_ARTC_train = []

    start_time = time.time()
    for batch_idx, batch in enumerate(DataLoader(train_set, batch_size=config.train_batch_size // config.accumulate_step, 
                                                shuffle=True, drop_last=False, collate_fn=train_set.collate_fn)):
        # AAEC数据集经常遇到AC为空的情况：
        flag = False
        for sublist in batch.indices_ac:
            if sublist:
                flag = True
                break 
        if not flag:
            continue

        loss, AC_predict_train, AR_predict_train = model(batch, True)
        # record loss
        summarizer.scalar_summary('train/loss', loss, summarizer_step)
        summarizer_step += 1
        
        loss = loss * (1 / config.accumulate_step)
        loss.backward()
        

        # --------------------------------------
        losses.append(loss)
        if (batch_idx + 1) % config.accumulate_step == 0:
            progress.update(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clipping)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
        
        
        spans_train = batch.indices_ac
        
        # 所有label
        labels_ACI_train = batch.lbl_idxs_ac
        labels_ARI_train = batch.lbl_idxs_ar
        
        for i, row in enumerate(spans_train):
            if row:
                a_label_final_ACI_train = [(x[0], x[1], y) for x, y in zip(spans_train[i], labels_ACI_train[i].cpu().numpy().tolist())]
                a_predict_final_ACI_train = [(x[0], x[1], y) for x, y in zip(spans_train[i], torch.max(AC_predict_train[i][:, :num_AC_type], dim=-1)[1].cpu().numpy().tolist())]
                label_final_ACI_train.append(a_label_final_ACI_train) 
                predict_final_ACI_train.append(a_predict_final_ACI_train)     
            else:
                label_final_ACI_train.append([]) 
                predict_final_ACI_train.append([])    
                AC_predict_train.insert(i,[])

            # # AR
            if len(row) > 1:
                AR_predict_i_train = torch.max(AR_predict_train[i][:, num_AC_type:], dim=-1)[1]
                a_label_final_ARI_train = []
                a_label_final_ARTC_train = []
                a_predict_final_ARI_train = []
                a_predict_final_ARTC_train = []
                for meaningful_ARs_index_train, pair_train in enumerate(get_meaningful_ARs_turple(len(spans_train[i]))):
                    pair_src_train, pair_tgt_train = pair_train
                    if config.dataset == "CDCP":
                        if pair_tgt_train - pair_src_train > 12:
                            continue
                    pair_relation_train = AR_predict_i_train.cpu().numpy().tolist()[meaningful_ARs_index_train]
                    if pair_relation_train != 2:
                        a_predict_final_ARI_train.append((spans_train[i][pair_src_train][0], 
                                                spans_train[i][pair_src_train][1],
                                                spans_train[i][pair_tgt_train][0],
                                                spans_train[i][pair_tgt_train][1],
                                                ))
                        a_predict_final_ARTC_train.append((spans_train[i][pair_src_train][0], 
                                                spans_train[i][pair_src_train][1],
                                                spans_train[i][pair_tgt_train][0],
                                                spans_train[i][pair_tgt_train][1],
                                                pair_relation_train))
                        
                    pair_relation_lbl_train = labels_ARI_train[i].cpu().numpy().tolist()[meaningful_ARs_index_train]
                    assert pair_relation_lbl_train != -1
                    if pair_relation_lbl_train != 2:
                        a_label_final_ARI_train.append((spans_train[i][pair_src_train][0], 
                                                spans_train[i][pair_src_train][1],
                                                spans_train[i][pair_tgt_train][0],
                                                spans_train[i][pair_tgt_train][1],
                                                ))
                        a_label_final_ARTC_train.append((spans_train[i][pair_src_train][0], 
                                                spans_train[i][pair_src_train][1],
                                                spans_train[i][pair_tgt_train][0],
                                                spans_train[i][pair_tgt_train][1],
                                                pair_relation_lbl_train))
                
                label_final_ARI_train.append(a_label_final_ARI_train)
                label_final_ARTC_train.append(a_label_final_ARTC_train)
                predict_final_ARI_train.append(a_predict_final_ARI_train)
                predict_final_ARTC_train.append(a_predict_final_ARTC_train)

                
            else:
                label_final_ARI_train.append([])
                label_final_ARTC_train.append([])
                predict_final_ARI_train.append([])
                predict_final_ARTC_train.append([])
                AR_predict_train.insert(i,[])

            
    result_ACI_train = args_metric(label_final_ACI_train, predict_final_ACI_train)
    result_ARI_train = args_metric(label_final_ARI_train, predict_final_ARI_train)
    result_ARTC_train = args_metric(label_final_ARTC_train, predict_final_ARTC_train)   
    f1_ACI_train = result_ACI_train['f1']
    if config.dataset == "CDCP":  
        result_ACI_macro_train = eval_component_cdcp(predict_final_ACI_train, label_final_ACI_train)
        try:
            macro_ACI_train = (result_ACI_macro_train["value"]['f'] + result_ACI_macro_train["policy"]['f'] + result_ACI_macro_train["testimony"]['f'] + result_ACI_macro_train["fact"]['f'] + result_ACI_macro_train["reference"]['f'])/5
        except:
            macro_ACI_train = (result_ACI_macro_train["value"]['f'] + result_ACI_macro_train["policy"]['f'] + result_ACI_macro_train["testimony"]['f'] + result_ACI_macro_train["fact"]['f'])/4
    elif config.dataset == "AAEC": 
        result_ACI_macro_train = eval_component_PE(predict_final_ACI_train, label_final_ACI_train) 
        macro_ACI_train = (result_ACI_macro_train["Premise"]['f'] + result_ACI_macro_train["Claim"]['f'] + result_ACI_macro_train["MajorClaim"]['f'])/3

    else:
        raise ValueError
    
    f1_ARI_train = result_ARI_train['f1']
    f1_ARTC_train = result_ARTC_train['f1']
    
    if config.dataset == "CDCP":
        result_ARTC_macro_train = eval_edge_cdcp(predict_final_ARTC_train, label_final_ARTC_train)
        macro_ARTC_train = (result_ARTC_macro_train['reason']['f'] + result_ARTC_macro_train['evidence']['f'])/2
    elif config.dataset == "AAEC":  
        result_ARTC_macro_train = eval_edge_PE(predict_final_ARTC_train, label_final_ARTC_train)
        macro_ARTC_train = (result_ARTC_macro_train['Support']['f'] + result_ARTC_macro_train['Attack']['f'])/2
    else:
        raise ValueError

    acc_ACI_train = result_ACI_train['acc']
    acc_ARI_train = result_ARI_train['acc']
    acc_ARTC_train = result_ARTC_train['acc']
    

    train_scores = {
        'F1_ACTC': f1_ACI_train,
        'macro_ACTC': macro_ACI_train,
        'F1_ARI': f1_ARI_train,
        'F1_ARTC': f1_ARTC_train,
        'macro_ARTC': macro_ARTC_train,
        'acc_ACTC': acc_ACI_train,
        'acc_ARI': acc_ARI_train,
        'acc_ARTC': acc_ARTC_train
    }
    
    
    
    logger.info("--------------------------Train Scores---------------------------------")
    logger.info('ACTC - F: {:5.2f}'.format(
        train_scores['F1_ACTC'] * 100.0))
    logger.info('ACTC - Macro: {:5.2f}'.format(
        train_scores['macro_ACTC'] * 100.0))
    logger.info('ARI - F: {:5.2f}'.format(
        train_scores['F1_ARI'] * 100.0))
    logger.info('ARTC - F: {:5.2f}'.format(
        train_scores['F1_ARTC'] * 100.0))
    logger.info('ARTC - Macro: {:5.2f}'.format(
        train_scores['macro_ARTC'] * 100.0))
    logger.info('ACTC - acc: {:5.2f}'.format(
        train_scores['acc_ACTC'] * 100.0))
    logger.info('ARI - acc: {:5.2f}'.format(
        train_scores['acc_ARI'] * 100.0))
    logger.info('ARTC - acc: {:5.2f}'.format(
        train_scores['acc_ARTC'] * 100.0))
    logger.info("---------------------------------------------------------------------")

        
    progress.close()
    end_time = time.time()
    time_result = end_time - start_time
    loss_result['train_time'].append(time_result)
    losses_avg = torch.mean(torch.stack(losses)).tolist() 
    loss_result['sum'].append(losses_avg)

    
    logger.info("Average training loss : {}...".format(losses_avg))

    
    # eval dev set
    progress = tqdm.tqdm(total=dev_batch_num, ncols=75, desc='Dev {}'.format(epoch))
    model.eval()
    best_dev_flag = False
    
    
    label_final_ACI = []
    label_final_ARI = []
    label_final_ARTC = []
    predict_final_ACI = []
    predict_final_ARI = []
    predict_final_ARTC = []
    best_outputs_predict_dev = []

    start_time = time.time()
    for batch_idx, batch in enumerate(DataLoader(dev_set, batch_size=config.eval_batch_size, 
                                                shuffle=False, collate_fn=dev_set.collate_fn)):
        progress.update(1)
        __, AC_predict, AR_predict = model(batch, False)

        spans = batch.indices_ac
        
        # 所有label
        labels_ACI = batch.lbl_idxs_ac
        labels_ARI = batch.lbl_idxs_ar
        
        for i, row in enumerate(spans):
            if row:
                a_label_final_ACI = [(x[0], x[1], y) for x, y in zip(spans[i], labels_ACI[i].cpu().numpy().tolist())]
                a_predict_final_ACI = [(x[0], x[1], y) for x, y in zip(spans[i], torch.max(AC_predict[i][:, :num_AC_type], dim=-1)[1].cpu().numpy().tolist())]
                label_final_ACI.append(a_label_final_ACI) 
                predict_final_ACI.append(a_predict_final_ACI)     
            else:
                label_final_ACI.append([]) 
                predict_final_ACI.append([])    
                AC_predict.insert(i,[])

            # # AR
            if len(row) > 1:
                AR_predict_i = torch.max(AR_predict[i][:, num_AC_type:], dim=-1)[1]
                a_label_final_ARI = []
                a_label_final_ARTC = []
                a_predict_final_ARI = []
                a_predict_final_ARTC = []
                for meaningful_ARs_index, pair in enumerate(get_meaningful_ARs_turple(len(spans[i]))):
                    pair_src, pair_tgt = pair
                    if config.dataset == "CDCP":
                        if pair_tgt - pair_src > 12:
                            continue
                    pair_relation = AR_predict_i.cpu().numpy().tolist()[meaningful_ARs_index]
                    if pair_relation != 2:
                        a_predict_final_ARI.append((spans[i][pair_src][0], 
                                                spans[i][pair_src][1],
                                                spans[i][pair_tgt][0],
                                                spans[i][pair_tgt][1],
                                                ))
                        a_predict_final_ARTC.append((spans[i][pair_src][0], 
                                                spans[i][pair_src][1],
                                                spans[i][pair_tgt][0],
                                                spans[i][pair_tgt][1],
                                                pair_relation))
                        
                    pair_relation_lbl = labels_ARI[i].cpu().numpy().tolist()[meaningful_ARs_index]
                    assert pair_relation_lbl != -1
                    if pair_relation_lbl != 2:
                        a_label_final_ARI.append((spans[i][pair_src][0], 
                                                spans[i][pair_src][1],
                                                spans[i][pair_tgt][0],
                                                spans[i][pair_tgt][1],
                                                ))
                        a_label_final_ARTC.append((spans[i][pair_src][0], 
                                                spans[i][pair_src][1],
                                                spans[i][pair_tgt][0],
                                                spans[i][pair_tgt][1],
                                                pair_relation_lbl))
                
                label_final_ARI.append(a_label_final_ARI)
                label_final_ARTC.append(a_label_final_ARTC)
                predict_final_ARI.append(a_predict_final_ARI)
                predict_final_ARTC.append(a_predict_final_ARTC)

                
            else:
                label_final_ARI.append([])
                label_final_ARTC.append([])
                predict_final_ARI.append([])
                predict_final_ARTC.append([])
                AR_predict.insert(i,[])

            
    outputs_result = []
    for i, __ in enumerate(label_final_ACI):
        an_output_result = {}
        an_output_result['essay_id'] = data_valid[i]['essay_id']
        an_output_result['para_text'] = data_valid[i]['para_text']
        an_output_result['adu_spans'] = data_valid[i]['adu_spans']
        an_output_result['AC_types'] = data_valid[i]['AC_types']
        an_output_result['AR_pairs'] = data_valid[i]['AR_pairs']
        an_output_result['AR_types'] = data_valid[i]['AR_types']
        an_output_result['predict_ACI'] = repr(predict_final_ACI[i])
        an_output_result['predict_ARI'] = repr(predict_final_ARI[i])
        an_output_result['predict_ARTC'] = repr(predict_final_ARTC[i])
        outputs_result.append(an_output_result)
        

    result_ACI = args_metric(label_final_ACI, predict_final_ACI)
    result_ARI = args_metric(label_final_ARI, predict_final_ARI)
    result_ARTC = args_metric(label_final_ARTC, predict_final_ARTC)   
    # f1_ACS = result_ACS['f1']
    f1_ACI = result_ACI['f1']
    if config.dataset == "CDCP":  
        result_ACI_macro = eval_component_cdcp(predict_final_ACI, label_final_ACI)
        try:
            macro_ACI = (result_ACI_macro["value"]['f'] + result_ACI_macro["policy"]['f'] + result_ACI_macro["testimony"]['f'] + result_ACI_macro["fact"]['f'] + result_ACI_macro["reference"]['f'])/5
        except:
            macro_ACI = (result_ACI_macro["value"]['f'] + result_ACI_macro["policy"]['f'] + result_ACI_macro["testimony"]['f'] + result_ACI_macro["fact"]['f'])/4
    elif config.dataset == "AAEC": 
        result_ACI_macro = eval_component_PE(predict_final_ACI, label_final_ACI) 
        macro_ACI = (result_ACI_macro["Premise"]['f'] + result_ACI_macro["Claim"]['f'] + result_ACI_macro["MajorClaim"]['f'])/3

    else:
        raise ValueError
    
    f1_ARI = result_ARI['f1']
    f1_ARTC = result_ARTC['f1']
    
    if config.dataset == "CDCP":
        result_ARTC_macro = eval_edge_cdcp(predict_final_ARTC, label_final_ARTC)
        macro_ARTC = (result_ARTC_macro['reason']['f'] + result_ARTC_macro['evidence']['f'])/2
    elif config.dataset == "AAEC":  
        result_ARTC_macro = eval_edge_PE(predict_final_ARTC, label_final_ARTC)
        macro_ARTC = (result_ARTC_macro['Support']['f'] + result_ARTC_macro['Attack']['f'])/2
    else:
        raise ValueError

    acc_ACI = result_ACI['acc']
    acc_ARI = result_ARI['acc']
    acc_ARTC = result_ARTC['acc']
    
    progress.close()
    end_time = time.time()
    time_result = end_time - start_time
    loss_result['predict_time_dev'].append(time_result)

    dev_scores = {
        'F1_ACTC': f1_ACI,
        'macro_ACTC': macro_ACI,
        'F1_ARI': f1_ARI,
        'F1_ARTC': f1_ARTC,
        'macro_ARTC': macro_ARTC,
        'acc_ACTC': acc_ACI,
        'acc_ARI': acc_ARI,
        'acc_ARTC': acc_ARTC
    }
    
    
    
    logger.info("--------------------------Dev Scores---------------------------------")
    logger.info('ACTC - F: {:5.2f}'.format(
        dev_scores['F1_ACTC'] * 100.0))
    logger.info('ACTC - Macro: {:5.2f}'.format(
        dev_scores['macro_ACTC'] * 100.0))
    logger.info('ARI - F: {:5.2f}'.format(
        dev_scores['F1_ARI'] * 100.0))
    logger.info('ARTC - F: {:5.2f}'.format(
        dev_scores['F1_ARTC'] * 100.0))
    logger.info('ARTC - Macro: {:5.2f}'.format(
        dev_scores['macro_ARTC'] * 100.0))
    logger.info('ACTC - acc: {:5.2f}'.format(
        dev_scores['acc_ACTC'] * 100.0))
    logger.info('ARI - acc: {:5.2f}'.format(
        dev_scores['acc_ARI'] * 100.0))
    logger.info('ARTC - acc: {:5.2f}'.format(
        dev_scores['acc_ARTC'] * 100.0))
    logger.info("---------------------------------------------------------------------")
    
    # check best dev model
    dev_score_sum = dev_scores['F1_ACTC'] + (dev_scores['F1_ARI'] + dev_scores['F1_ARTC'])/2
    best_dev_score_sum = best_dev_scores['F1_ACTC'] + (best_dev_scores['F1_ARI'] + best_dev_scores['F1_ARTC'])/2
    if dev_score_sum > best_dev_score_sum:
        best_dev_flag = True
        cnt_stop = 0
    else:
        cnt_stop += 1
        
    # if best dev, save model and evaluate test set
    if best_dev_flag:    
        best_dev_scores = dev_scores
        best_dev_epoch = epoch
    
        # save best model
        logger.info('Saving best model')
        torch.save(model.state_dict(), best_model_path)
        
        # save dev result
        best_outputs_predict_dev = outputs_result
        with open(dev_prediction_path, 'w', encoding='utf-8') as fp:
            json.dump(best_outputs_predict_dev, fp, indent=4, ensure_ascii=False)
            
    if True:
        # eval test set
        test_start_time = time.time()
        test_progress = tqdm.tqdm(total=test_batch_num, ncols=75, desc='Test {}'.format(epoch))
        

        test_label_final_ACI = []
        test_label_final_ARI = []
        test_label_final_ARTC = []
        test_predict_final_ACI = []
        test_predict_final_ARI = []
        test_predict_final_ARTC = []
        
        
        for batch_idx, batch in enumerate(DataLoader(test_set, batch_size=config.eval_batch_size, 
                                                    shuffle=False, collate_fn=test_set.collate_fn)):
            test_progress.update(1)
            
            __, test_AC_predict, test_AR_predict = model(batch, False)

            test_spans = batch.indices_ac
            # 所有label
            test_labels_ACI = batch.lbl_idxs_ac
            test_labels_ARI = batch.lbl_idxs_ar
            for i, row in enumerate(test_spans):
                if row:
                    test_a_label_final_ACI = [(x[0], x[1], y) for x, y in zip(test_spans[i], test_labels_ACI[i].cpu().numpy().tolist())]
                    test_a_predict_final_ACI = [(x[0], x[1], y) for x, y in zip(test_spans[i], torch.max(test_AC_predict[i][:, :num_AC_type], dim=-1)[1].cpu().numpy().tolist())]
                    test_label_final_ACI.append(test_a_label_final_ACI)
                    test_predict_final_ACI.append(test_a_predict_final_ACI)
                    
                else:
                    test_AC_predict.insert(i,[])
                    test_label_final_ACI.append([])
                    test_predict_final_ACI.append([])

                if len(row)>1:
                    test_AR_predict_i = torch.max(test_AR_predict[i][:, num_AC_type:], dim=-1)[1]
                    test_a_label_final_ARI = []
                    test_a_label_final_ARTC = []
                    test_a_predict_final_ARI = []
                    test_a_predict_final_ARTC = []
                    for test_meaningful_ARs_index, test_pair in enumerate(get_meaningful_ARs_turple(len(test_spans[i]))):
                        test_pair_src, test_pair_tgt = test_pair
                        if config.dataset == "CDCP":
                            if test_pair_tgt - test_pair_src > 12:
                                continue
                        test_pair_relation = test_AR_predict_i.cpu().numpy().tolist()[test_meaningful_ARs_index]
                        if test_pair_relation != 2:
                            test_a_predict_final_ARI.append((test_spans[i][test_pair_src][0], 
                                                    test_spans[i][test_pair_src][1],
                                                    test_spans[i][test_pair_tgt][0],
                                                    test_spans[i][test_pair_tgt][1],
                                                    ))
                            test_a_predict_final_ARTC.append((test_spans[i][test_pair_src][0], 
                                                    test_spans[i][test_pair_src][1],
                                                    test_spans[i][test_pair_tgt][0],
                                                    test_spans[i][test_pair_tgt][1],
                                                    test_pair_relation))
                            
                        test_pair_relation_lbl = test_labels_ARI[i].cpu().numpy().tolist()[test_meaningful_ARs_index]
                        assert test_pair_relation_lbl != -1
                        if test_pair_relation_lbl != 2:
                            test_a_label_final_ARI.append((test_spans[i][test_pair_src][0], 
                                                    test_spans[i][test_pair_src][1],
                                                    test_spans[i][test_pair_tgt][0],
                                                    test_spans[i][test_pair_tgt][1],
                                                    ))
                            test_a_label_final_ARTC.append((test_spans[i][test_pair_src][0], 
                                                    test_spans[i][test_pair_src][1],
                                                    test_spans[i][test_pair_tgt][0],
                                                    test_spans[i][test_pair_tgt][1],
                                                    test_pair_relation_lbl))
                    test_label_final_ARI.append(test_a_label_final_ARI)
                    test_label_final_ARTC.append(test_a_label_final_ARTC)
                    test_predict_final_ARI.append(test_a_predict_final_ARI)
                    test_predict_final_ARTC.append(test_a_predict_final_ARTC)
                else:
                    test_label_final_ARI.append([])
                    test_label_final_ARTC.append([])
                    test_predict_final_ARI.append([])
                    test_predict_final_ARTC.append([])
                    test_AR_predict.insert(i,[])
                                    
                
                
        outputs_result_test = []
        for i, __ in enumerate(test_label_final_ACI):
            an_output_result_test = {}
            an_output_result_test['essay_id'] = data_test[i]['essay_id']
            an_output_result_test['para_text'] = data_test[i]['para_text']
            an_output_result_test['adu_spans'] = data_test[i]['adu_spans']
            an_output_result_test['AC_types'] = data_test[i]['AC_types']
            an_output_result_test['AR_pairs'] = data_test[i]['AR_pairs']
            an_output_result_test['AR_types'] = data_test[i]['AR_types']
            an_output_result_test['predict_ACI'] = repr(test_predict_final_ACI[i])
            an_output_result_test['predict_ARI'] = repr(test_predict_final_ARI[i])
            an_output_result_test['predict_ARTC'] = repr(test_predict_final_ARTC[i])
            outputs_result_test.append(an_output_result_test)
        
        test_result_ACI = args_metric(test_label_final_ACI, test_predict_final_ACI)
        test_result_ARI = args_metric(test_label_final_ARI, test_predict_final_ARI)
        test_result_ARTC = args_metric(test_label_final_ARTC, test_predict_final_ARTC)
        
        
        test_f1_ACI = test_result_ACI['f1']
        if config.dataset == "CDCP":  
                test_result_ACI_macro = eval_component_cdcp(test_predict_final_ACI, test_label_final_ACI)
                test_macro_ACI = (test_result_ACI_macro["value"]['f'] + test_result_ACI_macro["policy"]['f'] + test_result_ACI_macro["testimony"]['f'] + test_result_ACI_macro["fact"]['f'] + test_result_ACI_macro["reference"]['f'])/5
        elif config.dataset == "AAEC":  
            test_result_ACI_macro = eval_component_PE(test_predict_final_ACI, test_label_final_ACI)
            test_macro_ACI = (test_result_ACI_macro["Premise"]['f'] + test_result_ACI_macro["Claim"]['f'] + test_result_ACI_macro["MajorClaim"]['f'])/3
        else:
            raise ValueError

        test_f1_ARI = test_result_ARI['f1']
        test_f1_ARTC = test_result_ARTC['f1']
        if config.dataset == "CDCP":  
            test_result_ARTC_macro = eval_edge_cdcp(test_predict_final_ARTC, test_label_final_ARTC)
            test_macro_ARTC = (test_result_ARTC_macro['reason']['f'] + test_result_ARTC_macro['evidence']['f'])/2
        elif config.dataset == "AAEC":
            test_result_ARTC_macro = eval_edge_PE(test_predict_final_ARTC, test_label_final_ARTC)
            test_macro_ARTC = (test_result_ARTC_macro['Support']['f'] + test_result_ARTC_macro['Attack']['f'])/2
        else:
            raise ValueError
        test_acc_ACI = test_result_ACI['acc']
        test_acc_ARI = test_result_ARI['acc']
        test_acc_ARTC = test_result_ARTC['acc']
        
        progress.close()
        end_time = time.time()
        time_result = end_time - start_time
        loss_result['predict_time_test'].append(time_result)
        
        test_scores = {
            'F1_ACTC': test_f1_ACI,
            'macro_ACTC': test_macro_ACI,
            'F1_ARI': test_f1_ARI,
            'F1_ARTC': test_f1_ARTC,
            'macro_ARTC': test_macro_ARTC,
            'acc_ACTC': test_acc_ACI,
            'acc_ARI': test_acc_ARI,
            'acc_ARTC': test_acc_ARTC
        }
        if best_dev_flag:
            best_dev_test_scores = test_scores
        
        # print scores
        logger.info("--------------------------TEST Scores---------------------------------")
        logger.info('ACTC - F: {:5.2f}'.format(
            test_scores['F1_ACTC'] * 100.0))
        logger.info('ACTC - Macro: {:5.2f}'.format(
            test_scores['macro_ACTC'] * 100.0))
        logger.info('ARI - F: {:5.2f}'.format(
            test_scores['F1_ARI'] * 100.0))
        logger.info('ARTC - F: {:5.2f}'.format(
            test_scores['F1_ARTC'] * 100.0))
        logger.info('ARTC - Macro: {:5.2f}'.format(
            test_scores['macro_ARTC'] * 100.0))
        logger.info('ACTC - acc: {:5.2f}'.format(
            test_scores['acc_ACTC'] * 100.0))
        logger.info('ARI - acc: {:5.2f}'.format(
            test_scores['acc_ARI'] * 100.0))
        logger.info('ARTC - acc: {:5.2f}'.format(
            test_scores['acc_ARTC'] * 100.0))
        logger.info("---------------------------------------------------------------------")
            
        # save test result
    if best_dev_flag:
        with open(test_prediction_path, 'w', encoding='utf-8') as fp:
            json.dump(outputs_result_test, fp, indent=4, ensure_ascii=False)
            
    logger.info({"epoch": epoch, "dev_scores": dev_scores})
    logger.info("Current best:-----")
    logger.info({"best_dev_epoch": best_dev_epoch, "best_dev_scores": best_dev_scores})
    logger.info({"test_score_in_best_dev": best_dev_test_scores})
    logger.info("------------------\n\n")
    
    try:
        df = pd.DataFrame(loss_result)
        df.to_excel(output_dir+'/time_and_loss_'+config.dataset+'.xlsx', index=False)
    except:
        pass

logger.info("train complete!\nResults:")
logger.info({"best_dev_epoch": best_dev_epoch, "best_dev_scores": best_dev_scores})
logger.info({"test_score_in_best_dev": best_dev_test_scores})
