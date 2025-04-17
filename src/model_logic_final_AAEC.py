import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import LongformerModel
from sparsemax import Sparsemax
from collections import OrderedDict
import random

class Find(nn.Module):
    def __init__(self, k, emb_dim, trans_dim, tokenizer, PLM_path, device, gpu_id):
        super(Find, self).__init__()

        self.tokenizer = tokenizer
        self.PLM_model = LongformerModel.from_pretrained(PLM_path,
                                        output_hidden_states=False,
                                        output_attentions=False).cuda(device=gpu_id)
        self.device = device
        self.emb_dim = emb_dim
        self.trans_dim = trans_dim
        self.softmax = nn.Softmax(dim=-1)
        self.sparsemax = Sparsemax(dim=-1)
        self.dropout = nn.Dropout(0.2)
        self.topk = k
        self.gru = nn.GRU(emb_dim, trans_dim // 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.context = nn.Sequential(nn.Dropout(0.1), nn.Linear(emb_dim, trans_dim))
        self.Wfindm = nn.Sequential(nn.Dropout(0.1), nn.Linear(2 * trans_dim, 1, bias=False))


        
    def forward(self, data, flag):
        enc_idxs = data.enc_idxs
        enc_attn = data.enc_attn
        glb_attn = data.glb_attn
        indices_ac = data.indices_ac


        outputs = self.PLM_model(
            input_ids=enc_idxs,
            global_attention_mask=glb_attn,
            attention_mask=enc_attn,
        )

        all_tokens_representations = outputs.last_hidden_state[:,1:,:]

        rep_ac_batch = []
        rep_context_batch = []
        for i, a_indice_ac in enumerate(indices_ac):
            rep_ac = []
            if a_indice_ac:
                for ac_span in a_indice_ac:

                    a_rep_ac = all_tokens_representations[i, ac_span[0]:ac_span[1]+1,:].unsqueeze(0)
                    a_rep_ac = torch.mean(a_rep_ac,dim=1)  

                    rep_ac.append(a_rep_ac)
                rep_ac = torch.stack(rep_ac, dim=0)
                rep_ac = rep_ac.squeeze(1).unsqueeze(0)
                rep_ac = self.gru(rep_ac)[0]
                rep_ac_batch.append(rep_ac.squeeze(0).unsqueeze(1))
            else:
                rep_ac_batch.append([])
            

            a_enc_attn = enc_attn[i][1:]
            rep_context = all_tokens_representations[i]
            a_enc_attn = torch.nonzero(a_enc_attn).squeeze()
            rep_context = rep_context[a_enc_attn][:-1].unsqueeze(0)
            rep_context = torch.mean(rep_context, dim=1)
            rep_context = self.context(rep_context)
            rep_context_batch.append(rep_context)
            

        rep_context_batch = torch.stack(rep_context_batch, dim=0)


        rep_ac_1_hop_batch = []
        scores_ac_1_hop_batch = []
        indices_topk_1_hop_batch = []
        for i, a_rep_ac_batch in enumerate(rep_ac_batch):
            if a_rep_ac_batch != []:

                a_rep_all_batch = torch.cat([a_rep_ac_batch, rep_context_batch[i:i+1]], dim=0)
                a_rep_all_1_hop_tgt = a_rep_all_batch.squeeze(1).unsqueeze(0).repeat(a_rep_all_batch.size(0), 1, 1)  
                a_rep_ac_1_hop_selected, score1, indices_topk = self.findm_attention(a_rep_ac_batch, a_rep_all_1_hop_tgt[:-1], flag)
                indices_topk_1_hop_batch.append(indices_topk)
                rep_ac_1_hop_batch.append(a_rep_ac_1_hop_selected)
                scores_ac_1_hop_batch.append(score1)
            else:

                indices_topk_1_hop_batch.append([])
                rep_ac_1_hop_batch.append([])
                scores_ac_1_hop_batch.append([])

        return rep_ac_batch, rep_context_batch, rep_ac_1_hop_batch, scores_ac_1_hop_batch, indices_topk_1_hop_batch



    def findm_attention(self, query, key, flag):

        score1 = self.Wfindm(torch.cat([query.repeat(1, key.size(1), 1), key], dim=-1))
        score1 = torch.sigmoid(score1)

        ind1_topk = torch.topk(score1.view(query.size(0), -1), k=min(self.topk, key.size(1)), dim=1)[1]
        v1 = []
        scores_selected = []
        for i in range(ind1_topk.size(0)):
            ind1_topk_ = ind1_topk[i]
            key_ = key[i]
            scores = score1[i]
            v1_ = key_[ind1_topk_]
            scores_ = scores[ind1_topk_]
            v1.append(v1_)
            scores_selected.append(scores_)

        v1 = torch.stack(v1, dim=0)
        scores_selected = torch.stack(scores_selected, dim=0)


        return v1, scores_selected, ind1_topk

# Multi-hop logic reasoner
class Model(nn.Module):
    
    # def __init__(self, qrel_dic, emb_dim, trans_dim, att_dim, rel_dim, nrel, nclause, nclause1, nclause2, nclause3, ktop):
    def __init__(self, config, tokenizer):
        super(Model, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        gpu_id = 'cuda:' + str(self.config.gpu_device)
        self.device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")

        self.find = Find(config.ktop, config.emb_dim, config.trans_dim, self.tokenizer, self.config.plm_model_path, self.device, self.config.gpu_device)
        if self.config.dataset == "CDCP":
            self.qrel_dic_ac = OrderedDict([['value', 0], ['policy', 1], ['testimony', 2], ['fact', 3], ['reference', 4]])
            self.qrel_dic_ar = OrderedDict([['reason', 0], ['evidence', 1], ['no relation', 2]])
            self.nrel = config.nrel
        elif self.config.dataset == "AAEC":
            self.qrel_dic_ac = OrderedDict([['Premise', 0], ['Claim', 1], ['MajorClaim', 2]])
            self.qrel_dic_ar = OrderedDict([['Support', 0], ['Attack', 1], ['no relation', 2]])
            self.nrel = config.nrel

        self.emb_dim = config.emb_dim
        self.att_dim = config.att_dim
        self.trans_dim = config.trans_dim

        self.nclause = config.nclause
        self.nclause1 = config.nclause1
        self.nclause2 = config.nclause2
        self.nclause3 = config.nclause3
        self.ktop = config.ktop

        self.query_emb = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(len(self.qrel_dic_ac) + len(self.qrel_dic_ar), config.att_dim)), requires_grad=True)
        self.key_emb = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(self.nrel, config.att_dim)), requires_grad=True)
        self.pred0_emb = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(config.nclause1, config.att_dim)), requires_grad=True)
        self.pred1_emb = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(config.nclause2, config.att_dim)), requires_grad=True)

        self.Wkey = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(config.att_dim, int(config.nclause * config.att_dim / 2))), requires_grad=True)
        self.Wquery = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(config.att_dim, int(config.nclause * config.att_dim / 2))), requires_grad=True)
        self.Wkey_0 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(config.att_dim, config.att_dim)), requires_grad=True)
        self.Wpred_0 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(config.att_dim, config.att_dim)), requires_grad=True)
        self.Wkey_l1 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(config.att_dim, config.att_dim)), requires_grad=True)
        self.Wkey_r1 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(config.att_dim, config.att_dim)), requires_grad=True)
        self.Wpred_l1 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(config.att_dim, config.att_dim)), requires_grad=True)
        self.Wpred_r1 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(config.att_dim, config.att_dim)), requires_grad=True)
        self.Wvalue_1 = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(config.att_dim, config.att_dim)), requires_grad=True)

        self.Wrel = nn.Linear(2 * config.trans_dim, self.nrel)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.sparsemax = Sparsemax(dim=-1)


    def forward(self, data, flag):
        

        rep_ac_batch, rep_context_batch, rep_ac_1_hop_batch, scores_ac_1_hop_batch, indices_topk_1_hop_batch  = self.find(data, flag)

        # AC任务的推理
        final_score_all_batch_AC = []
        for i, a_rep_ac_batch in enumerate(rep_ac_batch):
            if a_rep_ac_batch != []:
                a_rep_ac_1_hop_batch = rep_ac_1_hop_batch[i]
                a_indices_topk_1_hop_batch = indices_topk_1_hop_batch[i]
                
                a_rep_context_batch = rep_context_batch[i].unsqueeze(0).repeat([a_rep_ac_batch.size(0), 1, 1])
                score_0_AC = self.ilp_hop_0_new(a_rep_ac_batch, a_rep_context_batch)

                a_indices_topk_1_hop_batch_dim_rep = a_indices_topk_1_hop_batch.unsqueeze(-1).repeat(1, 1, a_rep_context_batch.size(-1))
                a_rep_ac_hop1_src = a_rep_ac_batch.repeat(1, a_indices_topk_1_hop_batch_dim_rep.size(1), 1)
                a_rep_ac_hop1_middle = a_rep_ac_1_hop_batch
                a_rep_context_batch_tgt = a_rep_context_batch.repeat(1, a_indices_topk_1_hop_batch_dim_rep.size(1), 1)
                score_1_AC = self.ilp_hop_1_new(a_rep_ac_hop1_src, a_rep_context_batch_tgt, a_rep_ac_hop1_middle, scores_ac_1_hop_batch[i])

                final_score_all_AC = []
                for qrel in self.qrel_dic_ac.values():
                    query_emb = self.query_emb[qrel].unsqueeze(0)
                    final_score_AC = self.ilp_all(score_0_AC, score_1_AC, query_emb)
                    final_score_all_AC.append(final_score_AC)
                for qrel in self.qrel_dic_ar.values():
                    query_emb = self.query_emb[qrel+len(self.qrel_dic_ac)].unsqueeze(0)
                    final_score_AC = self.ilp_all(score_0_AC, score_1_AC, query_emb)
                    final_score_all_AC.append(final_score_AC)

                final_score_all_AC = torch.cat(final_score_all_AC, dim=1)
                final_score_all_batch_AC.append(final_score_all_AC)

        loss_AC = -1
        if final_score_all_batch_AC != []:
            logits_ACs_merge_batch = torch.cat(final_score_all_batch_AC, dim=-2)
            labels_ACs = data.lbl_idxs_ac
            labels_ACs= [tensor for tensor in labels_ACs if tensor.numel() > 0]
            labels_ACs_merge_batch = torch.cat(labels_ACs, dim=-1)

            loss_AC = self.compute_loss(logits_ACs_merge_batch, labels_ACs_merge_batch)

        # AR任务的推理
        final_score_all_batch_AR = []

        for i, a_rep_ac_batch in enumerate(rep_ac_batch):
            if a_rep_ac_batch != []:
                if a_rep_ac_batch.size(0) > 1:
                    a_rep_ac_1_hop_batch = rep_ac_1_hop_batch[i]
                    a_scores_ac_1_hop_batch = scores_ac_1_hop_batch[i]
                    a_indices_topk_1_hop_batch = indices_topk_1_hop_batch[i]
                    ac_num = a_rep_ac_batch.size(0)
                    

                    a_rep_ac_hop0_src = []
                    a_rep_ac_hop0_tgt = []


                    for ac_i in range(ac_num - 1):

                        a_rep_ac_hop0_src.append(a_rep_ac_batch[ac_i+1:])
                        a_rep_ac_hop0_tgt.append(a_rep_ac_batch[ac_i].unsqueeze(0).repeat([a_rep_ac_batch.size(0) - ac_i - 1, 1, 1]))


                    a_rep_ac_hop0_src = torch.cat(a_rep_ac_hop0_src, dim=0)
                    a_rep_ac_hop0_tgt = torch.cat(a_rep_ac_hop0_tgt, dim=0)

                    score_0_AR = self.ilp_hop_0_new(a_rep_ac_hop0_src, a_rep_ac_hop0_tgt)
                    
                    # 1跳
                    a_indices_topk_1_hop_batch_dim_rep = a_indices_topk_1_hop_batch.unsqueeze(-1).repeat(1, 1, a_rep_context_batch.size(-1))
                    a_rep_ac_hop1_src = []
                    a_rep_ac_hop1_tgt = []
                    a_rep_ac_hop1_middle = []
                    a_scores_hop1_middle = []


                    for ac_i in range(ac_num - 1):
                        a_rep_ac_hop1_src.append(a_rep_ac_batch[ac_i+1:].repeat(1, a_indices_topk_1_hop_batch_dim_rep.size(1), 1))                  
                        a_rep_ac_hop1_middle.append(a_rep_ac_1_hop_batch[ac_i+1:])
                        a_scores_hop1_middle.append(a_scores_ac_1_hop_batch[ac_i+1:])
                        a_rep_ac_hop1_tgt.append(a_rep_ac_batch[ac_i].unsqueeze(0).repeat([ac_num - ac_i - 1, a_indices_topk_1_hop_batch_dim_rep.size(1), 1]))
                        
                    a_rep_ac_hop1_src = torch.cat(a_rep_ac_hop1_src, dim=0)
                    a_rep_ac_hop1_tgt = torch.cat(a_rep_ac_hop1_tgt, dim=0)
                    a_rep_ac_hop1_middle = torch.cat(a_rep_ac_hop1_middle, dim=0)
                    a_scores_hop1_middle = torch.cat(a_scores_hop1_middle, dim=0)

                    score_1_AR = self.ilp_hop_1_new(a_rep_ac_hop1_src, a_rep_ac_hop1_tgt, a_rep_ac_hop1_middle, a_scores_hop1_middle)

                    final_score_all_AR = []
                    for qrel in self.qrel_dic_ac.values():
                        query_emb = self.query_emb[qrel].unsqueeze(0)
                        final_score_AR = self.ilp_all(score_0_AR, score_1_AR, query_emb)
                        final_score_all_AR.append(final_score_AR)
                    
                    for qrel in self.qrel_dic_ar.values():
                        query_emb = self.query_emb[qrel+len(self.qrel_dic_ac)].unsqueeze(0)
                        final_score_AR = self.ilp_all(score_0_AR, score_1_AR, query_emb)
                        final_score_all_AR.append(final_score_AR)

                    final_score_all_AR = torch.cat(final_score_all_AR, dim=1)
                    final_score_all_batch_AR.append(final_score_all_AR)

        # 可能会不存在
        loss_AR = -1
        if final_score_all_batch_AR != []:
            logits_ARs_merge_batch = torch.cat(final_score_all_batch_AR, dim=-2)
            labels_ARs = data.lbl_idxs_ar
            labels_ARs= [tensor for tensor in labels_ARs if tensor.numel() > 0]
            labels_ARs_merge_batch = torch.cat(labels_ARs, dim=-1)


            idx_to_remove = (labels_ARs_merge_batch == 2).nonzero().squeeze(1)
            # 随机选择要删除的行
            number_meaningful = torch.sum(labels_ARs_merge_batch != 2).item()
            if number_meaningful == 0:
                number_meaningful = 1 # 防止全不训练报错了
            number_meaningless = torch.sum(labels_ARs_merge_batch == 2).item()
            num_remove = max(int(number_meaningless - number_meaningful * self.config.a_AR), 0)
            rows_to_remove = random.sample(idx_to_remove.tolist(), num_remove)
            # 删除选定的行
            logits_ARs_merge_batch_ = torch.stack([logit for i, logit in enumerate(logits_ARs_merge_batch) if i not in rows_to_remove], dim=0)
            labels_ARs_merge_batch_ = torch.stack([lbl for i, lbl in enumerate(labels_ARs_merge_batch) if i not in rows_to_remove], dim=0)
            labels_ARs_merge_batch_ += len(self.qrel_dic_ac)
            loss_AR = self.compute_loss(logits_ARs_merge_batch_, labels_ARs_merge_batch_)
        
        if loss_AC != -1 and loss_AR != -1:
            loss = loss_AC + loss_AR
        elif loss_AC != -1 and loss_AR == -1:
            loss = loss_AC
        else:
            raise ValueError
        
        return loss, final_score_all_batch_AC, final_score_all_batch_AR

    def ilp_all(self, score_0, score_1, query_emb):

        pred_emb = torch.cat((self.pred0_emb, self.pred1_emb), dim=0)

        query_emb_ = torch.mm(query_emb, self.Wquery).view(self.nclause, 1, -1)
        pred_emb_ = torch.mm(pred_emb, self.Wkey).view(pred_emb.size(0), self.nclause, -1)
        pred_emb_ = pred_emb_.permute(1, 2, 0)
        Wclause = torch.bmm(query_emb_, pred_emb_).squeeze(1)

        Wclause = self.sparsemax(Wclause)

        score = torch.cat((score_0, score_1), dim=2)
        Wclause = Wclause.unsqueeze(0).repeat([score.size(0), 1, 1])

        final_scores = torch.exp(torch.matmul(Wclause, torch.log((score + 1e-6).transpose(1, 2))))

        final_scores = torch.max(final_scores, dim=1)[0]

        return final_scores


    def ilp_hop_0_new(self, qemb, cemb):

        rel_scores = self.score_new(qemb, cemb)
        Wclause = torch.matmul(torch.mm(self.pred0_emb, self.Wpred_0), torch.mm(self.key_emb, self.Wkey_0).transpose(0,1)) # size (npred, nkey)
        Wclause = self.sparsemax(Wclause)
        Wclause = Wclause.unsqueeze(0).repeat([rel_scores.size(0), 1, 1])
        clause_scores = torch.exp(torch.matmul(Wclause, torch.log((rel_scores + 1e-6).transpose(1, 2))))
        clause_scores = clause_scores.transpose(1, 2) 
        
        return clause_scores

    
    def ilp_hop_1_new(self, qemb, cemb, m1emb, m1score):
        Wclause_l = torch.matmul(torch.mm(self.pred1_emb, self.Wpred_l1), torch.mm(self.key_emb, self.Wkey_l1).transpose(0,1)) # size (npred, nkey)
        Wclause_l = self.sparsemax(Wclause_l)
        new_query_emb = self.pred1_emb + torch.matmul(Wclause_l, torch.mm(self.key_emb, self.Wvalue_1))


        Wclause_r = torch.matmul(torch.mm(new_query_emb, self.Wpred_r1), torch.mm(self.key_emb, self.Wkey_r1).transpose(0,1)) # size (npred, nkey)
        Wclause_r = self.sparsemax(Wclause_r)

        rel1_scores = self.score_new(qemb, m1emb)
        rel2_scores = self.score_new(m1emb, cemb)
    
        Wclause_l = Wclause_l.unsqueeze(0).repeat([rel1_scores.size(0), 1, 1])
        Wclause_r = Wclause_r.unsqueeze(0).repeat([rel1_scores.size(0), 1, 1])

        clause_scores = (torch.exp(torch.matmul(Wclause_l, torch.log((rel1_scores + 1e-6).transpose(1, 2)))) * \
                            torch.exp(torch.matmul(Wclause_r, torch.log((rel2_scores + 1e-6).transpose(1, 2))))) ** 0.5
        clause_scores = torch.min(torch.ones(clause_scores.size()).to(self.device), clause_scores)
        clause_scores = clause_scores.view(clause_scores.size(0), clause_scores.size(1), -1, 1).contiguous()


        clause_scores = torch.max(clause_scores * m1score.transpose(1, 2).unsqueeze(-1), dim=2)[0]
        clause_scores = clause_scores.transpose(1, 2)


        return clause_scores

    def score_new(self, e1_emb, e2_emb):
        rel_score = self.Wrel(self.dropout(torch.tanh(torch.cat([e1_emb, e2_emb], dim=-1))))

        rel_score_final = self.softmax(rel_score)

        return rel_score_final


    def compute_loss(self, predictions, l):

        predictions_negative = 1.0 - predictions
        predictions_ = torch.stack([predictions_negative, predictions], dim=2)
        labels = torch.zeros((predictions.size(0), predictions.size(1)), dtype=torch.long, device=l.device)
        labels.scatter_(1, l.unsqueeze(1), 1)
        index = labels.unsqueeze(-1)
        loss_flat = -torch.log(torch.gather(predictions_.contiguous(), dim=2, \
                                            index=index)).squeeze(-1)
        loss_flat[torch.isnan(loss_flat)] = 0

        loss = loss_flat.sum() / loss_flat.size(1)

        return loss