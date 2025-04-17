from tensorboardX import SummaryWriter
import torch

class Summarizer(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)


def args_metric(true_args_list, pred_args_list):
    tp, tn, fp, fn = 0, 0, 0, 0
    for true_args, pred_args in zip(true_args_list, pred_args_list):
        true_args_set = set(true_args)
        pred_args_set = set(pred_args)
        assert len(true_args_set) == len(true_args)
        assert len(pred_args_set) == len(pred_args)
        tp += len(true_args_set & pred_args_set)
        fp += len(pred_args_set - true_args_set)
        fn += len(true_args_set - pred_args_set)
    if tp + fp == 0:
        pre = tp/(tp + fp + 1e-10)
    else:
        pre = tp/(tp + fp)
    if tp + fn == 0:
        rec = tp/(tp + fn + 1e-10)
    else:
        rec = tp/(tp + fn)
    if pre == 0. and rec == 0.:
        f1 = (2 * pre * rec)/(pre + rec + 1e-10)
    else:
        f1 = (2 * pre * rec)/(pre + rec)
    acc = (tp + tn)/(tp + tn + fp + fn + 1e-10)
    return {'pre': pre, 'rec': rec, 'f1': f1, 'acc': acc}

class Scorer:
    def __init__(self):
        self.s = 0
        self.g = 0
        self.c = 0
        return

    def add(self, predict, gold):
        self.s += len(predict)
        self.g += len(gold)
        self.c += len(gold & predict)
        return

    @property
    def p(self):
        return self.c / self.s if self.s else 0.

    @property
    def r(self):
        return self.c / self.g if self.g else 0.

    @property
    def f(self):
        p = self.p
        r = self.r
        return (2. * p * r) / (p + r) if p + r > 0 else 0.0

    def dump(self):
        return {
            'g': self.g,
            's': self.s,
            'c': self.c,
            'p': self.p,
            'r': self.r,
            'f': self.f
        }
        

def eval_edge_cdcp(predict_list, gold_list):
    # Obtain edge labels
    edge_labels = set()
    for g_sample in gold_list:
        labels = [e[4] for e in g_sample]
        edge_labels |= set(labels)
    assert len(edge_labels) == 2, print(edge_labels)
    # Calculate label scores
    label_scores = dict()
    label2name = {0: "reason" , 1: "evidence"}
    for label in edge_labels:
        scorer = Scorer()
        for p_sample, g_sample in zip(predict_list, gold_list):
            scorer.add(
                predict=set([
                    (
                        edge[0],
                        edge[1],
                        edge[2],
                        edge[3]
                    )
                    for edge in p_sample if edge[4] == label
                ]),
                gold=set([
                    (
                        edge[0],
                        edge[1],
                        edge[2],
                        edge[3]
                    )
                    for edge in g_sample if edge[4] == label
                ]),
            )
        label_scores[label2name[label]] = scorer.dump()

    return label_scores

def eval_component_cdcp(predict_list, gold_list):
    # Obtain edge labels
    edge_labels = set()
    for g_sample in gold_list:
        labels = [e[2] for e in g_sample]
        edge_labels |= set(labels)
    # assert len(edge_labels) == 5, print(edge_labels)
    # Calculate label scores
    label_scores = dict()
    label2name = {0: "value" , 1: "policy", 2: "testimony", 3: "fact", 4: "reference"}
    for label in edge_labels:
        scorer = Scorer()
        for p_sample, g_sample in zip(predict_list, gold_list):
            scorer.add(
                predict=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in p_sample if edge[2] == label
                ]),
                gold=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in g_sample if edge[2] == label
                ]),
            )
        label_scores[label2name[label]] = scorer.dump()

    return label_scores

def eval_edge_PE(predict_list, gold_list):
    # Obtain edge labels
    edge_labels = set()
    for g_sample in gold_list:
        labels = [e[4] for e in g_sample]
        edge_labels |= set(labels)
    assert len(edge_labels) == 2, print(edge_labels)
    # Calculate label scores
    label_scores = dict()
    label2name = {0: "Support" , 1: "Attack"}
    for label in edge_labels:
        scorer = Scorer()
        for p_sample, g_sample in zip(predict_list, gold_list):
            scorer.add(
                predict=set([
                    (
                        edge[0],
                        edge[1],
                        edge[2],
                        edge[3]
                    )
                    for edge in p_sample if edge[4] == label
                ]),
                gold=set([
                    (
                        edge[0],
                        edge[1],
                        edge[2],
                        edge[3]
                    )
                    for edge in g_sample if edge[4] == label
                ]),
            )
        label_scores[label2name[label]] = scorer.dump()

    return label_scores

def eval_component_PE(predict_list, gold_list):
    # Obtain edge labels
    edge_labels = set()
    for g_sample in gold_list:
        labels = [e[2] for e in g_sample]
        edge_labels |= set(labels)
    # assert len(edge_labels) == 3, print(edge_labels)
    # Calculate label scores
    label_scores = dict()
    label2name = {0: "Premise" , 1: "Claim", 2: "MajorClaim"}
    for label in edge_labels:
        scorer = Scorer()
        for p_sample, g_sample in zip(predict_list, gold_list):
            scorer.add(
                predict=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in p_sample if edge[2] == label
                ]),
                gold=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in g_sample if edge[2] == label
                ]),
            )
        label_scores[label2name[label]] = scorer.dump()

    return label_scores

    # Obtain edge labels
    edge_labels = set()
    for g_sample in gold_list:
        labels = [e[4] for e in g_sample]
        edge_labels |= set(labels)
    # assert len(edge_labels) == 4, print(edge_labels)
    # Calculate label scores
    label_scores = dict()
    label2name = {0: "support" , 1: "undercut", 3: "rebut", 4: "example"}
    for label in edge_labels:
        scorer = Scorer()
        for p_sample, g_sample in zip(predict_list, gold_list):
            scorer.add(
                predict=set([
                    (
                        edge[0],
                        edge[1],
                        edge[2],
                        edge[3]
                    )
                    for edge in p_sample if edge[4] == label
                ]),
                gold=set([
                    (
                        edge[0],
                        edge[1],
                        edge[2],
                        edge[3]
                    )
                    for edge in g_sample if edge[4] == label
                ]),
            )
        label_scores[label2name[label]] = scorer.dump()

    return label_scores

    # Obtain edge labels
    edge_labels = set()
    for g_sample in gold_list:
        labels = [e[2] for e in g_sample]
        edge_labels |= set(labels)
    assert len(edge_labels) == 2, print(edge_labels)
    # Calculate label scores
    label_scores = dict()
    label2name = {0: "opponent" , 1: "proponent"}
    for label in edge_labels:
        scorer = Scorer()
        for p_sample, g_sample in zip(predict_list, gold_list):
            scorer.add(
                predict=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in p_sample if edge[2] == label
                ]),
                gold=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in g_sample if edge[2] == label
                ]),
            )
        label_scores[label2name[label]] = scorer.dump()

    return label_scores



    # Obtain edge labels
    edge_labels = set()
    for g_sample in gold_list:
        labels = [e[2] for e in g_sample]
        edge_labels |= set(labels)
    assert len(edge_labels) == 3, print(edge_labels)
    # Calculate label scores
    label_scores = dict()
    label2name = {0: "Evidence" , 1: "Claim", 2: "MajorClaim"}
    for label in edge_labels:
        scorer = Scorer()
        for p_sample, g_sample in zip(predict_list, gold_list):
            scorer.add(
                predict=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in p_sample if edge[2] == label
                ]),
                gold=set([
                    (
                        edge[0],
                        edge[1]
                    )
                    for edge in g_sample if edge[2] == label
                ]),
            )
        label_scores[label2name[label]] = scorer.dump()

    return label_scores