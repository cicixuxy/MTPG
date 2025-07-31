import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  # 计算平均损失


def compute_val_metrics(pred_score, true_score):
    lcc_mean = pearsonr(pred_score, true_score)  # PLCC
    srcc_mean = spearmanr(pred_score, true_score)  # SRCC

    true_score = np.array(true_score)
    true_score_lable = np.where(true_score <= 5, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_lable = np.where(pred_score <= 5, 0, 1)
    acc = accuracy_score(true_score_lable, pred_score_lable)

    print('acc: {:.3}  srcc_mean: {:.3}  lcc_mean: {:.3}'.format(acc, srcc_mean[0], lcc_mean[0]))

    return lcc_mean[0], srcc_mean[0], acc

