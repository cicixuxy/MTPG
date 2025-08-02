import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

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

