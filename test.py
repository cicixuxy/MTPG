import torch.nn as nn
import torchvision.models as models
import pdb
import torch
import logging
import clip
import torch.nn.functional as F
import torch.optim as optim
import time
from tqdm import tqdm
import os
import numpy as np
import argparse
import pandas as pd
import random
import copy

from torch.utils.data import DataLoader
from itertools import product


from dataset import BAIDDatasetDF
from longclip import longclip
from MTPG import MTPG

from utils import compute_val_metrics

seed = 20240716
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
levels = ['bad', 'poor', 'fair', 'good', 'perfect']

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


aes_prompt = torch.cat(
    [longclip.tokenize(f"A photo with {a} aesthetics.")
     for a in product(levels)
    ]
).to(device)

class AIAA(object):

    def __init__(self, data_loader, model):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.init_lr = 5e-6
        self.epochs = 5

        self.train_loader, self.test_loader = data_loader

        model.to(self.device)
        self.model = model
        self._init_weights()
    def test(self):
            true_score = []

            pred_score = []
            self.model.eval()

            with torch.no_grad():
                for _, sample in enumerate(self.test_loader):

                    x = sample["inputs"].to(self.device)
                    infos = sample["info"]
                    info = torch.cat([longclip.tokenize(info) for info in infos]).to(self.device)

                    gts = sample["aes_mean"].to(self.device)  # tensor(bs, )
                    true_score = true_score + gts.data.cpu().numpy().tolist()

                    logits_aesthetic = self.model(x,info, aes_prompt)
                    preds = 1 * logits_aesthetic[:, 0] + 3 * logits_aesthetic[:, 1] + 5 * logits_aesthetic[:,
                                                                                          2] + 7 * logits_aesthetic[:,
                                                                                                   3] + 9 * logits_aesthetic[
                                                                                                            :,
                                                                                                            4]
                    pred_score = pred_score + preds.cpu().tolist()

            plcc, srcc, acc = compute_val_metrics(pred_score, true_score)

            return plcc, srcc, acc
    def _init_weights(self):

        self.model.load_state_dict(torch.load('checkpoints/model_best.pt'),
                                   strict=False)

if __name__ == "__main__":
    model = MTPG(input_dim=768)
    BAID_train_csv = "/dataset/train_comments.csv"
    BAID_test_csv = "/dataset/test_comments.csv"
    df_train = pd.read_csv(BAID_train_csv)
    df_test = pd.read_csv(BAID_test_csv)

    BAID_train_ds = BAIDDatasetDF(df_train, if_train=True, aug_num=1)
    BAID_test_ds = BAIDDatasetDF(df_test, if_train=False, aug_num=1)

    BAID_train_loader = DataLoader(BAID_train_ds, batch_size=16, num_workers=16, shuffle=True)
    BAID_test_loader = DataLoader(BAID_test_ds, batch_size=16, num_workers=16, shuffle=False)
    aiaa_baid = AIAA((BAID_train_loader, BAID_test_loader), model)
    aiaa_baid.test()
