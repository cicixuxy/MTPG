import torch
import pandas as pd
from torch.utils.data import DataLoader
from itertools import product
from dataset import BAIDDatasetDF
from longclip import longclip
from MTPG import MTPG
from utils import compute_val_metrics


levels = ['bad', 'poor', 'fair', 'good', 'perfect']

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


aes_prompt = torch.cat(
    [longclip.tokenize(f"An image with {a} aesthetics.")
     for a in product(levels)
    ]
).to(device)

class AIAA(object):

    def __init__(self, data_loader, model):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.test_loader = data_loader

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

                    gts = (sample["aes_mean"]/10).to(self.device) 
                    true_score = true_score + gts.data.cpu().numpy().tolist()

                    logits_aesthetic = self.model(x,info, aes_prompt)
                    preds = 0.1 * logits_aesthetic[:, 0] + 0.3 * logits_aesthetic[:, 1] + 0.5 * logits_aesthetic[:,
                                                                                          2] + 0.7 * logits_aesthetic[:,
                                                                                                   3] + 0.9 * logits_aesthetic[
                                                                                                            :,
                                                                                                            4]
                    pred_score = pred_score + preds.cpu().tolist()

            plcc, srcc, acc = compute_val_metrics(pred_score, true_score)

            return plcc, srcc, acc
    def _init_weights(self):

        self.model.load_state_dict(torch.load('checkpoints/model_save.pt'),
                                   strict=False)

if __name__ == "__main__":
    model = MTPG(input_dim=768)

    BAID_test_csv = "dataset/test_descriptions.csv"

    df_test = pd.read_csv(BAID_test_csv)

    BAID_test_ds = BAIDDatasetDF(df_test, if_train=False, aug_num=1)

    BAID_test_loader = DataLoader(BAID_test_ds, batch_size=16, num_workers=16, shuffle=False)
    aiaa_baid = AIAA( BAID_test_loader, model)
    aiaa_baid.test()
