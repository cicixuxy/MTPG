import os
import random
import torch
from torch.utils.data import Dataset
from PIL import ImageFile
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import warnings
from img_loader import PilCloudLoader,PilCloudLoader_pre

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BAIDDatasetDF(Dataset):

    def __init__(self, df, if_train, aug_num=1, size=(224, 224)):

        self.if_train = if_train
        self.aug_num = aug_num
        self.df = df

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.img_loader = PilCloudLoader(handle_exceptions=False, size=size, aug_num = aug_num)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):

        row = self.df.iloc[item]    
        image_path = os.path.join('/media/boot/BAID/images', os.path.join(row["image"]))

        inputs = self.img_loader(image_path, self.if_train)

        aes_mean = row["score"]

        info =str(row['comment'])


        if self.if_train:
            sample = {
                'aes_mean': aes_mean,

                'info': info,

            }
            for idx in range(self.aug_num):
                sample["inputs" + str(idx)] = inputs[idx]
        else:
            sample = {
                'path':image_path,
                'inputs': inputs,

                'aes_mean': aes_mean,

                'info': info,
            }
        return sample



