import traceback
import logging
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)  

def pil_transform(if_train):
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406] 
    IMAGE_NET_STD = [0.229, 0.224, 0.225] 

    # 归一化
    normalize = transforms.Normalize(
        mean=IMAGE_NET_MEAN,
        std=IMAGE_NET_STD)

    if if_train:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),  
            transforms.RandomCrop((224, 224)), 
            transforms.ToTensor(),
            normalize])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
    return transform

def pil_transform_pre(if_train):
    IMAGE_NET_MEAN = [0.485, 0.456, 0.406]  
    IMAGE_NET_STD = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(
        mean=IMAGE_NET_MEAN,
        std=IMAGE_NET_STD)

    if if_train:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            normalize])

    return transform
class PilCloudLoader(object):
    def __init__(self, handle_exceptions=False, size=(224, 224), aug_num=1):
        self.handle_exceptions = handle_exceptions  
        self.aug_num = aug_num
        self.size = size

    def __call__(self, path, if_train):
        # transform模块
        transform = pil_transform(if_train=if_train)

        if if_train:
            imgs = []
            for _ in range(self.aug_num):
                try:
                    img = Image.open(path).convert('RGB')
                except OSError as e:
                    if self.handle_exceptions:
                        img = Image.new('RGB', self.size, 0)
                        logger.error(f"{path}: {traceback.format_exc()}")
                    else:
                        raise e

                img = transform(img) 
                imgs.append(img)  
            return imgs
        else:
            try:
                img = Image.open(path).convert('RGB')
            except OSError as e:
                if self.handle_exceptions:
                    img = Image.new('RGB', self.size, 0)
                    logger.error(f"{path}: {traceback.format_exc()}")
                else:
                    raise e

            img = transform(img)
            return img

class PilCloudLoader_pre(object):
    def __init__(self, handle_exceptions=False, size=(224, 224),aug_num=1):
        self.handle_exceptions = handle_exceptions  
        self.aug_num = aug_num
        self.size = size

    def __call__(self, path, if_train):
        # transform模块
        transform = pil_transform_pre(if_train=if_train)

        if if_train:
            imgs = []
            for _ in range(self.aug_num):
                try:
                    img = Image.open(path).convert('RGB')
                except OSError as e:
                    if self.handle_exceptions:
                        img = Image.new('RGB', self.size, 0)
                        logger.error(f"{path}: {traceback.format_exc()}")
                    else:
                        raise e

                img = transform(img)
                imgs.append(img)
            return imgs
        else:
            try:
                img = Image.open(path).convert('RGB')
            except OSError as e:
                if self.handle_exceptions:
                    img = Image.new('RGB', self.size, 0)
                    logger.error(f"{path}: {traceback.format_exc()}")
                else:
                    raise e

            img = transform(img)
            return img

