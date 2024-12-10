from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from PIL import Image
import os
import torch


min_age = 21
max_age = 60
age_interval = 1

NUM_CLASSES = int((max_age - min_age) / age_interval + 1)

class UTKFaceDataset(Dataset):

    def __init__(self, csv_path, img_dir, transform=None,copies=1):
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['file'].values
        self.y = df['age'].values
        self.transform = transform
        self.copies = copies

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                        self.img_names[index]))

        if self.transform is not None:
            if self.copies >1:
                imgs=[]
                for i in range(self.copies):
                    imgs.append(self.transform(img))
                img = torch.stack(imgs)
            else:
                img = self.transform(img)

        label = self.y[index]//3
        levels = [1]*label + [0]*(NUM_CLASSES - 1 - label)
        levels = torch.tensor(levels, dtype=torch.float32)
        sample = {'image': img, 'classification_label': label, 'age': self.y[index]+min_age}
        return sample

    def __len__(self):
        return self.y.shape[0]