from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from sklearn import preprocessing
from PIL import Image
import os
import pandas as pd
import random


class MyDataSet(Dataset):
    def __init__(self, image_paths, label_map_dict={}, random_rotate=False, unsupervised_label_start=256, exclude_unsupervised_label=True):
        
        self.train_df = pd.DataFrame()
        self.images = []
        self.labels = []
        self.le = preprocessing.LabelEncoder()
        self.label_map_dict = label_map_dict
        self.random_rotate = random_rotate
        self.unsupervised_label_start = unsupervised_label_start
        self.max_label = 0
        for key, value in label_map_dict.items():
            self.max_label = max(self.max_label, value)
        self.expand_class_num = 0

        for path in image_paths:
            label = int(os.path.basename(path).replace(".png", "").split("_")[-1])
            label_trans = self.trans_label(label)
            self.expand_class_num = max(self.expand_class_num, label_trans + 1)

            if exclude_unsupervised_label and label >= unsupervised_label_start:
                continue

            self.images.append(path)
            self.labels.append(label_trans)

        self.le.fit(self.labels)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]))
        label = self.labels[idx]

        if self.random_rotate:
            # ランダムに回転する回数を決定（0, 1, 2, 3のいずれか）
            # 0: 0度 (回転なし), 1: 90度, 2: 180度, 3: 270度
            # rotation_times = np.random.randint(0, 4)
            rotation_times = random.randint(0, 4)

            # 画像を回転
            image = np.rot90(image, rotation_times)

            # 反転するか決定
            # if np.random.choice([0, 1]) > 0:
            if random.choice([0, 1]) > 0:
                image = np.fliplr(image)

        return self.transform(Image.fromarray(image)), int(label)
    
    
    def trans_label(self, label:int) -> int:
        if label >= self.unsupervised_label_start:
            return self.max_label + 1 + (label - self.unsupervised_label_start)
        
        if type(self.label_map_dict) is type({}):
            if label in self.label_map_dict:
                return self.label_map_dict[label]
            else:
                return 0
        else:
            return label

