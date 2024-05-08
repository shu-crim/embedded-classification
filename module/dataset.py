from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from sklearn import preprocessing
from PIL import Image
import os
import pandas as pd


class MyDataSet(Dataset):
    def __init__(self, image_paths, label_map_dict=None, random_rotate=False):
        
        self.train_df = pd.DataFrame()
        self.images = []
        self.labels = []
        self.le = preprocessing.LabelEncoder()
        self.label_map_dict = label_map_dict
        self.random_rotate = random_rotate

        for path in image_paths:
            label = int(os.path.basename(path).replace(".png", "").split("_")[-1])
            self.images.append(path)
            self.labels.append(self.trans_label(label))

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
            rotation_times = np.random.randint(0, 4)

            # 画像を回転
            image = np.rot90(image, rotation_times)

            # 反転するか決定
            if np.random.choice([0, 1]) > 0:
                image = np.fliplr(image)

        return self.transform(Image.fromarray(image)), int(label)
    
    
    def trans_label(self, label:int) -> int:
        if type(self.label_map_dict) is type({}):
            if label in self.label_map_dict:
                return self.label_map_dict[label]
            else:
                return 0
        else:
            return label

