import torch
import torchvision
import torchinfo
import torch.nn as nn
import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
import datetime
import shutil
import csv
from module.dataset import MyDataSet
import cv2
from collections import Counter
import sklearn.metrics
import random
import json
from enum import Enum


def AnalyzeResNetModel(output_root, image_list_path, model_path, num_class, num_patch_per_object, label_map_dict=None):
    # output
    now = datetime.datetime.now()
    output_dir = os.path.join(output_root, now.strftime('%Y%m%d_%H%M%S_analyze_' + os.path.basename(image_list_path).replace(".csv", "")))
    os.makedirs(output_dir)
    shutil.copy(os.path.abspath(__file__), output_dir)

    # モデル読み込み
    model = torchvision.models.resnet50()
    model.fc = nn.Sequential(
        torch.nn.Linear(model.fc.in_features, dim_embedded),
        torch.nn.Linear(dim_embedded, num_class)
    )
    model.load_state_dict(torch.load(model_path))

    # モデル情報の保存
    with open(os.path.join(output_dir, "network.txt"), "w") as f:
        original_stdout = sys.stdout # 標準出力の現在の状態を保存
        sys.stdout = f # 標準出力をファイルにリダイレクト
        print(model)
        torchinfo.summary(
                model,
                input_size=(1, 3, 224, 224),
                col_names=["output_size", "num_params"],
            )
        sys.stdout = original_stdout # 標準出力を元に戻す

    # 画像リストを読み込む
    train_paths = []
    valid_paths = []
    with open(image_list_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == "train":
                train_paths.append(os.path.join(os.path.dirname(image_list_path), row[1]))
            elif row[0] == "valid":
                valid_paths.append(os.path.join(os.path.dirname(image_list_path), row[1]))

    # debug:高速化のため間引き
    # train_paths = train_paths[::num_patch_per_object * 10]
    # valid_paths = valid_paths[::num_patch_per_object * 10]

    train_dataset = MyDataSet(train_paths, random_rotate=True, label_map_dict=label_map_dict)
    valid_dataset = MyDataSet(valid_paths, label_map_dict=label_map_dict)
    print(f"train_dataset: {len(train_dataset)}")
    print(f"val_dataset: {len(valid_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

    # 識別結果を得る
    device = torch.device('cuda')
    model.cuda()

    def estimate(model, data_loader):
        model.eval()

        estimation = []
        label = []
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            output = model(data).cpu().detach().numpy().copy()[0]
            estimation.append(np.argmax(output))
            label.append(int(target))

        return estimation, label

    train_estimation, train_label = estimate(model, train_loader)
    valid_estimation, valid_label = estimate(model, valid_loader)
    
    print("---------------- Train ----------------")
    print(sklearn.metrics.classification_report(train_label, train_estimation, zero_division=0))
    cmat_train = sklearn.metrics.confusion_matrix(train_label, train_estimation, labels=[i for i in range(num_class)])
    print(cmat_train)

    print("---------------- Valid ----------------")
    print(sklearn.metrics.classification_report(valid_label, valid_estimation, zero_division=0))
    cmat_valid = sklearn.metrics.confusion_matrix(valid_label, valid_estimation, labels=[i for i in range(num_class)])
    print(cmat_valid)

    def confusion_matrix_csv(confusion_matrix):
        num_estimation = [0 for i in range(num_class)]
        true_positive = [0 for i in range(num_class)]
        ret = ""
        ret += ","
        for i in range(num_class):
            ret += f"{i},"
        ret += "num_gt,TP,Recall\n"
        for row in range(num_class):
            num_gt = 0
            ret += f"{row},"
            for col in range(num_class):
                num_gt += confusion_matrix[row, col]
                num_estimation[col] += confusion_matrix[row, col]
                if row == col:
                    true_positive[row] = confusion_matrix[row, col]
                ret += f"{confusion_matrix[row, col]},"
            ret += f"{num_gt},{true_positive[row]},"
            if num_gt > 0:
                ret += f"{(true_positive[row] / num_gt):#.3g}\n"
            else:
                ret += "----\n"
        ret += "num_estimation,"
        for col in range(num_class):
            ret += f"{num_estimation[col]},"
        ret += "\nTP,"
        for col in range(num_class):
            ret += f"{true_positive[col]},"
        ret += "\nPrecision,"
        for col in range(num_class):
            if num_estimation[col] > 0:
                ret += f"{(true_positive[col] / num_estimation[col]):#.3g},"
            else:
                ret += "----,"
        ret += "\n"
        return ret

    # confusion matrixをcsvに書き出し
    with open(os.path.join(output_dir, "evaluate.csv"), "w", encoding="utf-8") as f:
        f.write("Train,\n")
        f.write(confusion_matrix_csv(cmat_train))

        f.write("\nValid,\n")
        f.write(confusion_matrix_csv(cmat_valid))

    # GTと識別結果をファイル名に付与して検証画像をコピー
    os.makedirs(os.path.join(output_dir, "evaluate_images"), exist_ok=True)
    for index in range(len(valid_paths)):
        new_filename = f"{valid_label[index]:02}_to_{valid_estimation[index]:02}_{os.path.basename(valid_paths[index])}"
        shutil.copy2(valid_paths[index], os.path.join(output_dir, "evaluate_images", new_filename))
        

if __name__ == '__main__':
    # 設定を読み込む
    with open('setting.json', 'r', encoding="utf-8") as f:
        setting = json.load(f)

    output_root = setting["common"]["output_root"]
    image_list_path = setting["common"]["image_list_path"]
    num_class = int(setting["common"]["num_class"])
    dim_embedded = int(setting["common"]["dim_embedded"])
    num_patch_per_object = int(setting["common"]["num_patch_per_object"])
    label_map_dict_raw = setting["common"]["label_map_dict"]
    label_map_dict = {}
    for key, value in label_map_dict_raw.items():
        label_map_dict[int(key)] = value

    model_path = setting["valid"]["model_path"]
    use_calculated_embedded = setting["valid"]["use_calculated_embedded"]
    dir_calculated_embedded = setting["valid"]["dir_calculated_embedded"]
    knn_k = int(setting["valid"]["knn_k"])

    AnalyzeResNetModel(output_root, image_list_path, model_path, num_class, num_patch_per_object, label_map_dict)
