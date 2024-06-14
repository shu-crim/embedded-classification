import torch
import torchvision
import torchinfo
import numpy as np
import os
import sys
from PIL import Image
import datetime
import shutil
import csv
import json
from enum import Enum

import torch.nn as nn
import torch.optim as optim

import sklearn.metrics

from module.dataset import MyDataSet


def loadDataset(image_list_path, label_map_dict=None):
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

    train_dataset = MyDataSet(train_paths, random_rotate=True, label_map_dict=label_map_dict)
    val_dataset = MyDataSet(valid_paths, label_map_dict=label_map_dict)

    return train_dataset, val_dataset


def train(train_dataset, val_dataset, num_epoch, output_dir, num_class, dim_embedded=256):
    # 拡張されたクラス数
    expand_num_class = train_dataset.expand_class_num

    # モデル作成
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        torch.nn.Linear(model.fc.in_features, dim_embedded),
        torch.nn.Linear(dim_embedded, expand_num_class)
    )

    # モデル設計の出力
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


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    min_valid_loss = 1e10
    epoch_min_valid_loss = 0
    max_valid_acc = 0
    epoch_max_valid_acc = 0

    for epoch in range(num_epoch):
        # train
        total_loss = 0
        total_size = 0
        model.train()
        pred = []
        Y = []
        timestamp = datetime.datetime.now().timestamp()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_size += data.size(0)
            loss.backward()
            optimizer.step()

            pred += [int(l.argmax()) for l in output]
            Y += [int(l) for l in target]

            if batch_idx % 5 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), total_loss / total_size))
        print(f"学習時間: {datetime.datetime.now().timestamp() - timestamp:.3f} s")
        loss_train = total_loss / total_size
        accuracy_train = sklearn.metrics.accuracy_score(Y, pred)
        print("---------------- Train ----------------")
        print(f'Average Train loss: {total_loss / total_size:.6f}')
        print(sklearn.metrics.classification_report(Y, pred, zero_division=0))
        print(sklearn.metrics.confusion_matrix(Y, pred))

        # debug
        data = data.cpu().detach().numpy().copy()
        for i in range(min(1, data.shape[0])):
            Image.fromarray((data[i,:].transpose(1,2,0) * 255).astype(np.uint8)).save(
                os.path.join(output_dir, f"train_{epoch:06}_{i:02}.png"))
                
        # valid
        pred = []
        Y = []
        total_loss = 0
        total_size = 0
        model.eval()
        timestamp = datetime.datetime.now().timestamp()
        for i, (data, target) in enumerate(val_loader):
            with torch.no_grad():
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                total_size += data.size(0)
            pred += [int(l.argmax()) for l in output]
            Y += [int(l) for l in target]
        print(f"検証時間: {datetime.datetime.now().timestamp() - timestamp:.3f} s")
        loss_valid = total_loss / total_size
        accuracy_valid = sklearn.metrics.accuracy_score(Y, pred)
        print("---------------- Valid ----------------")
        print(f'Average Valid loss: {total_loss / total_size:.6f}')
        print(sklearn.metrics.classification_report(Y, pred, zero_division=0))
        print(sklearn.metrics.confusion_matrix(Y, pred))

        if loss_valid < min_valid_loss:
            min_valid_loss = loss_valid
            epoch_min_valid_loss = epoch

        if accuracy_valid > max_valid_acc:
            max_valid_acc = accuracy_valid
            epoch_max_valid_acc = epoch
            # モデルを保存する。
            torch.save(model.state_dict(), os.path.join(output_dir, "model_best_acc.pth"))

        # モデルを保存する。
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch:06}.pth"))

        if not os.path.exists(os.path.join(output_dir, "loss.csv")):
            with open(os.path.join(output_dir, "loss.csv"), "w") as f:
                f.write(f"epoch,train_loss,valid_loss,accuracy_train,accuracy_valid,min_valid_loss,epoch_min_valid_loss,max_valid_acc,epoch_max_valid_acc\n")
        with open(os.path.join(output_dir, "loss.csv"), "a") as f:
            f.write(f"{epoch},{loss_train},{loss_valid},{accuracy_train},{accuracy_valid},{min_valid_loss},{epoch_min_valid_loss},{max_valid_acc},{epoch_max_valid_acc}\n")


class DataEntryMethod(Enum):
    NotRandomSequential = 1, # データセットの順でそのまま登録
    Sequential = 2, # ランダムな順で登録
    ClassBreadthFirst = 3, # 各クラスを横断的な順で登録
    PrecisionRecallBottleneck = 4, # P/Rが最低値なクラスについてP/Rを上げるデータを動的に選択して登録(Recallを上げる際のデータ選択はランダム)

def readSetting(path = 'setting.json'):
    # 設定を読み込む
    with open(path, 'r', encoding='utf-8') as f:
        setting = json.load(f)

    setting["common"]["num_class"] = num_class = int(setting["common"]["num_class"])
    setting["common"]["dim_embedded"] = int(setting["common"]["dim_embedded"])
    setting["common"]["num_patch_per_object"] = int(setting["common"]["num_patch_per_object"])

    label_map_dict_raw = setting["common"]["label_map_dict"]
    setting["common"]["label_map_dict"] = {}
    for key, value in label_map_dict_raw.items():
        setting["common"]["label_map_dict"][int(key)] = value

    label_name_dict = setting["common"]["label_name_dict"]
    setting["common"]["label_name_dict"] = {}
    for key, value in label_name_dict.items():
        setting["common"]["label_name_dict"][int(key)] = value

    setting["valid"]["knn_k"] = int(setting["valid"]["knn_k"])
    setting["valid"]["random_seed"] = int(setting["valid"]["random_seed"])
    setting["valid"]["num_run_valid"] = int(setting["valid"]["num_run_valid"])
    setting["valid"]["round_run_valid"] = int(setting["valid"]["round_run_valid"])

    data_entry_method = setting["valid"]["data_entry_method"]
    if data_entry_method == "NotRandomSequential":
        setting["valid"]["data_entry_method"] = DataEntryMethod.NotRandomSequential
    elif data_entry_method == "Sequential":
        setting["valid"]["data_entry_method"] = DataEntryMethod.Sequential
    elif data_entry_method == "ClassBreadthFirst":
        setting["valid"]["data_entry_method"] = DataEntryMethod.ClassBreadthFirst
    elif data_entry_method == "PrecisionRecallBottleneck":
        setting["valid"]["data_entry_method"] = DataEntryMethod.PrecisionRecallBottleneck
    else:
        setting["valid"]["data_entry_method"] = DataEntryMethod.Sequential

    setting["combine"]["stats_start_row"] = int(setting["combine"]["stats_start_row"])
    setting["combine"]["stats_start_col"] = int(setting["combine"]["stats_start_col"])

    return setting


if __name__ == '__main__':
    SETTING_FILE_NAME = 'setting.json'
    setting = readSetting(SETTING_FILE_NAME)

    output_root = setting["common"]["output_root"]
    image_list_path = setting["common"]["image_list_path"]
    num_class = setting["common"]["num_class"]
    dim_embedded = setting["common"]["dim_embedded"]
    label_map_dict = setting["common"]["label_map_dict"]

    # データセット作成
    train_dataset, val_dataset = loadDataset(image_list_path, label_map_dict)
    print(f"train_dataset: {len(train_dataset)}")
    print(f"val_dataset: {len(val_dataset)}")

    # outputディレクトリ作成
    now = datetime.datetime.now()
    output_dir = os.path.join(output_root, now.strftime('%Y%m%d_%H%M%S_train') + f"_{os.path.basename(image_list_path).split('.')[0]}")
    os.makedirs(output_dir)
    shutil.copy(os.path.abspath(__file__), output_dir)
    shutil.copy(image_list_path, output_dir)
    shutil.copy(SETTING_FILE_NAME, output_dir)

    # train実行
    train(train_dataset, val_dataset, num_epoch=50, output_dir=output_dir, num_class=num_class, dim_embedded=dim_embedded)
