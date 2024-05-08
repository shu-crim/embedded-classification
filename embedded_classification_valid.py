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


class DataEntryMethod(Enum):
    NotRandomSequential = 1, # データセットの順でそのまま登録
    Sequential = 2, # ランダムな順で登録
    ClassBreadthFirst = 3, # 各クラスを横断的な順で登録
    PrecisionRecallBottleneck = 4, # P/Rが最低値なクラスについてP/Rを上げるデータを動的に選択して登録(Recallを上げる際のデータ選択はランダム)


def embeddedCsv(output_root, image_list_path, model_path, num_class, dim_embedded, label_map_dict=None):
    # output
    now = datetime.datetime.now()
    output_dir = os.path.join(output_root, now.strftime('%Y%m%d_%H%M%S_valid_' + os.path.basename(image_list_path).replace(".csv", "")))
    os.makedirs(output_dir)
    shutil.copy(os.path.abspath(__file__), output_dir)

    # モデル読み込み
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(
        torch.nn.Linear(model.fc.in_features, dim_embedded),
        torch.nn.Linear(dim_embedded, num_class)
    )
    model.load_state_dict(torch.load(model_path))

    # 最終層の取り外し
    model.fc[1] = torch.nn.Identity()

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

    train_dataset = MyDataSet(train_paths, random_rotate=True, label_map_dict=label_map_dict)
    valid_dataset = MyDataSet(valid_paths, label_map_dict=label_map_dict)
    print(f"train_dataset: {len(train_dataset)}")
    print(f"val_dataset: {len(valid_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

    # 埋め込みベクトルを取得する
    device = torch.device('cuda')
    model.cuda()

    def calcEmbedded(model, data_loader):
        model.eval()

        embedded = []
        label = []
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            embedded.append(model(data).cpu().detach().numpy().copy()[0])
            label.append(int(target))

        return embedded, label


    train_embedded, train_label = calcEmbedded(model, train_loader)
    valid_embedded, valid_label = calcEmbedded(model, valid_loader)

    def writeEmbedded(output_path, label, embedded):
        with open(output_path, "w") as f:
            f.write("label,")
            for iDim in range(dim_embedded):
                f.write(f"x_{iDim},")
            f.write("\n")

            for iData in range(len(label)):
                f.write(f"{label[iData]},")
                for iEmbedded in range(dim_embedded):
                    f.write(f"{embedded[iData][iEmbedded]},")
                f.write("\n")
            
    writeEmbedded(os.path.join(output_dir, "train_embedded.csv"), train_label, train_embedded)
    writeEmbedded(os.path.join(output_dir, "valid_embedded.csv"), valid_label, valid_embedded)

    return output_dir


def readEmbedded(csv_path, ):
    labels = []
    embeddeds = []
    with open(csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            labels.append(int(row[0]))
            embedded = []
            for i in range(1, dim_embedded + 1):
                embedded.append(np.float32(row[i]))
            embeddeds.append(embedded)

    return np.array(labels), np.array(embeddeds)


def majorityVote(int_list):
    """
    整数のリストから多数決を求める関数

    Args:
        int_list (list): 整数のリスト

    Returns:
        int: 多数決の値 (もっとも頻繁に出現する整数)
    """
    # 整数の出現回数をカウント
    count_dict = Counter(int_list)
    
    # 最頻値を求める
    most_common_value = count_dict.most_common(1)[0][0]
    
    return most_common_value


def remove_value_from_list(lst, value_to_remove):
    """
    Removes all occurrences of a specific value from a list.

    Args:
        lst (list): The input list.
        value_to_remove: The value to be removed from the list.

    Returns:
        list: A new list with the specified value removed.
    """
    return [item for item in lst if item != value_to_remove]


def caclAccuracy(label_gt, label_estimation):
    accuracy = sklearn.metrics.accuracy_score(label_gt, label_estimation)
    precision = sklearn.metrics.precision_score(label_gt, label_estimation, average=None, labels=[i for i in range(num_class)], zero_division=np.nan)
    recall = sklearn.metrics.recall_score(label_gt, label_estimation, average=None, labels=[i for i in range(num_class)], zero_division=np.nan)
    confusion_matrix = sklearn.metrics.confusion_matrix(label_gt, label_estimation, labels=[i for i in range(num_class)])

    return accuracy, precision, recall, confusion_matrix


def evaluate(valid_dir, num_patch_per_object:int, K:int=1, data_entry_method:DataEntryMethod=DataEntryMethod.Sequential, random_seed:int=-1):
    # embeddedの読み込み
    label_train, embedded_train = readEmbedded(os.path.join(valid_dir, "train_embedded.csv"))
    label_valid, embedded_valid = readEmbedded(os.path.join(valid_dir, "valid_embedded.csv"))
    num_patch = label_valid.shape[0]
    num_object = num_patch // num_patch_per_object

    # validパッチをオブジェクト単位に分ける
    valid_object_patch = []
    valid_object_label = []
    patches = []
    for iValid in range(len(label_valid)):
        patches.append(embedded_valid[iValid])
        if (iValid + 1) % num_patch_per_object == 0:
            valid_object_label.append(label_valid[iValid])
            valid_object_patch.append(patches)
            patches = []

    # 出力ファイルの準備
    result_csv_path = os.path.join(valid_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S_valid.csv'))
    with open(result_csv_path, "w") as f:
        f.write(f"num_patch,{num_patch}\n")
        f.write(f"num_object,{num_object}\n")
        f.write("\n")
        f.write(f"num_add,addition_index,addition_class,accuracy_patch,accuracy_object,")
        for iClass in range(num_class):
            f.write(f"class{iClass}_precision,class{iClass}_recall,")
        f.write("\n")

    # データを追加しながらkNNで識別
    addition_indices = [i for i in range(num_object)]

    # データの追加順をシャッフル
    if data_entry_method != DataEntryMethod.NotRandomSequential:
        if random_seed < 0:
            random_seed = random.randint(0, 10000000)
        random.seed(random_seed)
        random.shuffle(addition_indices)

    # 各クラスごとのセットを作成
    each_class_list = [[] for i in range(num_class)]
    for index in addition_indices:
        each_class_list[valid_object_label[index]].append(index)

    if data_entry_method == DataEntryMethod.ClassBreadthFirst:
        # 各クラスを1つずつ順に登録
        addition_indices = []
        while True:
            exist = False
            for iClass in range(num_class):
                if len(each_class_list[iClass]) > 0:
                    addition_indices.append(each_class_list[iClass].pop())
                    exist = True
            if not exist:
                break

    # 評価と登録のループ
    for iValidObject in range(num_object + 1):
        # Trainデータの登録
        knn = cv2.ml.KNearest_create()
        knn.train(embedded_train, cv2.ml.ROW_SAMPLE, label_train)

        # Validデータの識別
        ret, results, neighbours, dist = knn.findNearest(embedded_valid, K)
        label_estimation = results.reshape(label_valid.shape[0]).astype(int)

        # 識別対象のデータ(のインデックス)ごとに、どのGTクラスがどのクラスに識別されたかリスト化
        if data_entry_method == DataEntryMethod.PrecisionRecallBottleneck:
            confusion_matrix_valid_index = [[[] for i in range(num_class)] for j in range(num_class)]
            confusion_matrix_debug = [[0 for i in range(num_class)] for j in range(num_class)]
            for i in range(len(results)):
                confusion_matrix_valid_index[label_valid[i]][label_estimation[i]].append(i // num_patch_per_object) # 追加するindexはオブジェクトindexへと変換
                confusion_matrix_debug[label_valid[i]][label_estimation[i]] += 1

        # 1データごとの詳細情報
        # now = datetime.datetime.now()
        # with open(os.path.join(valid_dir, now.strftime('%Y%m%d_%H%M%S_valid_detail.csv')), "w") as f:
        #     f.write("gt_label,est_label,")
        #     for k in range(K):
        #         f.write(f"neighbor_{k+1},")
        #     for k in range(K):
        #         f.write(f"distance_{k+1},")
        #     f.write("\n")
        #     for iValid in range(len(label_valid)):
        #         f.write(f"{label_valid[iValid]},{label_estimation[iValid]},")
        #         for k in range(K):
        #             f.write(f"{neighbours[iValid][k]},")
        #         for k in range(K):
        #             f.write(f"{dist[iValid][k]},")
        #         f.write("\n")

        # each patch
        accuracy_patch = np.sum(label_estimation == label_valid) / num_patch

        # each object
        estimation_label_per_object = []
        vote = []
        for iValid in range(len(label_valid)):
            vote.append(label_estimation[iValid])
            if (iValid + 1) % num_patch_per_object == 0:
                estimation_label_per_object.append(majorityVote(vote))
                vote = []
        accuracy_object, precision_object, recall_object, confusion_matrix_object = caclAccuracy(valid_object_label, estimation_label_per_object)

        # ファイル出力
        with open(result_csv_path, "a") as f:
            f.write(f"{iValidObject},{'-' if iValidObject==0 else addition_indices[iValidObject-1]},{'-' if iValidObject==0 else valid_object_label[addition_indices[iValidObject-1]]},{accuracy_patch},{accuracy_object},")
            for iClass in range(num_class):
                f.write(f"{precision_object[iClass]},{recall_object[iClass]},")
            f.write("\n")

        # trainデータの追加
        if iValidObject < num_object:
            # リストから順に選択
            addition_index = addition_indices[iValidObject]

            # P/Rのボトルネックへ対処するデータを登録
            if data_entry_method == DataEntryMethod.PrecisionRecallBottleneck:
                # Recallが最も低いクラスのデータを選択
                recall_object[np.isnan(recall_object)] = 2. # nanのクラスは無視
                while True:
                    min_recall_class = np.argmin(recall_object)
                    if len(each_class_list[min_recall_class]) == 0:
                        recall_object[min_recall_class] = 2.
                        continue
                    else:
                        break

                # Precisionが最も低いクラスが最も間違えている先のクラスのデータを選択
                precision_object[np.isnan(precision_object)] = 2. # nanのクラスは無視
                # precision_object[[False,False,True,True,True,False,False,False,True,True,False]] = 2. # 評価対象外のクラスは無視
                while True:
                    min_precision_class = np.argmin(precision_object)
                    confusion_matrix_object[min_precision_class, min_precision_class] = 0 # 自身のクラスは潰しておく
                    if np.sum(confusion_matrix_object[:, min_precision_class]) == 0: # 間違え元が無い場合
                        # Precisionが飽和
                        if precision_object[min_precision_class] > 1:
                            break
                        precision_object[min_precision_class] = 2.
                        continue

                    min_precision_max_error_class = np.argmax(confusion_matrix_object[:,min_precision_class])
                    if len(confusion_matrix_valid_index[min_precision_max_error_class][min_precision_class]) == 0:
                        confusion_matrix_object[min_precision_max_error_class, min_precision_class] = 0
                        continue
                    else:
                        break

                # RecallとPrecisionの低いほうを選択
                if recall_object[min_recall_class] <= precision_object[min_precision_class]:
                    addition_index = each_class_list[min_recall_class].pop()
                    confusion_matrix_valid_index[min_recall_class][estimation_label_per_object[addition_index]] = remove_value_from_list(confusion_matrix_valid_index[min_recall_class][estimation_label_per_object[addition_index]], addition_index)
                    print(f"support class{estimation_label_per_object[addition_index]}->class{min_recall_class} recall: {recall_object[min_recall_class]:.2} add class{min_recall_class}")
                else:
                    addition_index = majorityVote(confusion_matrix_valid_index[min_precision_max_error_class][min_precision_class])
                    confusion_matrix_valid_index[min_precision_max_error_class][min_precision_class] = remove_value_from_list(confusion_matrix_valid_index[min_precision_max_error_class][min_precision_class], addition_index)
                    each_class_list[min_precision_max_error_class] = remove_value_from_list(each_class_list[min_precision_max_error_class], addition_index)
                    print(f"support class{min_precision_max_error_class}->class{min_precision_class} precision: {precision_object[min_precision_class]:.2} add class{min_precision_max_error_class}")

            # Trainデータに追加
            additional_embedded = np.array(valid_object_patch[addition_index])
            embedded_train = np.concatenate([embedded_train, additional_embedded])
            label_train = np.concatenate([label_train, [valid_object_label[addition_index] for i in range(num_patch_per_object)]])
            print(f"{iValidObject}:train data[{addition_index}] added")


if __name__ == '__main__':
    # 設定を読み込む
    with open('setting.json', 'r') as f:
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
    data_entry_method = setting["valid"]["data_entry_method"]
    random_seed = int(setting["valid"]["random_seed"])
    if data_entry_method == "NotRandomSequential":
        data_entry_method = DataEntryMethod.NotRandomSequential
    elif data_entry_method == "Sequential":
        data_entry_method = DataEntryMethod.Sequential
    elif data_entry_method == "ClassBreadthFirst":
        data_entry_method = DataEntryMethod.ClassBreadthFirst
    elif data_entry_method == "PrecisionRecallBottleneck":
        data_entry_method = DataEntryMethod.PrecisionRecallBottleneck
    else:
        data_entry_method = DataEntryMethod.Sequential

    # 埋め込みベクトルを算出
    if use_calculated_embedded:
        # 算出済みの埋め込みベクトルを使用する
        valid_dir = dir_calculated_embedded
    else:
        # 埋め込みベクトルを算出する
        valid_dir = embeddedCsv(output_root, image_list_path, model_path, num_class, dim_embedded, label_map_dict)

    # 評価を実行
    evaluate(valid_dir, num_patch_per_object, knn_k, data_entry_method, random_seed)
