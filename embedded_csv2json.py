import csv
import json
import os
import random
import numpy as np
from embedded_classification_train import readSetting, SETTING_FILE_NAME, fix_seed
from embedded_classification_valid import readEmbedded

if __name__ == '__main__':
    # 設定を読み込む
    setting = readSetting(SETTING_FILE_NAME)

    # 読み込み
    label_train, embedded_train = readEmbedded(os.path.join(setting["valid"]["dir_calculated_embedded"], "train_embedded.csv"), setting["common"]["dim_embedded"])
    label_valid, embedded_valid = readEmbedded(os.path.join(setting["valid"]["dir_calculated_embedded"], "valid_embedded.csv"), setting["common"]["dim_embedded"])

    # 1オブジェクト1データに削減
    label_train = label_train[::setting["common"]["num_patch_per_object"]]
    embedded_train = embedded_train[::setting["common"]["num_patch_per_object"]]
    label_valid = label_valid[::setting["common"]["num_patch_per_object"]]
    embedded_valid = embedded_valid[::setting["common"]["num_patch_per_object"]]

    # データのランダムドロップアウト
    num_data = 1000
    fix_seed(setting["common"]["random_seed"])
    np.random.shuffle(label_train)
    label_train = label_train[:num_data]
    fix_seed(setting["common"]["random_seed"])
    np.random.shuffle(embedded_train)
    embedded_train = embedded_train[:num_data]
    fix_seed(setting["common"]["random_seed"])
    np.random.shuffle(label_valid)
    label_valid = label_valid[:num_data]
    fix_seed(setting["common"]["random_seed"])
    np.random.shuffle(embedded_valid)
    embedded_valid = embedded_valid[:num_data]

    # データ成形
    goal_rate = 0.6
    output_data = {}
    output_data["data"] = [{}]
    output_data["data"][0]["vector"] = embedded_valid.tolist()
    output_data["data"][0]["gt"] = label_valid.tolist()
    output_data["data"][0]["train"] = embedded_train.tolist()
    output_data["data"][0]["train-gt"] = label_train.tolist()
    output_data["data"][0]["goal"] = goal_rate

    # 書き込み
    with open(os.path.join(setting["valid"]["dir_calculated_embedded"], "embedded.json"), "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=4)  # indent引数で整形
