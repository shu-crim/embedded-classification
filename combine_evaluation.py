import csv
import os
import glob
from embedded_classification_train import readSetting
import numpy as np

def read_evaluation_csv(csv_path, stats_start_row, stats_start_col, num_class):
    num_addition_list = []
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for i in range(stats_start_row):
            next(reader)
        precision = []
        recall = []
        for row in reader:
            if len(row) < stats_start_col + num_class * 2:
                continue

            precision_row = []
            for r in range(stats_start_col, stats_start_col + num_class * 2, 2):
                precision_row.append(np.float32(row[r]))
            precision.append(precision_row)

            recall_row = []
            for r in range(stats_start_col + 1, stats_start_col + num_class * 2, 2):
                recall_row.append(np.float32(row[r]))
            recall.append(recall_row)

            num_addition_list.append(int(row[0]))

    return precision, recall, num_addition_list


if __name__ == '__main__':
    # 設定を読み込む
    SETTING_FILE_NAME = 'setting.json'
    setting = readSetting(SETTING_FILE_NAME)
    stats_csv_dir = setting["combine"]["stats_csv_dir"]

    stats_csv_paths = glob.glob(os.path.join(stats_csv_dir, "*.csv"))
    if len(stats_csv_paths) == 0:
        print("No csv exists.")
        exit()

    precision_list = []
    recall_list = []
    for stats_csv_path in stats_csv_paths:
        precision, recall, num_addition_list = read_evaluation_csv(stats_csv_path, setting["combine"]["stats_start_row"], setting["combine"]["stats_start_col"], setting["common"]["num_class"])
        precision_list.append(precision)
        recall_list.append(recall)
    
    precision_arr = np.average(np.array(precision_list), 0)
    recall_arr = np.average(np.array(recall_list), 0)

    with open(os.path.join(stats_csv_dir, "average_" + os.path.basename(stats_csv_dir) + ".csv"), "w", encoding="utf-8") as f:
        f.write("num_addition,")
        for iClass in range(setting["common"]["num_class"]):
            f.write(f'{iClass}_{setting["common"]["label_name_dict"][iClass]}_precision,{iClass}_{setting["common"]["label_name_dict"][iClass]}_recall,')
        f.write("\n")

        for iAddition in range(len(num_addition_list)):
            f.write(f"{num_addition_list[iAddition]},")
            for iClass in range(setting["common"]["num_class"]):
                f.write(f"{precision_arr[iAddition][iClass]},{recall_arr[iAddition][iClass]},")
            f.write("\n")



