import csv
import os
import glob
from embedded_classification_train import readSetting
import numpy as np
import datetime

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

    stats_csv_paths = glob.glob(setting["combine"]["glob_path"])
    if len(stats_csv_paths) == 0:
        print("No csv exists.")
        exit()
    num_seed = len(stats_csv_paths)

    precision_list = []
    recall_list = []
    for stats_csv_path in stats_csv_paths:
        precision, recall, num_addition_list = read_evaluation_csv(stats_csv_path, setting["combine"]["stats_start_row"], setting["combine"]["stats_start_col"], setting["common"]["num_class"])
        precision_list.append(precision)
        recall_list.append(recall)

    all_precision_arr = np.array(precision_list)
    all_recall_arr = np.array(recall_list)
    
    precision_avg_arr = np.average(all_precision_arr, 0)
    recall_avg_arr = np.average(all_recall_arr, 0)
    precision_std_arr = np.std(all_precision_arr, 0)
    recall_std_arr = np.std(all_recall_arr, 0)

    def writeStatsCsv(path, precisions, recalls):
        with open(path, "w", encoding="utf-8") as f:
            f.write("num_addition,")
            for iClass in range(setting["common"]["num_class"]):
                f.write(f'{iClass}_{setting["common"]["label_name_dict"][iClass]}_precision,{iClass}_{setting["common"]["label_name_dict"][iClass]}_recall,')
            f.write("\n")

            for iAddition in range(len(num_addition_list)):
                f.write(f"{num_addition_list[iAddition]},")
                for iClass in range(setting["common"]["num_class"]):
                    f.write(f"{precisions[iAddition][iClass]},{recalls[iAddition][iClass]},")
                f.write("\n")

    output_file_name = os.path.basename(os.path.dirname(setting["combine"]["glob_path"]))

    writeStatsCsv(os.path.join(setting["combine"]["output_dir"], f"average_{output_file_name}.csv"), precision_avg_arr, recall_avg_arr)
    writeStatsCsv(os.path.join(setting["combine"]["output_dir"], f"std_{output_file_name}.csv"), precision_std_arr, recall_std_arr)


    def writeOneClassStats(path, class_index):
        with open(path, "w", encoding="utf-8") as f:
            f.write("num_addition,")
            for iSeed in range(num_seed):
                f.write(f'precision_seed{iSeed},')
            for iSeed in range(num_seed):
                f.write(f'recall_seed{iSeed},')
            f.write("\n")

            for iAddition in range(len(num_addition_list)):
                f.write(f"{num_addition_list[iAddition]},")
                for iSeed in range(num_seed):
                    f.write(f"{all_precision_arr[iSeed][iAddition][class_index]},")
                for iSeed in range(num_seed):
                    f.write(f"{all_recall_arr[iSeed][iAddition][class_index]},")
                f.write("\n")

    target_class = 8
    writeOneClassStats(os.path.join(setting["combine"]["output_dir"], f"class{target_class}_stats.csv"), target_class)
