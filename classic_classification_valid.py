import os
import numpy as np
from datetime import datetime
from PIL import Image
import csv
import cv2
import random
import enum
import time
from multiprocessing import Process

from embedded_classification_train import readSetting


class TargetObject:
    genus_number: int = 0
    rect_top: int = 0
    rect_left: int = 0
    rect_height: int = 0
    rect_width: int = 0
    img_data: np.ndarray = None
    file_name: str = ""


def load_object_list_from_dir(file_paths, label_map_dict, file_skip=0):
    object_list = []
    if file_skip > 0:
        file_paths = file_paths[::file_skip]

    for path in file_paths:
        obj = TargetObject()
        obj.img_data = np.array(Image.open(path))

        genus_number = os.path.basename(path).split(".")[0].split("_")[-1]
        genus_number = int(genus_number)
        obj.genus_number = label_map_dict[genus_number] if genus_number in label_map_dict else 0

        obj.file_name = os.path.basename(path)

        object_list.append(obj)

    return object_list


class Stats:
    num_gt:int = 0
    TP:int = 0
    FP:int = 0
    FN:int = 0


class Feature(enum.Enum):
    ORB = 0
    AKAZE = 1


def evaluate(descriptors_learned, field_learn_object_list, result_csv_path:str, overview_csv_path:str, feature:Feature, label_name_dict:dict):
    if feature == Feature.AKAZE:
        bf = cv2.BFMatcher(cv2.NORM_L1)
        akaze = cv2.AKAZE_create()
    elif feature == Feature.ORB:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        orb = cv2.ORB_create()
    
    evaluation = {}
    for genus_number in label_name_dict.keys():
        evaluation[genus_number] = Stats()

    # with open(result_csv_path, "w") as fp:
    #     fp.write("filename,genus,estimation,distance\n")

    for obj in field_learn_object_list:
        obj:TargetObject = obj

        gray = cv2.cvtColor(obj.img_data[:,:,::-1], cv2.COLOR_BGR2GRAY) 

        # 特徴量抽出
        if feature == Feature.AKAZE:
            keypoint, descriptor = akaze.detectAndCompute(gray, None) 
        elif feature == Feature.ORB:
            keypoint, descriptor = orb.detectAndCompute(gray, None)

        # 識別
        if descriptor is None:
            estimate_genus = 0 # 特徴点が無く識別不能
            min_distance = -1
        else:
            # 属クラス別にマッチングを取る
            min_distance = 1000000000
            min_dist_genus_number = 0
            for genus_number in descriptors_learned:
                if descriptors_learned[genus_number].size == 0:
                    continue
                matches = bf.match(descriptor, descriptors_learned[genus_number])

                # matchesをdescriptorsのdistance順(似ている順)にsortする 
                matches = sorted(matches, key = lambda x:x.distance)
                
                # 最も距離が近いものを採用（暫定）
                distance = matches[0].distance

                # 更新判定
                update = False
                if distance < min_distance:
                    update = True
                elif distance == min_distance:
                    # 距離が同一のときは（主に0で同一となる）、識別の優先順を使う
                    if min_dist_genus_number == 255 or min_dist_genus_number == 0:
                        update = True
                    elif min_dist_genus_number == 254 and genus_number < 254:
                        update = True
                    # elif genus_number > min_dist_genus_number: # 属同士では番号が後ろのほうを優先（アナベナの事前確率が高い問題への抵抗）
                    elif genus_number < min_dist_genus_number: # 属同士では番号が前のほうを優先
                        update = True 

                if update:
                    min_distance = distance
                    min_dist_genus_number = genus_number
            
            estimate_genus = min_dist_genus_number

        # 全結果の書き出し
        # with open(result_csv_path, "a") as fp:
        #     fp.write(f"{obj.file_name},{obj.genus_number},{estimate_genus},{min_distance}\n")

        # 評価
        if estimate_genus == 0:
            estimate_genus = 0 # 特徴点抽出できずはinvalidへ

        stats:Stats = evaluation[obj.genus_number]
        stats.num_gt += 1
        if obj.genus_number == estimate_genus:
            stats.TP += 1
        else:
            stats.FN += 1
            evaluation[estimate_genus].FP += 1
        
    # 評価結果(まとめ)の書き出し
    with open(overview_csv_path, "a") as fp:
        genus_numbers = sorted(list(label_name_dict.keys()))
        for genus_number in genus_numbers:
            # if genus_number == 0:
            #     continue
            s:Stats = evaluation[genus_number] if genus_number in evaluation else Stats()
            # fp.write(f"{genus_number},{s.num_gt},{s.TP},{s.FN},{s.FP},{s.TP/(s.TP+s.FP) if (s.TP+s.FP)>0 else '-'},{s.TP/s.num_gt if s.num_gt>0 else '-'},")
            fp.write(f"{s.TP/(s.TP+s.FP) if (s.TP+s.FP)>0 else 1},{s.TP/s.num_gt if s.num_gt>0 else 1},")

        fp.write("\n")


def total_evaluate(output_dir_root:str, label_map_dict:dict, image_list_path, file_skip:int=1, target_genus_number_list:list=[], feature:Feature=Feature.AKAZE, random_seed:int=-1, feature_skip:int=1, label_name_dict:dict={}):
    if len(target_genus_number_list) == 0:
        evaluation_conditions = "_all_images"
    else:
        evaluation_conditions = ""
        for target_genus in target_genus_number_list:
            evaluation_conditions += f"_{label_name_dict[target_genus]}"

    str_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir_root, f"{str_date}_akaze{evaluation_conditions}_seed{random_seed}")
    os.makedirs(output_dir, exist_ok=True)

    # 画像リストを読み込む
    print("loading images...")
    train_paths = []
    valid_paths = []
    with open(image_list_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == "train":
                train_paths.append(os.path.join(os.path.dirname(image_list_path), row[1]))
            elif row[0] == "valid":
                valid_paths.append(os.path.join(os.path.dirname(image_list_path), row[1]))

    # debug
    # train_paths = train_paths[::50]
    # valid_paths = valid_paths[::50]

    # 画像を読み込む
    pre_learn_object_list = load_object_list_from_dir(train_paths, label_map_dict, file_skip)
    field_learn_object_list = load_object_list_from_dir(valid_paths, label_map_dict, file_skip)

    # 現場で学ぶデータをシャッフル
    random.seed(random_seed)
    random.shuffle(field_learn_object_list)

    print("Images have loaded.")

    if feature == Feature.AKAZE:
        # AKAZE検出器の生成
        akaze = cv2.AKAZE_create() 
    elif feature == Feature.ORB:
        orb = cv2.ORB_create()

    # 事前学習データから特徴量を抽出してリスト化
    print("Calculating Pre-Learning descriptor...")
    descriptors_learned = {}
    for genus_number in label_name_dict.keys():
        descriptors_learned[genus_number] = []

    for obj in pre_learn_object_list:
        obj:TargetObject = obj

        gray = cv2.cvtColor(obj.img_data[:,:,::-1], cv2.COLOR_BGR2GRAY) 

        # 特徴量抽出
        if feature == Feature.AKAZE:
            keypoint, descriptor = akaze.detectAndCompute(gray, None) 
        elif feature == Feature.ORB:
            keypoint, descriptor = orb.detectAndCompute(gray, None)

        if descriptor is None:
            continue

        # 特徴量を間引く
        descriptor = descriptor[::feature_skip]

        descriptors_learned[obj.genus_number] += descriptor.tolist()

    for key in descriptors_learned:
        print(f"genus:{key}, num:{len(descriptors_learned[key])}")
        descriptors_learned[key] = np.array(descriptors_learned[key], dtype=np.uint8)

    print("")
    print("Estimation starts.")

    # 結果のまとめcsvを作成
    num_field_learn_img = 0
    with open(os.path.join(output_dir, "result_overview.csv"), "w") as fp:
        fp.write("\n")
        fp.write(f"num_object,{len(field_learn_object_list)}\n")
        fp.write(f"random_seed,{random_seed}\n")
        fp.write("\n")
        fp.write("\n")
        fp.write("num_add,field_learn_image,addition_class,accuracy,-,")
        genus_numbers = sorted(list(label_name_dict.keys()))
        for genus_number in genus_numbers:
            # if genus_number == 0:
            #     continue
            # fp.write(f"{label_name_dict[genus_number]},num_gt,TP,FN,FP,Precision,Recall,")
            fp.write(f"{genus_number}_{label_name_dict[genus_number]}_precision,{genus_number}_{label_name_dict[genus_number]}_recall,")
        fp.write("\n")

    # 初期状態での評価
    with open(os.path.join(output_dir, "result_overview.csv"), "a") as fp:
        fp.write("0,-,-,-,-,")
    evaluate(descriptors_learned, field_learn_object_list, os.path.join(output_dir, f"result_{num_field_learn_img:0>8}_images_learned.csv"), os.path.join(output_dir, "result_overview.csv"), feature, label_name_dict)

    for obj in field_learn_object_list:
        obj:TargetObject = obj

        # 特定の属のみ学習していく
        if len(target_genus_number_list) > 0: # 空の場合はすべて学習
            if not obj.genus_number in target_genus_number_list:
                continue

        print(f"learn: {obj.file_name}")
        num_field_learn_img += 1

        gray = cv2.cvtColor(obj.img_data[:,:,::-1], cv2.COLOR_BGR2GRAY) 

        # 特徴量抽出
        if feature == Feature.AKAZE:
            keypoint, descriptor = akaze.detectAndCompute(gray, None) 
        elif feature == Feature.ORB:
            keypoint, descriptor = orb.detectAndCompute(gray, None)
        
        if descriptor is not None:
            # 特徴量を間引く
            descriptor = descriptor[::feature_skip]

            descriptors_list = descriptors_learned[obj.genus_number].tolist() # 一旦listに変換
            descriptors_list += descriptor.tolist() # 特徴量を追加
            descriptors_learned[obj.genus_number] = np.array(descriptors_list, dtype=np.uint8) # 再びnumpyへ
        
            # 評価
            with open(os.path.join(output_dir, "result_overview.csv"), "a") as fp:
                fp.write(f"{num_field_learn_img},{obj.file_name},{obj.genus_number},-,-,")
            evaluate(descriptors_learned, field_learn_object_list, os.path.join(output_dir, f"result_{num_field_learn_img:0>8}_images_learned.csv"), os.path.join(output_dir, "result_overview.csv"), feature, label_name_dict)


def writeFeatureVector(output_root, label_map_dict, label_name_dict, image_list_path, feature, file_skip):
    str_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_root, f"{str_date}_akaze_features")
    os.makedirs(output_dir, exist_ok=True)

    # 画像リストを読み込む
    print("loading images...")
    train_paths = []
    valid_paths = []
    with open(image_list_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == "train":
                train_paths.append(os.path.join(os.path.dirname(image_list_path), row[1]))
            elif row[0] == "valid":
                valid_paths.append(os.path.join(os.path.dirname(image_list_path), row[1]))

    def writeDescriptors(path, object_list, label_name_dict, feature):
        if feature == Feature.AKAZE:
            # AKAZE検出器の生成
            akaze = cv2.AKAZE_create() 
        elif feature == Feature.ORB:
            orb = cv2.ORB_create()

        with open(path, "w", encoding="utf-8") as fp:
            fp.write("filename,class,name,num_features,\n")

            for obj in object_list:
                obj:TargetObject = obj
                gray = cv2.cvtColor(obj.img_data[:,:,::-1], cv2.COLOR_BGR2GRAY) 

                # 特徴量抽出
                if feature == Feature.AKAZE:
                    keypoints, descriptors = akaze.detectAndCompute(gray, None) 
                elif feature == Feature.ORB:
                    keypoints, descriptors = orb.detectAndCompute(gray, None)

                # ファイルごとに、特徴点抽出数を出力
                fp.write(f"{os.path.basename(obj.file_name)},{obj.genus_number},{label_name_dict[obj.genus_number]},{len(keypoints)}\n")

                # if descriptors is None:
                #     continue

                # for descriptor in descriptors:
                #     fp.write(f"{os.path.basename(obj.file_name)},{obj.genus_number},{label_name_dict[obj.genus_number]},")
                #     for value in descriptor:
                #         fp.write(f"{value},")
                #     fp.write("\n")


    # 事前学習データから特徴量を抽出してリスト化
    print("Loading train images...")
    pre_learn_object_list = load_object_list_from_dir(train_paths, label_map_dict, file_skip)

    print("Writing train descriptor...")
    writeDescriptors(os.path.join(output_dir, "pre_learning_features.csv"), pre_learn_object_list, label_name_dict, feature)

    print("Loading valid images...")
    field_learn_object_list = load_object_list_from_dir(valid_paths, label_map_dict, file_skip)

    print("Writing valid descriptor...")
    writeDescriptors(os.path.join(output_dir, "valid_features.csv"), field_learn_object_list, label_name_dict, feature)


def main():
    # 設定を読み込む
    SETTING_FILE_NAME = 'setting.json'
    setting = readSetting(SETTING_FILE_NAME)
    random_seed = setting["valid"]["random_seed"]
    feature_skip = 5

    # 特徴量の抽出数を書き出し
    # writeFeatureVector(setting["common"]["output_root"], setting["common"]["label_map_dict"], setting["common"]["label_name_dict"], setting["common"]["image_list_path"], Feature.AKAZE, file_skip=1)
    # exit()

    # 評価
    start_time = time.time()

    for round in range(setting["valid"]["round_run_valid"]):
        tasks = []
        for i in range(setting["valid"]["num_run_valid"]):
            if random_seed < -1:
                random_seed = random.randint(0, 100000)

            tasks.append(Process(target=total_evaluate, args=(
                setting["common"]["output_root"], setting["common"]["label_map_dict"], setting["common"]["image_list_path"], setting["common"]["num_patch_per_object"], [], Feature.AKAZE, random_seed, feature_skip, setting["common"]["label_name_dict"]
                )))
            
            random_seed += 1
            
        for task in tasks:
            task.start()

        for task in tasks:
            task.join()

    print(f"Proctime: {time.time()-start_time} s")


if __name__ == "__main__":
    main()
