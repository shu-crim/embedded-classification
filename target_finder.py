import os
import random
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from enum import Enum
import copy
import glob
import datetime
import shutil
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from embedded_classification_train import readSetting


SETTING_FILE_NAME = 'setting.json'

def recognition(input_data):
    # ここに処理を書く
    gray = cv2.cvtColor(input_data, cv2.COLOR_RGB2GRAY)

    scale_inv = 4
    resized = cv2.resize(gray, (gray.shape[1] // scale_inv, gray.shape[0] // scale_inv), interpolation=cv2.INTER_AREA)

    # 二値化→メディアンフィルタ
    # th, otsu = cv2.threshold(255 - resized, 0, 255, cv2.THRESH_OTSU)
    karnel_width = 200 // scale_inv
    if karnel_width % 2 == 0:
        karnel_width += 1
    img_bin = cv2.adaptiveThreshold(255 - resized, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, karnel_width, -10)

    img_bin = cv2.medianBlur(img_bin, 3)
    bool_bin = img_bin == 255

    # sobelフィルタ→しきい値処理→closing
    sobel = np.abs(cv2.Sobel(resized, cv2.CV_32F, 1, 1, ksize=3))
    th_sobel = 20
    sobel_binary = (sobel > th_sobel).astype(np.uint8) * 255
    kernel = np.ones((5,5),np.uint8)
    sobel_closing = cv2.morphologyEx(sobel_binary, cv2.MORPH_CLOSE, kernel)

    # sobel_closingの小さなblobを除去
    th_area = 250 // scale_inv
    bool_soble_noise_reduction = np.zeros(sobel_closing.shape, bool)
    nLabels_pred, label_image_raw, stats, center = cv2.connectedComponentsWithStats(sobel_closing)
    for index in range(1, nLabels_pred):
        # 面積が小さいblobを除外
        if stats[index, 4] < th_area:
            continue
        bool_soble_noise_reduction[label_image_raw == index] = True

    # フロック領域 = 二値化 | 輝度勾配
    bool_fg_area = (bool_bin | bool_soble_noise_reduction)
    fg_area = bool_fg_area.astype(np.uint8) * 255

    # th_area = 300
    label_image = np.zeros(img_bin.shape, np.uint8)
    nLabels_pred, label_image_raw, stats, center = cv2.connectedComponentsWithStats(fg_area)

    num_label = 0
    for index in range(1, nLabels_pred):
        # 面積が小さいblobを除外
        if stats[index, 4] < th_area:
            continue
        num_label += 1
        label_image[label_image_raw == index] = num_label

    # 元の解像度に戻す
    label_image = cv2.resize(label_image, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    img_bin = cv2.resize(bool_bin.astype(np.uint8)*255, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    img_sobel = cv2.resize(bool_soble_noise_reduction.astype(np.uint8)*255, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    return num_label, label_image, img_bin, img_sobel


def patchImage(src_img:np.ndarray, row, col, width, fill_lumi:np.uint8=127) -> np.ndarray:
    if src_img.ndim == 2:
        patch_img = np.ones((width, width), src_img.dtype) * fill_lumi
    elif src_img.ndim == 3:
        patch_img = np.ones((width, width, src_img.shape[2]), src_img.dtype) * fill_lumi
    else:
        raise

    # 座標算出
    top = row - width // 2
    if top < 0:
        src_top = 0
        dst_top = -top
    else:
        src_top = top
        dst_top = 0

    left = col - width // 2
    if left < 0:
        src_left = 0
        dst_left = -left
    else:
        src_left = left
        dst_left = 0

    if top + width > src_img.shape[0]:
        src_bottom = src_img.shape[0]
        dst_bottom = src_img.shape[0] - (top + width)
    else:
        src_bottom = top + width
        dst_bottom = width

    if left + width > src_img.shape[1]:
        src_right = src_img.shape[1]
        dst_right = src_img.shape[1] - (left + width)
    else:
        src_right = left + width
        dst_right = width

    patch_img[dst_top:dst_bottom, dst_left:dst_right] = src_img[src_top:src_bottom, src_left:src_right]

    return patch_img


def createPatch(src_img, num_label, label_image, output_dir, src_img_name, patch_width=224, patch_per_label=3, fill_lumi:np.uint8=127):
    indices = [[] for i in range (num_label)]
    for row_idx, row in enumerate(label_image):
        for col_idx, value in enumerate(row):
            if value > 0:
                indices[value - 1].append((row_idx, col_idx))

    for i in range(num_label):
        num_indices = len(indices[i])
        for iPatch in range(patch_per_label):
            random_index = random.randint(0, num_indices-1)
            row, col = indices[i][random_index]

            patch_img = patchImage(src_img, row, col, patch_width, fill_lumi)
            Image.fromarray(patch_img).save(os.path.join(output_dir, f"{src_img_name}_{i:03}_{iPatch:03}.png"))


def clustering(patch_dir, output_dir, pca_dim=64, num_cluster=20, start_class_index=0):
    images = []
    paths = glob.glob(os.path.join(patch_dir, "*.png"))
    for path in paths:
        # 画像を読み込んでリストに追加
        images.append(np.array(Image.open(path)))

    # 画像リストを1次元に平坦化
    flattened_images = np.array(images).reshape(len(images), -1)

    # PCAを適用
    n_components = pca_dim  # 主成分の数を指定
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(flattened_images)

    # 主成分の寄与率を表示
    print("Explained variance ratio for each component:")
    print(pca.explained_variance_ratio_)

    # 主成分の重要度を表示
    # print("Principal components:")
    # for i in range(n_components):
    #     print(f"Component {i+1}: {pca.components_[i]}")

    # # 低次元の特徴空間に射影されたデータを取得
    # reduced_images = pca.inverse_transform(pca_result)
    # print(f"reduced_images.shape: {reduced_images.shape}")

    # # 画像を元の形に戻す
    # restored_images = reduced_images.reshape(len(images), *images[0].shape)

    # # ここでrestored_imagesにPCAを適用した画像が格納されています
    # for i, image in enumerate(restored_images):
    #     img_restore = np.clip(image, 0, 255).astype(np.uint8) 
    #     Image.fromarray(img_restore).save(os.path.join(output_dir, os.path.basename(paths[i])))

    # k-meansモデルを初期化
    kmeans = KMeans(n_clusters=num_cluster)

    # データをクラスタリング
    kmeans.fit(pca_result)

    # 各データポイントの所属クラスタを取得
    labels = kmeans.labels_

    # # クラスタの中心を取得
    # centroids = kmeans.cluster_centers_

    # # 結果を表示
    # print("クラスタラベル:", labels)
    # print("クラスタ中心:", centroids)

    for i, path in enumerate(paths):
        shutil.copy2(path, os.path.join(output_dir, f"kmeans_{labels[i]:03}_" + os.path.basename(path).replace(".png", f"_{start_class_index+labels[i]:03}.png")))


def main():
    # 設定を読み込む
    setting = readSetting(SETTING_FILE_NAME)

    # output
    # now = datetime.datetime.now()
    # output_dir = os.path.join(setting["common"]["output_root"], now.strftime('%Y%m%d_%H%M%S_target_finder_' + os.path.basename(setting["target_finder"]["input_dir"])))
    # os.makedirs(output_dir)
    # shutil.copy(os.path.abspath(__file__), output_dir)
    # shutil.copy(SETTING_FILE_NAME, output_dir)
    # patch_dir = os.path.join(output_dir, "patch")
    # os.makedirs(patch_dir)
    # kmeans_dir = os.path.join(output_dir, "k_means")
    # os.makedirs(kmeans_dir)

    # paths = glob.glob(os.path.join(setting["target_finder"]["input_dir"], "*.png"))
    # for path in paths:
    #     print(f"Proccessing {os.path.basename(path)}")

    #     # 画像読み込み
    #     img_src = np.array(Image.open(path))
    #     num_label, label_image, img_bin, img_sobel = recognition(img_src)

    #     # 画像書き出し
    #     dst_img_fg = img_src // 2
    #     dst_img_fg[label_image > 0] += np.array((0,255,255), np.uint8) // 2
    #     Image.fromarray(dst_img_fg).save(os.path.join(output_dir, os.path.basename(path).split('.')[0] + "_fg.png"))

    #     # dst_img_otsu = img_src // 2
    #     # dst_img_otsu[img_bin > 0] += np.array((255,255,0), np.uint8) // 2
    #     # Image.fromarray(dst_img_otsu).save(os.path.join(output_dir, os.path.basename(path).split('.')[0] + "_otsu.png"))

    #     # dst_img_sobel = img_src // 2
    #     # dst_img_sobel[img_sobel > 0] += np.array((255,0,255), np.uint8) // 2
    #     # Image.fromarray(dst_img_sobel).save(os.path.join(output_dir, os.path.basename(path).split('.')[0] + "_sobel.png"))

    #     # パッチ生成して保存
    #     createPatch(img_src, num_label, label_image, patch_dir, os.path.basename(path).split(".")[0])

    # パッチを読み込んでクラスタリング
    patch_dir = r"../output/20240613_161841_target_finder_ラベル画像なし/patch" #debug
    kmeans_dir = r"../output/20240613_161841_target_finder_ラベル画像なし/k_means" #debug

    clustering(patch_dir, kmeans_dir, pca_dim=64, num_cluster=64, start_class_index=256)

if __name__ == "__main__":
    main()
