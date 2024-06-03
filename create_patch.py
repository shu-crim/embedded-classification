import cv2
import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import datetime

class Patch:
    label: int
    area: int
    top: int
    left: int
    width: int
    height: int

    def __init__(self, label, area, top, left, width, height) -> None:
        self.label = label
        self.area = area
        self.top = top
        self.left = left
        self.width = width
        self.height = height


class PatchImage:
    src_img: np.ndarray
    mask: np.ndarray
    label: int

    def __init__(self, src_img, mask, label) -> None:
        self.src_img = src_img
        self.mask = mask
        self.label = label


def Patches(img_label:np.ndarray, th_area=250, patch_width=224, patch_num_per_object=5):
    patches = []

    if img_label.dtype != np.uint8 or img_label.ndim != 2:
        print("Error: img_label.dtype != np.uint8 or img_label.ndim != 2")
        return patches
    
    for label_index in range(1, 256):
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats((img_label == label_index).astype(np.uint8) * 255)

        for i, row in enumerate(stats):
            if i == 0:
                continue # index0は背景
            if row[cv2.CC_STAT_AREA] < th_area:
                continue # 面積がしきい値未満

            # 外接矩形
            # patches.append(Patch(label_index, row[cv2.CC_STAT_AREA], row[cv2.CC_STAT_TOP], row[cv2.CC_STAT_LEFT], row[cv2.CC_STAT_WIDTH], row[cv2.CC_STAT_HEIGHT]))

            print(f"label {label_index} - {i}")
            print(f"* topleft: ({row[cv2.CC_STAT_LEFT]}, {row[cv2.CC_STAT_TOP]})")
            print(f"* size: ({row[cv2.CC_STAT_WIDTH]}, {row[cv2.CC_STAT_HEIGHT]})")
            print(f"* area: {row[cv2.CC_STAT_AREA]}")

            indices = []
            for row_idx, row in enumerate(labels):
                for col_idx, value in enumerate(row):
                    if value == i:
                        indices.append((row_idx, col_idx))

            # print(f"* indices:{indices}")

            index_span = len(indices) // (patch_num_per_object + 1)
            for i_patch in range(1, patch_num_per_object + 1):
                coordinate = indices[i_patch * index_span]
                print(coordinate)
                patches.append(Patch(label_index, row[cv2.CC_STAT_AREA], coordinate[0] - patch_width//2, coordinate[1] - patch_width//2, patch_width, patch_width))

            print("")

    return patches


def patchImage(src_img:np.ndarray, label_img:np.ndarray, patch:Patch, fill_others_gray:bool=False, fill_lumi:np.uint8=127) -> np.ndarray:
    if src_img.shape[0] != label_img.shape[0] or src_img.shape[1] != label_img.shape[1]:
        raise

    if label_img.ndim != 2:
        raise

    if src_img.ndim == 2:
        patch_img = np.zeros((patch.height, patch.width), src_img.dtype)
    elif src_img.ndim == 3:
        patch_img = np.zeros((patch.height, patch.width, src_img.shape[2]), src_img.dtype)
    else:
        raise

    patch_label = np.zeros((patch.height, patch.width), label_img.dtype)


    # 座標算出
    if patch.top < 0:
        src_top = 0
        dst_top = -patch.top
    else:
        src_top = patch.top
        dst_top = 0

    if patch.left < 0:
        src_left = 0
        dst_left = -patch.left
    else:
        src_left = patch.left
        dst_left = 0

    if patch.top + patch.height > src_img.shape[0]:
        src_bottom = src_img.shape[0]
        dst_bottom = src_img.shape[0] - (patch.top + patch.height)
    else:
        src_bottom = patch.top + patch.height
        dst_bottom = patch.height

    if patch.left + patch.width > src_img.shape[1]:
        src_right = src_img.shape[1]
        dst_right = src_img.shape[1] - (patch.left + patch.width)
    else:
        src_right = patch.left + patch.width
        dst_right = patch.height

    patch_img[dst_top:dst_bottom, dst_left:dst_right] = src_img[src_top:src_bottom, src_left:src_right]
    patch_label[dst_top:dst_bottom, dst_left:dst_right] = label_img[src_top:src_bottom, src_left:src_right]

    if fill_others_gray:
        fill_mask = np.logical_and(patch_label != patch.label, patch_label != 0) # 対象のlabelでもラベル0でもない領域
        patch_img[fill_mask] = fill_lumi

    # return patch_img, patch_label
    return patch_img
    

# 設定を読み込む
with open('setting.json', 'r', encoding="utf-8") as f:
    setting = json.load(f)

input_dir = setting["create_patch"]["input_dir"]
output_root = setting["common"]["output_root"]
fill_others_gray = setting["create_patch"]["fill_others_gray"]

output_dir = os.path.join(output_root, datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + f"_patch_{os.path.basename(input_dir)}")
os.makedirs(output_dir)

png_paths = glob.glob(os.path.join(input_dir, "*_rgb.png"))
# print(png_paths)

for png_path in png_paths:
    output_files = glob.glob(os.path.join(output_dir, os.path.basename(png_path).replace("_rgb.png", "*.png")))
    if len(output_files) > 0:
        continue

    img_src_pil = Image.open(png_path)
    img_src = np.array(img_src_pil)
    if not os.path.exists(png_path.replace("_rgb.png", ".png")):
        continue
    img_label = np.array(Image.open(png_path.replace("_rgb.png", ".png")))
    # plt.imshow((img_src == 1).astype(np.uint8), "gray")
    # plt.show()

    patches = Patches(img_label)
    num_patch = 0
    for patch in patches:
        patch:Patch = patch
        patch_img = patchImage(img_src, img_label, patch, fill_others_gray=True)
        # plt.imshow(patch_img)
        # plt.show()
        Image.fromarray(patch_img, img_src_pil.mode).save(os.path.join(output_dir, os.path.basename(png_path).replace("_rgb.png", f"_{num_patch:06}_{patch.label:03}.png")))
        num_patch += 1

    
