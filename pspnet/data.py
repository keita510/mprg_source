# coding: utf-8
import os
from os.path import join
import csv
import shutil
import cv2
from PIL import Image
import numpy as np
from glob import glob
from augmentation import Augmentation

# ファイルデータを元に入力画像と教師画像を整形して返す関数(セグメンテーション画像はニアレストネイバーで処理)
"""os.sepのやつがあることで開発環境Windowsでもおk。
   #入力画像と教師画像のファイル名が同一でない時に必要
def list_img_and_seg_path(txt_file, img_dir, seg_dir):
def list_img_and_seg_path(img_dir, seg_dir):
    # txtファイルからlistを作成する
    with open(txt_file, 'r') as f:
        reader = csv.reader(f)
        path_list = [row for row in reader]

    # listから入力画像と教師画像のパスがセットになったlistを作成
    for i, f in enumerate(path_list):
        img_path = join(img_dir, f[0] + ".bmp").replace(os.sep, '/')
        seg_path = join(seg_dir, f[0] + ".bmp").replace(os.sep, '/')
        path_list[i] = [img_path, seg_path]
    return path_list
"""
def list_img_and_seg_path(img_dir, seg_dir):
    #画像のソートを行う関数
    img_path = glob(img_dir)
    img_path.sort()
    seg_path = glob(seg_dir)
    seg_path.sort()
    path_list = [img_path, seg_path]
    return path_list

def read_img(img_path, resize):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 3ch image
    img = cv2.resize(img, resize, interpolation=cv2.INTER_NEAREST)
    return img.transpose(2, 1, 0)  # OpenCVで画像を読み込むと形が変わるため整える (さらに正規化しておく)

def read_seg(seg_path, resize):
    seg = cv2.imread(seg_path, 0)  # 1ch image(grayscale)
    seg = cv2.resize(seg, resize, interpolation=cv2.INTER_NEAREST)
    seg = seg/20
#    seg[0] = -1  # 邪魔なラベルは-1にすることでchainerの誤差関数等で無視する
    return seg.transpose(1, 0)

#画像の読み込み(train.pyで呼んでいる)
def fetch_img_and_label_by_list(img_path_list, seg_path_list, resize=(480,640)):
    img_data = []
    seg_data = []
    #augmentation = Augmentation(img_data, seg_data, HEIGHT, WIDTH)
#zipを使えばimgとsegごとにリストの要素を取得することができる
    for img, seg in zip(img_path_list, seg_path_list):
        _img = read_img(img_path=img, resize=resize)
        img_data.append(_img)
        _seg = read_seg(seg, resize)
        seg_data.append(_seg)
    return img_data, seg_data


if __name__ == '__main__':
    img_dir = "./dataset1/
    seg_dir = "./dataset2/

    data = list_img_and_seg_path(img_dir=img_dir, seg_dir=seg_dir)
    print(data)
