# -*- coding:utf-8 -*-
# python 2.7 (anaconda2-4.2.0)

# ディレクトリの中に存在する画像を指定のサイズに変更するスクリプト


import cv2
import os
import glob
from PIL import Image

# 画像のあるディレクトリを指定(最後にスラッシュつける)
img_dir = "./3dobject_arc_syoki/"
# 出力したいディレクトリの指定(最後にスラッシュつける)
out_path = "./3dobject_arc/"
# 出力するファイル名 連番なら:0，入力と同じなら:1
out_name_flag = 0
# 画像サイズ
width, height = (640, 480)


################################################################################


# 出力ディレクトリの作成
if os.path.exists(out_path) is False:
	os.mkdir(out_path)


# ファイルリストの取得
img_file = glob.glob(img_dir+"*.png")
# ファイルのソート
img_file.sort()
#print("file list : \n{}".format(img_file))

for file in enumerate(img_file, start=0):
	# 画像情報の取得
	img = cv2.imread(file[1], 1)
	# height, width = img.shape[:2] # 画像から取得
#	print(img)
	# リサイズ(適宜変更)
	img = cv2.resize(img,(width,height), interpolation=cv2.INTER_NEAREST )

	# 画像の出力
	if   out_name_flag is 0: save_name = "{}{:07d}.png".format(out_path, file[0])
	elif out_name_flag is 1: save_name = "{}{}".format(out_path, file[1])
	cv2.imwrite(save_name,img)
	print("save image : {}".format(save_name))
