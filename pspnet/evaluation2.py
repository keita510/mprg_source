# -*- coding: utf-8 -*

import time
import numpy as np
import cv2
from glob import glob
import sys
import numba as nb

# 画像サイズ指定
h = 320
w = 960

#"""----------  numbaによる高速化処理  ----------"""

@nb.jit(nb.float32[:,:](nb.float32[:,:], nb.int64, nb.uint16[:,:,:], nb.uint16[:,:,:], nb.float64[:,:]),nopython=True)
def gaso_calc(matrix,index,gt_img,pre_img,array):
    for height in range(h):
        for width in range(w):
            for Input in range(index):
                if gt_img[height, width, 2] == array[Input, 0] and gt_img[height, width, 1] == array[Input, 1] and gt_img[height, width, 0] == array[Input, 2]:
                    for Output in range(index):    
                        if pre_img[height, width, 2] == array[Output, 0] and pre_img[height, width, 1] == array[Output, 1] and pre_img[height, width, 0] == array[Output, 2]:
                             matrix[Input, Output] += 1.0
    return matrix

#"""----------------------------------------------"""

# 教師画像のPATH
FILE_gt = glob('./OUT_tame/eval_test/*.png')
# 出力画像のPATH
FILE_pre = glob('./OUT_tame/fcn_result/*.png')


#教師と出力画像の枚数を取得
gt_num = len(FILE_gt)
pre_num = len(FILE_pre)

#教師と出力の要素数が違った場合
if gt_num != pre_num:
    print(" Each File numbers is different !! ")
    print(" ---------------------STOP---------------------- ")
    sys.exit()
else:
    print("Test image numbers are " + str(gt_num) + ".")

array = np.ones([900, 3])
array[:, :] = 999
index = 0
print("Extracting RGB Color")
for num in xrange(gt_num):
    # 教師画像の読み込み
    gt_img = cv2.imread('./OUT_tame/eval_test/' + str(num+1) + '.png')
    # 出力画像の読み込み
    pre_img = cv2.imread('./OUT_tame/fcn_result/' + str(num+1) + '.png')

    b = gt_img[:, :, 0]
    g = gt_img[:, :, 1]
    r = gt_img[:, :, 2]

    for height in xrange(h):
        for width in xrange(w):
            b = gt_img[height, width, 0]
            g = gt_img[height, width, 1]
            r = gt_img[height, width, 2]

            same = 0
            for counter in xrange(index):
                if array[counter, 0] == r and array[counter, 1] == g and array[counter, 2] == b:
                    same += 1

            if same == 0:
                np.vstack((array, (r, g, b)))
                array[index, 0] = r
                array[index, 1] = g
                array[index, 2] = b
                index += 1

    print(str(num+1) + '/' + str(gt_num))

for class_num in xrange(index):
    print(array[class_num])
print("Included RGB Color is " + str(index))
matrix = np.zeros([index, index]).astype(np.float32)

for num in xrange(gt_num):
    # 教師画像の読み込み
    gt_img = cv2.imread('./OUT_tame/eval_test/' + str(num+1) + '.png').astype(np.uint16)
    # 出力画像の読み込み
    pre_img = cv2.imread('./OUT_tame/fcn_result/' + str(num+1) + '.png').astype(np.uint16)
    # 速度計算開始
    start = time.time()
    # numbaによる計算処理
    matrix = gaso_calc(matrix,index,gt_img,pre_img,array)
    # 計算終了
    print num
    end = time.time()
    print("処理時間" + str(end-start) + "[sec]")
    # 時間のリセット
    end = 0
    start = 0


G_pixel = 0
for G_num in xrange(index):
    G_pixel += matrix[G_num, G_num]
#正解したピクセル/画像全体のピクセル
GA = G_pixel / matrix.sum()
print("Global Accuracy -----> " + str(GA * 100) + "%")

ca = 0
for C_num in xrange(index):
    ca += matrix[C_num, C_num] / matrix[C_num, :].sum()
#対象クラスの正推定の合計/対象クラスの正解の合計
CA = ca / index
print("Class Accuracy ------> " + str(CA * 100) + "%")

mi = 0
for M_num in xrange(index):
    mi += matrix[M_num, M_num] / (matrix[M_num, :].sum() + matrix[:, M_num].sum() - matrix[M_num, M_num])
#対象クラスの正推定の合計/対象クラスの正解の合計+対象クラスの誤識別の合計
MI = mi / index
print("Mean IOU -------------> " + str(MI * 100) + "%")


