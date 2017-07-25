# -*- coding: utf-8 -*

import os
import cv2
import numpy as np
import glob
from augmentation import Augmentation


WIDTH = 1280
HEIGHT = 960
'----------------------------------------'

#入力画像のパス
img_path = "/home/"
label_path = "/home/"

augmentation = Augmentation(img_path,label_path,HEIGHT,WIDTH)

images = glob.glob(img_path)
print(images)
def main(img_path):
	#nは要素番号0,1,2...みたいな
	for n, f in enumerate(images):
		print(f)	#リストの要素の表示（今回はパス）
		img ,_= augmentation()
		print(img.shape)	#shapeを表示させてる
		img = np.transpose(img, (1,2,0))
		print(img.shape)
		cv2.imwrite("./out/{}.bmp".format(str(n+1)), img)  #出力画像
		print(str(n+1)+"image was saved.")
	

if __name__ == '__main__':
	main(img_path)


