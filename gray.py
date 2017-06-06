# -*- coding: utf-8 -*

import os
import cv2
import numpy as np
#入力画像のパス
img_path = "/home/keita/trainaisin/"

def main(img_path):
	#画像の読み込み
	img_file = os.listdir(img_path)
	#画像のソート
	img_file.sort()	

	for num in range(50):
		#画像の読み込み
		img = cv2.imread(img_path+img_file[num])
		#画像上をラスタスキャンし，画素値の指定を行う
		for y in range(960):
			for x in range(1280):
				#画像上のBGRを取得
				blue = img.item(y,x,0)
				green = img.item(y,x,1)
				red = img.item(y,x,2)
				#グレースケール変換式
				#gr = 0.299*red + 0.587*green + 0.114*blue
				
				#人クラスの時、画素値指定20
				if (blue == 85) and (green == 170) and (red == 255):
					img.itemset((y,x,0),20)
					img.itemset((y,x,1),20)
					img.itemset((y,x,2),20)
				#車クラスの時、画素値指定40
				elif (blue == 170) and (green == 170) and (red == 0):
					img.itemset((y,x,0),40)
                                	img.itemset((y,x,1),40)
                                	img.itemset((y,x,2),40)
				#障害物クラスの時、画素値指定60
                        	elif (blue == 85) and (green == 85) and (red == 170):
                                	img.itemset((y,x,0),60)
                                	img.itemset((y,x,1),60)
                                	img.itemset((y,x,2),60)
				#白線クラスの時、画素値指定80
                        	elif (blue == 170) and (green == 0) and (red == 255):
                                	img.itemset((y,x,0),80)
                                	img.itemset((y,x,1),80)
                                	img.itemset((y,x,2),80)
				#空クラスの時、画素値指定
                        	elif (blue == 170) and (green == 85) and (red == 0):
                                	img.itemset((y,x,0),100)
                                	img.itemset((y,x,1),100)
                                	img.itemset((y,x,2),100)
				#車止めクラスの時、画素値指定
                        	elif (blue == 170) and (green == 0) and (red == 85):
                                	img.itemset((y,x,0),120)
                                	img.itemset((y,x,1),120)
                                	img.itemset((y,x,2),120)
				#道路クラスの時、画素値指定
                        	elif (blue == 85) and (green == 85) and (red == 85):
                                	img.itemset((y,x,0),140)
                                	img.itemset((y,x,1),140)
                                	img.itemset((y,x,2),140)
				#歩道クラスの時、画素値指定
                      		elif (blue == 0) and (green == 255) and (red == 255):
                                	img.itemset((y,x,0),160)
                                	img.itemset((y,x,1),160)
                                	img.itemset((y,x,2),160)
				#段差クラスの時、画素値指定
                        	elif (blue == 0) and (green == 85) and (red == 0):
                                	img.itemset((y,x,0),180)
                                	img.itemset((y,x,1),180)
                                	img.itemset((y,x,2),180)
				#バイククラスの時、画素値指定
                        	elif (blue == 85) and (green == 0) and (red == 255):
                                	img.itemset((y,x,0),200)
                                	img.itemset((y,x,1),200)
                                	img.itemset((y,x,2),200)
				#その他クラスの時、画素値指定
                        	elif (blue == 0) and (green == 0) and (red == 0):
                                	img.itemset((y,x,0),0)
                                	img.itemset((y,x,1),0)
                                	img.itemset((y,x,2),0)

	# RGBからグレースケール
     #   gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	#画像の保存
#	cv2.imwrite("./hyouka/0000000.bmp", img);
		cv2.imwrite("./outgray/"+str(num+1)+".bmp", img)  #出力画像
		print(str(num+1)+"image was saved.")
	

if __name__ == '__main__':
	main(img_path)


