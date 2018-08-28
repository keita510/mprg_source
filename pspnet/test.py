#! /usr/bin/env python
# -*- coding: utf-8 -*- 
import sys 		
import time 
import numpy as np	 
import cv2  		
import os


import chainer		
from chainer import optimizers	
from chainer import serializers	
from chainer import cuda	
import chainer.functions as F 
#import data
#PSPNet
from pspnet import Net
#SegNet
#from net import Model	
from dataloader import loader	

'''-----------------------------------------------------'''
# テスト用の入力画像が入ったフォルダのPATH
RGB_PATH = "./dataset1/*.png"
# テスト用の教師画像が入ったフォルダのPATH
LABEL_PATH = "./dataset2/*.png"
# ネットワークへの入力画像サイズ(学習時と同じ)
WIDTH = 640
HEIGHT = 480

# 学習モデルのPATH
MODEL_PATH = "./loss_seg/400psp_weight.npz"
# 使用するGPU番号
GPU_ID = 0
'''-----------------------------------------------------'''


def test(label_colours):			
	# net.pyのネットワークの呼び出し
	model = Net(n_class=10, input_size=(HEIGHT,WIDTH), n_blocks=[3, 4, 6, 3],
        pyramids=[6, 3, 2, 1], mid_stride=True, mean=np.array([123.68, 116.779, 103.939]), 
        comm=None, pretrained_model=None, initialW=None)

	#CPUモードからGPUモードに変更
	cuda.get_device(GPU_ID).use()		
	model.to_gpu()

	# 学習モデルの読み込み
	serializers.load_npz(MODEL_PATH, model)

	load = loader()
	

	for num in range(102): #102 			
		num = num			
		image, label = load.Test(RGB_PATH, LABEL_PATH, num, HEIGHT, WIDTH) 
		print(image.shape)
		image_gpu = cuda.to_gpu(image)				 
		INPUT = chainer.Variable(image_gpu)	 
		label_gpu = cuda.to_gpu(label)			         
		ANSER = chainer.Variable(label_gpu)

		pred = model(INPUT)
		pred = F.softmax(pred[0])		
		pred = cuda.to_cpu(pred[0].data)	    
		ind = np.argmax(pred.data, axis=0)		    

		r = ind.copy()						
		g = ind.copy()						#gにindのコピーしたものを入れる
		b = ind.copy()						#bにindのコピーしたものを入れる
	
		r_gt = label.copy()					#r_gtにlabelのコピーしたものを入れる
		g_gt = label.copy()					#g_gtにlabelのコピーしたものを入れる
		b_gt = label.copy()					#b_gtにlabelのコピーしたものを入れる
		
		for l in range(0, label_colours.shape[0]):		
			r[ind==l] = label_colours[l,0]
			g[ind==l] = label_colours[l,1]
			b[ind==l] = label_colours[l,2]
			r_gt[label==l] = label_colours[l,0]
			g_gt[label==l] = label_colours[l,1]
			b_gt[label==l] = label_colours[l,2]

		rgb = np.zeros((ind.shape[0], ind.shape[1], 3)) 
		rgb[:,:,0] = r					
		rgb[:,:,1] = g						 
		rgb[:,:,2] = b						

		rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3)) 
		rgb_gt[:,:,0] = r_gt
		rgb_gt[:,:,1] = g_gt
		rgb_gt[:,:,2] = b_gt

		image = np.squeeze(image[0,:,:,:])
		image = (image.swapaxes(0, 2)).swapaxes(0, 1)		#配列の要素の入れ替え
		

		cv2.imwrite("./output_psp/" + str(num+1) + ".bmp", rgb)  #出力画像
		cv2.imwrite("./true/" + str(num+1) + ".bmp", rgb_gt)  #教師画像
		cv2.imwrite("./image/" + str(num+1) + ".png", image)  #入力画像

		print(str(num+1) + "image was saved.")		#saveしたメッセージを出力

def color_array():				#色の指定
	color = np.array([[0, 0, 0],    #その他
                          [255, 170, 85],    #人
                          [0, 170, 170],     #車
                          [170, 85, 85],     #障害物
                          [255, 0, 170],     #白線
			  [0, 85, 170],	     #空
			  [85, 0, 170],	     #車止め
			  [85, 85, 85],	     #道路
			  [255, 255, 0],     #歩道
#			  [0, 85, 0],	     #段差
			  [255, 0, 85]])     #バイク
	color = color[:, ::-1]
	return color

if __name__ == '__main__':
	color = color_array()
	print(color)
	test(color)
