# coding:utf-8:

import numpy as np
from glob import glob

import chainer
from chainer import optimizers
from chainer import serializers
from chainer import Variable
from chainer import cuda
import chainer.computational_graph as cg
#cuda関係
if chainer.cuda.available:
    xp = chainer.cuda.cupy
    chainer.cuda.get_device_from_id(0).use()
else:
    xp = np

import time
import random
import sys
import os
from os.path import join

from pspnet import Net
from augmentation import Augmentation
#data.pyの呼び出し
import data
from dataloader import loader


def train():
#--------------------------------------------------------------------------------- 
    #PATH
    img_dir = "./dataset/*.png"
    seg_dir = "./dataset/*.png
        
    # トレーニングデータの設定(保存先とかバッチサイズとか)
    SAVE_DIR = "./loss_seg"
    LOG_TXT = join(SAVE_DIR, "log.txt")
    HEIGHT = 480
    WIDTH = 640
    BATCH_SIZE = 7
    EPOCH_NUM = 600
    GPU_ID = 0

#---------------------------------------------------------------------------------

    # --- Create model object
    model = Net(n_class=10, input_size = (HEIGHT,WIDTH), n_blocks=[3, 4, 6, 3],
                pyramids=[6, 3, 2, 1], mid_stride=True, mean=np.array([123.68, 116.779, 103.939]),
                comm=None,
                pretrained_model=None, initialW=None)
    
    # --- Set CPU or GPU
    if GPU_ID >= 0:
        model.to_gpu(GPU_ID)

    # --- Optimizer Setting ()
    optimizer = optimizers.AdaDelta()
    optimizer.setup(model)

    #augmentation and Dataloader
    augmentation = Augmentation(img_dir, seg_dir, HEIGHT, WIDTH)
    load = loader()
    
    image = np.zeros([BATCH_SIZE, 3, HEIGHT, WIDTH], dtype=np.float32)
    label = np.zeros([BATCH_SIZE, HEIGHT, WIDTH], dtype=np.int32)

    # --- Data準備
    #data.pyの中にある関数に引数を渡して処理
    data_path = data.list_img_and_seg_path(img_dir, seg_dir)
    total_image = len(data_path[0])
    print("ALL DATA NUM:{:>7}".format(total_image))
    train_path = data_path
    print("TRAIN DATA NUM:{:>7}".format(len(train_path[0])))
    #画像枚数分の要素番号をもったlist(サブリスト) 
    sublist = list(range(0,total_image-BATCH_SIZE)) 


    # +++++++++++ Train Loop +++++++++++
    
    #初期化
    start = time.time()
    itr = 0
    loss = 0.0
    log_data_header = "Iteration, TrainLoss\n"
    with open(LOG_TXT, 'w') as f:
        f.write(log_data_header + "\n")
    
    for epoch in range(1, EPOCH_NUM + 1):
        print('{}\nepoch:{:>4}'.format('-' * 40, epoch))
        # サブリストをシャッフル
        random.shuffle(sublist)   
	#学習
        for i in range(0, len(sublist), BATCH_SIZE):
            # --- LOAD BATCH ---
            b = sublist[i]
            #data.pyにある関数の呼び出し(imageとseg画像の読み込み)
            for batch in range(BATCH_SIZE):
                image[batch,:,:,:], label[batch,:,:] = augmentation()
            image,label = load.Train(img_dir,seg_dir,BATCH_SIZE,HEIGHT,WIDTH)
            label=label // 20
            image_gpu = cuda.to_gpu(image)
            INPUT = chainer.Variable(image_gpu)
            label_gpu = cuda.to_gpu(label)
            ANSER = chainer.Variable(label_gpu)
            # --- NETWORK UPDATE ---
            model.cleargrads()
            aux, h = model(INPUT)
            loss = chainer.functions.softmax_cross_entropy(aux, ANSER)
            loss.backward() 
            optimizer.update()
            itr += 1
            #log.txtへitrとlossの書き込み
            print("iteration = "+str(itr))
            print("loss = "+str(loss.data))
            log_data_itr = str(itr)
            log_data_loss = str(loss.data)
            with open(LOG_TXT, 'a') as f:
                f.write(log_data_itr + "," + log_data_loss + "\n")
              
    # Save Weight for 10epoch
        if (epoch%100 == 0) and (epoch != 0):
            model.to_cpu()
            serializers.save_npz(join(SAVE_DIR, str(epoch)+"psp_weight.npz"), model)
            serializers.save_npz(join(SAVE_DIR, str(epoch)+"psp_state.npz"), optimizer)
            model.to_gpu()

if __name__ == '__main__':
    train()




