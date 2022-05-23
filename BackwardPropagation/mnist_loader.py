import os
import struct
import numpy as np


def load_data(path, kind="train"):
    '''
    从path文件夹中加载mnist数据，类型为kind（train或t10k）
    返回二维数组images（即每行一张28*28的像素点图片），一维数组labels（对应图片的数字）
    '''
    labels_path = os.path.join(path, "%s-labels.idx1-ubyte" % kind)
    images_path = os.path.join(path, "%s-images.idx3-ubyte" % kind)
    with open(labels_path, "rb") as label_file:
        magic, n = struct.unpack('>II', label_file.read(8))
        print("label:", magic, n)
        labels = np.fromfile(label_file, dtype=np.uint8)
    with open(images_path, "rb") as image_file:
        magic, num, row, col = struct.unpack('>IIII', image_file.read(16))
        print("image:", magic, num, row, col)
        images = np.fromfile(image_file, dtype=np.uint8)
        images.shape = (num, row * col)
    return images, labels


def standardize(images, labels):
    '''
    将load_data得到的数组转换为神经网络训练用的向量组
    返回train_data的list，即（X，Y）型的二元列向量组
    '''
    n = len(labels)
    images.shape = (n, 28*28)
    dataX, dataY = [], []
    for i in range(n):
        X = images[i]/256
        Y = np.array([int(j == labels[i]) for j in range(10)])
        dataX.append(X)
        dataY.append(Y)
    dataX, dataY = np.array(dataX), np.array(dataY)
    return dataX, dataY
