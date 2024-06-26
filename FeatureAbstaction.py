from math import *
from tkinter.filedialog import *
from tkinter import *

import cv2
import numpy as np
import scipy
from PIL import ImageTk, Image
from cv2 import *
from scipy import ndimage
from scipy import signal
from Feature import ridge_segment, ridge_orient, ridge_freq, ridge_filter, VThin, HThin
import dataLoader
from dataLoader import writeData
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt

def enhancement(image):
    blksze = 16
    thresh = 0.1
    normim, mask = ridge_segment(image, blksze, thresh)  # normalise the image and find a ROI
    # imshow('origin', image)
    # imshow("norm", normim)

    gradientsigma = 1
    blocksigma = 7
    orientsmoothsigma = 7
    orientim = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma)  # find orientation of every pixel
    # imshow("orient", orientim)

    blksze = 38
    windsze = 5
    minWaveLength = 5
    maxWaveLength = 15
    freq, medfreq = ridge_freq(normim, mask, orientim, blksze, windsze, minWaveLength,
                               maxWaveLength)  # find the overall frequency of ridges
    # imshow("freq", freq)

    freq = medfreq * mask
    kx = 0.65
    ky = 0.65
    newim = ridge_filter(normim, orientim, freq, kx, ky)  # create gabor filter and do the actual filtering
    # imshow("new",newim)

    image = 255 * (newim >= -3)
    return image


def thinning(image):
    # iXihua = cv.CreateImage(cv.GetSize(image), 8, 1)
    # cv.Copy(image, iXihua)
    num = 10
    array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]

    # img = img.tolist()
    # img.copyTo(iXihua)
    for i in range(num):
        VThin(image, array)
        HThin(image, array)
    # print(img)
    return image


def feature(img):
    # print(img.shape)
    # endpoint1 = img
    # endpoint2 = img
    features = []
    # endpoint = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if img[i, j] == 0:  # 像素点为黑
                m = i
                n = j

                eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1], img[m, n + 1],
                              img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]

                if sum(eightField) / 255 == 7:  # 黑色块1个，端点

                    # 判断是否为指纹图像边缘
                    if sum(img[:i, j]) == 255 * i or sum(img[i + 1:, j]) == 255 * (w - i - 1) or sum(
                            img[i, :j]) == 255 * j or sum(img[i, j + 1:]) == 255 * (h - j - 1):
                        continue
                    canContinue = True
                    # print(m, n)
                    coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1], [m + 1, n - 1],
                                  [m + 1, n], [m + 1, n + 1]]
                    for o in range(8):  # 寻找相连接的下一个点
                        if eightField[o] == 0:
                            index = o
                            m = coordinate[o][0]
                            n = coordinate[o][1]
                            # print(m, n, index)
                            break
                    # print(m, n, index)
                    for k in range(4):
                        coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1],
                                      [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                        eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1], img[m, n + 1],
                                      img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                        if sum(eightField) / 255 == 6:  # 连接点
                            for o in range(8):
                                if eightField[o] == 0 and o != 7 - index:
                                    index = o
                                    m = coordinate[o][0]
                                    n = coordinate[o][1]
                                    # print(m, n, index)
                                    break
                        else:
                            # print("false", i, j)
                            canContinue = False
                    if canContinue:

                        if n - j != 0:
                            if i - m >= 0 and j - n > 0:
                                direction = atan((i - m) / (n - j)) + pi
                            elif i - m < 0 and j - n > 0:
                                direction = atan((i - m) / (n - j)) - pi
                            else:
                                direction = atan((i - m) / (n - j))
                        else:
                            if i - m >= 0:
                                direction = pi / 2
                            else:
                                direction = -pi / 2
                        feature = []
                        feature.append(i)
                        feature.append(j)
                        feature.append("endpoint")
                        feature.append(direction)
                        features.append(feature)

                elif sum(eightField) / 255 == 5:  # 黑色块3个，分叉点
                    coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1], [m + 1, n - 1],
                                  [m + 1, n], [m + 1, n + 1]]
                    junctionCoordinates = []
                    junctions = []
                    canContinue = True
                    # 筛除不符合的分叉点
                    for o in range(8):  # 寻找相连接的下一个点
                        if eightField[o] == 0:
                            junctions.append(o)
                            junctionCoordinates.append(coordinate[o])
                    for k in range(3):
                        if k == 0:
                            a = junctions[0]
                            b = junctions[1]
                        elif k == 1:
                            a = junctions[1]
                            b = junctions[2]
                        else:
                            a = junctions[0]
                            b = junctions[2]
                        if (a == 0 and b == 1) or (a == 1 and b == 2) or (a == 2 and b == 4) or (a == 4 and b == 7) or (
                                a == 6 and b == 7) or (a == 5 and b == 6) or (a == 3 and b == 5) or (a == 0 and b == 3):
                            canContinue = False
                            break

                    if canContinue:  # 合格分叉点
                        # print(junctions)
                        print(junctionCoordinates)
                        print(i, j, "合格分叉点")
                        directions = []
                        canContinue = True
                        for k in range(3):  # 分三路进行
                            if canContinue:
                                junctionCoordinate = junctionCoordinates[k]
                                m = junctionCoordinate[0]
                                n = junctionCoordinate[1]
                                print(m, n, "start")
                                eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1],
                                              img[m, n + 1],
                                              img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                                coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1],
                                              [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                                canContinue = False
                                for o in range(8):
                                    if eightField[o] == 0:
                                        a = coordinate[o][0]
                                        b = coordinate[o][1]
                                        print("a=", a, "b=", b)
                                        # print("i=", i, "j=", j)
                                        if (a != i or b != j) and (
                                                a != junctionCoordinates[0][0] or b != junctionCoordinates[0][1]) and (
                                                a != junctionCoordinates[1][0] or b != junctionCoordinates[1][1]) and (
                                                a != junctionCoordinates[2][0] or b != junctionCoordinates[2][1]):
                                            index = o
                                            m = a
                                            n = b
                                            canContinue = True
                                            print(m, n, index, "支路", k)
                                            break
                                if canContinue:  # 能够找到第二个支路点
                                    for p in range(3):
                                        coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1],
                                                      [m, n + 1],
                                                      [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                                        eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1],
                                                      img[m, n - 1],
                                                      img[m, n + 1],
                                                      img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                                        if sum(eightField) / 255 == 6:  # 连接点
                                            for o in range(8):
                                                if eightField[o] == 0 and o != 7 - index:
                                                    index = o
                                                    m = coordinate[o][0]
                                                    n = coordinate[o][1]
                                                    print(m, n, index, "支路尾")
                                                    # print(m, n, index)
                                                    break
                                        else:
                                            # print("false", i, j)
                                            canContinue = False
                                if canContinue:  # 能够找到3个连接点

                                    if n - j != 0:
                                        if i - m >= 0 and j - n > 0:
                                            direction = atan((i - m) / (n - j)) + pi
                                        elif i - m < 0 and j - n > 0:
                                            direction = atan((i - m) / (n - j)) - pi
                                        else:
                                            direction = atan((i - m) / (n - j))
                                    else:
                                        if i - m >= 0:
                                            direction = pi / 2
                                        else:
                                            direction = -pi / 2
                                    # print(direction)
                                    directions.append(direction)
                        if canContinue:
                            feature = []
                            feature.append(i)
                            feature.append(j)
                            feature.append("bifurcation")
                            feature.append(directions)
                            features.append(feature)
    # print(features)
    return features


def gray_features(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])  # 得到全局直方图统计数据
    h, w = img.shape
    hist = hist / (h * w)  # 将直方图归一化为0-1，概率的形式

    grayFeature = []
    # 灰度平均值
    mean_gray = 0
    for i in range(len(hist)):
        mean_gray += i * hist[i]
    grayFeature.append(mean_gray[0])

    # 灰度方差
    var_gray = 0
    for i in range(len(hist)):
        var_gray += hist[i] * ((i - mean_gray) ** 2)
    grayFeature.append(var_gray[0])

    # 能量sum(hist[i])**2
    max_ = np.max(hist)
    min_ = np.min(hist)
    histOne = (hist - min_) / (max_ - min_)
    # #求解能量
    energy = 0
    for i in range(len(histOne)):
        energy += histOne[i] ** 2
    grayFeature.append(energy[0])

    # 熵
    he = 0
    for i in range(len(hist)):
        if hist[i] != 0:  # 当等于0时，log无法进行计算，因此只需要计算非0部分的熵即可
            he += hist[i] * (np.log(hist[i]) / (np.log(2)))
    he = -he
    grayFeature.append(he[0])

    # 灰度对比度
    con = np.max(img) - np.min(img)
    grayFeature.append(con)
    return grayFeature


def writeFeatures(bufenfeatures, quanjufeatures, label=0):
    # 写入指纹特征，其中包括endpoint，bifurcation数量，位置信息
    FeatureMat = []
    print(len(bufenfeatures))
    endpointUVs = []
    endpointDirection = []
    bifurcationUVs = []
    bifurcationDirection = []
    for i in range(len(bufenfeatures)):
        endpointUV = []
        bifurcationUV = []
        if bufenfeatures[i][2] == 'endpoint':
            endpointUV.append(bufenfeatures[i][0])
            endpointUV.append(bufenfeatures[i][1])
            endpointUVs.append(endpointUV)
            endpointDirection.append(bufenfeatures[i][3])
        elif bufenfeatures[i][2] == 'bifurcation':
            bifurcationUV.append(bufenfeatures[i][0])
            bifurcationUV.append(bufenfeatures[i][1])
            bifurcationUVs.append(bifurcationUV)
            bifurcationDirection.append(np.var(bufenfeatures[i][3]) + 10 * pi)

    FeatureMat.append(len(endpointUVs))

    if len(bifurcationUVs) == 0:
        FeatureMat.append(0)
    else:
        FeatureMat.append(len(bifurcationUVs))

    FeatureMat.append(np.mean(endpointDirection))
    FeatureMat.append(np.var(endpointDirection))
    if len(bifurcationDirection) == 0:
        FeatureMat.append(0)
        FeatureMat.append(0)
    else:
        FeatureMat.append(np.mean(bifurcationDirection))
        FeatureMat.append(np.var(bifurcationDirection))

    FeatureMat.extend(quanjufeatures)

    FeatureMat.append(label)

    dictionaryname = './DB3_feature/'
    filename = 'features3.txt'

    dataLoader.writeData(dictionaryname + filename, FeatureMat)
    print('write in ', filename, 'successfully!')


def readimg(filename):
    image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转灰度图
    row, col = np.shape(image)
    aspect_ratio = np.double(row) / np.double(col)
    new_row = 200  # randomly selected number
    new_col = new_row / aspect_ratio
    image = cv2.resize(image, (int(new_col), int(new_row)))
    return image


def WriteDB3_B():
    dictionary = './DB3_B/'
    filename = '10'
    for i in range(1, 4):
        for j in range(1, 8):
            quanjuFeatures = []
            jubuFeatures = []
            img_path = dictionary + filename + str(i) + '_' + str(j) + '.tif'
            image = readimg(img_path)
            quanjuFeatures = gray_features(image)
            img1 = enhancement(image)
            img2 = thinning(img1)
            jubuFeatures = feature(img2)
            writeFeatures(jubuFeatures, quanjuFeatures, i)


def Writetest():
    dictionary = './DB3_B/'
    filename = '10'
    for i in range(1, 4):
        j = 8
        quanjuFeatures = []
        jubuFeatures = []
        img_path = dictionary + filename + str(i) + '_' + str(j) + '.tif'
        image = readimg(img_path)
        quanjuFeatures = gray_features(image)
        img1 = enhancement(image)
        img2 = thinning(img1)
        jubuFeatures = feature(img2)
        writeFeatures(jubuFeatures, quanjuFeatures, i)


def gray2rgb(imggray):
    # 原图 R G 通道不变，B 转换回彩图格式
    R = imggray
    G = imggray
    B = ((imggray) - 0.299 * R - 0.587 * G) / 0.114
    shape = imggray.shape
    shape = list(shape)
    shape.append(3)
    grayRgb = np.zeros(shape)

    grayRgb[:, :, 2] = B
    grayRgb[:, :, 0] = R
    grayRgb[:, :, 1] = G

    return grayRgb


if __name__ == '__main__':

    WriteDB3_B()

    # img_path = askopenfilename(initialdir='./DB3_B/', title='选择待识别图片',
    #                            filetypes=[("tif", "*.tif"), ("jpg", "*.jpg"), ("png", "*.png")])
    # if img_path:
    #     print(img_path)
    #     image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    #     print(type(image))
    #     if len(image.shape) > 2:
    #         img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转灰度图
    # rows, cols = np.shape(img)
    # aspect_ratio = np.double(rows) / np.double(cols)
    # new_rows = 200  # randomly selected number
    # new_cols = new_rows / aspect_ratio
    # img = cv2.resize(img, (int(new_cols), int(new_rows)))
    #
    # cv2.imshow('original pic', img)
    # img_enhanced = enhancement(img)
    #
    # cv2.imshow('enhanced pic', img_enhanced / 255)
    #
    # img_thinning = thinning(img_enhanced)
    # cv2.imshow('thinning img', img_enhanced / 255)
    #
    # grayFeature = gray_features(img)
    # bufenFeatures = feature(img_thinning)
    #
    # BGRimage = gray2rgb(img_thinning)
    #
    # for m in range(len(bufenFeatures)):
    #     if bufenFeatures[m][2] == "endpoint":
    #         cv2.circle(BGRimage, (bufenFeatures[m][1], bufenFeatures[m][0]), 3, (0, 0, 255), 1)
    #     else:
    #         cv2.circle(BGRimage, (bufenFeatures[m][1], bufenFeatures[m][0]), 3, (255, 0, 0), 1)
    #
    # cv2.imshow('FeatureImage', BGRimage / 255)
    # cv2.waitKey(0)
    #
    # print(grayFeature)
    # print(bufenFeatures)


    # writeFeatures(bufenFeatures, grayFeature)
