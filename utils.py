#coding=utf-8
import numpy as np
from PIL import Image
from pylab import *
import tensorflow.compat.v1 as tf
import cv2
from PIL import ImageFilter
import random
def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

def load_images(file):
    im = Image.open(file)
    h= im.size[0]-im.size[0]%4
    w= im.size[1]-im.size[1]%4
    return (np.array(im,dtype="float32") / 255.0)[0:w,0:h,:]
def white_world(image):
    mean_RGB = np.mean(np.mean(image,axis=0),axis=0)

    ratio = np.clip(mean_RGB/mean_RGB.min(),1.0,1.1)
    white_image = image
    white_image[:,:,0] =  white_image[:,:,0] / ratio[0]
    white_image[:,:,1] =  white_image[:,:,1] / ratio[1]
    white_image[:,:,2] =  white_image[:,:,2] / ratio[2]

    return white_image

def save_images(filepath, result_1, result_2 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')

def save_images2(filepath, result_1, result_2 = None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis = 1)

    im = Image.fromarray(np.clip(np.abs(cat_image * 255.0), 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')

def histeq(im,nbr_bins = 256):
    """对一幅灰度图像进行直方图均衡化"""
    #计算图像的直方图
    #在numpy中，也提供了一个计算直方图的函数histogram(),第一个返回的是直方图的统计量，第二个为每个bins的中间值
    imhist,bins = histogram(im.flatten(),nbr_bins,normed= True)
    cdf = imhist.cumsum()   #
    cdf = 1.0*cdf / cdf[-1]
    #使用累积分布函数的线性插值，计算新的像素值
    im2 = interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape)
def adapthisteq(im,NumTiles=8,ClipLimit=0.01,NBins=256):
    mri_img = im * 255.0;
    mri_img = mri_img.astype('uint8')

    r, c, h = mri_img.shape
    if h==1:
        temp = mri_img
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        mri_img = clahe.apply(temp)
    elif h==3: 
        for k in range(h):
            temp = mri_img[:,:,k]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            mri_img[:,:,k] = clahe.apply(temp)
    return  (np.array(mri_img, dtype="float32") / 255.0).reshape(im.shape)
def adapthisteq2(im,NumTiles=8,ClipLimit=0.01,NBins=256):
    mri_img = im * 255.0;
    mri_img = mri_img.astype('uint8')

    a,r, c, h = mri_img.shape
    if h==1:
        temp = np.squeeze(mri_img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        mri_img = clahe.apply(temp)
    elif h==3: 
        for k in range(h):
            temp = mri_img[:,:,k]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            mri_img[:,:,k] = clahe.apply(temp)
    return  (np.array(mri_img, dtype="float32") / 255.0).reshape(im.shape)

def mainFilter(I,winSize=(20,20)):
    mean_I = cv2.blur(I, winSize)      # I的均值平滑
    return mean_I[:,:,np.newaxis]
def meanFilter(I,winSize=(3,3)):
    # print(I.shape)
    I_ = np.squeeze(I)
    mean_I = cv2.GaussianBlur(I_, winSize,0,0)      # I的均值平滑
    # mean_I=I
    return mean_I.reshape(I.shape)
def medianBlur(I,winSize=3):
    # print(I.shape)
    I_ = np.squeeze(I)
    mean_I = cv2.medianBlur(I_, winSize)      # I的均值平滑
    # mean_I=I
    return mean_I.reshape(I.shape)


def guideFilter(I, p, winSize=(5,5), eps=0.01):

    #mean_I = I.filter(ImageFilter.BLUR,(3,3))

    mean_I = cv2.blur(I, winSize)      # I的均值平滑
    mean_p = cv2.blur(p, winSize)      # p的均值平滑

    mean_II = cv2.blur(I * I, winSize) # I*I的均值平滑
    mean_Ip = cv2.blur(I * p, winSize) # I*p的均值平滑

    var_I = mean_II - mean_I * mean_I  # 方差
    
    cov_Ip = mean_Ip - mean_I * mean_p # 协方差

    a = cov_Ip / (var_I + eps)         # 相关因子a
    b = mean_p - a * mean_I            # 相关因子b

    mean_a = cv2.blur(a, winSize)      # 对a进行均值平滑
    mean_b = cv2.blur(b, winSize)      # 对b进行均值平滑

    weight = mean_a

    q = mean_a * np.squeeze(I) + mean_b
    return q
def sigmoid(x):
    return 1/(1+np.exp(-x))

def gasuss_noise(image, mean=0, var=0.01):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    # image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = 0.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    # out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out
def gasuss_noise2(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    # image = np.array(image/255, dtype=float)
    output=[]
    # var=var*random.random()
    for idx in range(0,len(image)):
        noise = np.random.normal(mean, var ** 0.5, image[idx].shape)
        out = image[idx] + noise
        if out.min() < 0:
            low_clip = 0.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        output.append(out)
        # out = np.uint8(out*255)
        #cv.imshow("gasuss", out)
        # print(out.shape)
        # print(idx)
    return output
def gasuss_noise3(image, mean=0, var=0.01):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    # image = np.array(image/255, dtype=float)
    output=[]
    for idx in range(0,len(image)):
        noise = np.random.normal(mean, var ** 0.5, image[idx].shape)
        out = meanFilter(image[idx]) + noise
        if out.min() < 0:
            low_clip = 0.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        output.append(out)
        # out = np.uint8(out*255)
        #cv.imshow("gasuss", out)
    return output
def max_channel_index(im):
    max_channel_value = np.max(im,axis=3,keepdims=True)
    index_matrix = np.concatenate((max_channel_value,max_channel_value,max_channel_value),axis=3)
    im2=np.int64(index_matrix>=-0.000001)
    return im2.reshape(im.shape)