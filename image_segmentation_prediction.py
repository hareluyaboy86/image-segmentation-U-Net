# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:20:11 2023

@author: HZHONG
"""
import os #
import cv2 #
import pandas as pd #
import jsonlines #
import numpy as np #
from typing import List, Tuple #

import matplotlib.pyplot as plt
import tensorflow as tf #
import json #
import os #
import imantics  #
from PIL import Image #
from skimage.transform import resize 
import random #
from sklearn.model_selection import train_test_split #
from tensorflow.keras.utils import Sequence #
from keras.callbacks import ModelCheckpoint #

model = tf.keras.models.load_model('C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/model_07-0.96.h5')

# model.load_weights('C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/model.h5')

base_dir = 'C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature'
annote_dir = f'{base_dir}/polygons.jsonl'
# images_dir = f'{base_dir}/train' 

# image_size = 512
# input_image_size = (512,512)


def get_cartesian_coords(coords, img_height=512):
    coords_array = np.array(coords).squeeze()
    xs = coords_array[:, 0]
    ys = coords_array[:, 1]
    
    return xs, ys

import json
with open("C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/polygons.jsonl") as f:
    data = f.read()
    
    
res = []
for file in data.splitlines():
    d = json.loads(file)
    res.append(d)

for i in range(10):
    d = res[i]
    img_id = d["id"]
    path = f"C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/train/{img_id}.tif"
    # img = imread(path)
    
    
    # predict single image
    import keras.utils as image
    # new_img = image.load_img(path,target_size=(512, 512))
    # img = image.img_to_array(new_img)
    # img = np.expand_dims(img, axis=0)
    # img = img/255
    
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    # image=image[:,:,0][:,:,None]
    image=np.expand_dims(image, axis=0)
    image = image/255.0
    # classes = model.predict_classes(img)
    # np.argmax(model.predict(img),axis=1).astype("int32")
    from skimage.io import imread, imshow
    # import cv2
    
    import csv
    predict=model.predict(image)
    predict=predict.squeeze()
    
    # plt.imshow(predict[0,:,:,0]>0.3, cmap='gray')
    
    # file=open("C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/prediction.csv","w")
    # writer=csv.writer(file)
    # for row in predict:
    #     writer.writerow(row)
    # file.close()
    
    # predict=(predict[0,:,:,0]>0.95).astype(np.uint8)
    # plt.imshow(predict, cmap='gray')
    # prediction_image=predict.reshape(512,512,1)
    # plt.imshow(prediction_image, cmap='gray')
    # plt.imshow(new_img)
    
    
    predict_area=predict.squeeze()
    predict_area=predict_area>0.5
    # plt.imshow(predict_area, cmap='gray')
    
    contours, _ = cv2.findContours((predict.reshape(512,512,1)>0.5).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = [np.array(polygon).squeeze() for polygon in contours]
    
    img=imread(path)
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(img)
    ax[0].set_title('Prediction',fontweight="bold", size=20)
    for e in d["annotations"]:
        if e["type"] == "blood_vessel":
            coordinates = e["coordinates"]
            for i in range(len(polygons)):
                if len(polygons[i].shape)!=1:
                    x=polygons[i][:,0].tolist()
                    y=polygons[i][:,1].tolist()
            # x, y = get_cartesian_coords(coordinates)
                    ax[0].plot(x, y, c="red")
    # ax[0].title.set_text('Prediction')
    # fig.show()
    img=imread(path)
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax[1].imshow(img)
    ax[1].set_title('True',fontweight="bold", size=20)
    # plt.title("Prediction")
    for e in d["annotations"]:
        if e["type"] == "blood_vessel":
            coordinates = e["coordinates"]
            xs, ys = get_cartesian_coords(coordinates)
            ax[1].plot(xs, ys, c="red")
    # fig.show()
    # ax[1].title.set_text('True')
    fig.suptitle('Plot ID:'+str(img_id),fontsize=40)
    fig.savefig("C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/prediction/"+str(img_id)+".png",dpi=400)

# x=polygons[7][:,0].tolist()
# y=polygons[7][:,1].tolist()

# x=[]
# y=[]
# for i in range(len(polygons)):
#     # print(i)
#     if len(polygons[i].shape)!=1:
#         # x_=print(*polygons[i][:,0])
#         # x.append(x_)
#         x.append(polygons[i][:,0].tolist())
#         y.append(polygons[i][:,1].tolist())
# polygons = []

# from itertools import chain
# x=list(chain.from_iterable(x))
# y=list(chain.from_iterable(y))

# x=[]
# y=[]
# for obj in contours:
#     # coords = []
#     obj=np.array(obj).squeeze()
    
#     for i in range(len(obj)):
#         print(i)
#         x.append(int(obj[i][0]))
#         y.append(int(obj[i][1]))
    # for point in obj:
    #     # print(point)
    #     x.append(int(point[0]))
    #     y.append(int(point[1]))
        # x.append(int(point[0][0]))
        # y.append(int(point[0][1]))

# x=np.array(x)
# y=np.array(y)


# x=polygons[8][:,0].tolist()
# y=polygons[8][:,1].tolist()
# img=imread(path)
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.imshow(img)
# for e in d["annotations"]:
#     if e["type"] == "blood_vessel":
#         coordinates = e["coordinates"]
#         for i in range(len(polygons)):
#             if len(polygons[i].shape)!=1:
#                 x=polygons[i][:,0].tolist()
#                 y=polygons[i][:,1].tolist()
#         # x, y = get_cartesian_coords(coordinates)
#                 ax.plot(x, y, c="red")
# fig.show()

#     # polygons.append(coords)
# # model2=model.load_weights('C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/model.h5')
# # predict2=model2.predict(image)

# # prediction2_image=predict2.reshape(512,512,1)
# # plt.imshow(prediction2_image, cmap='gray')
# # np.delete(predict, 1,2,2)
# # image.array_to_img(predict)



# img=imread(path)
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.imshow(img)
# for e in d["annotations"]:
#     if e["type"] == "blood_vessel":
#         coordinates = e["coordinates"]
#         xs, ys = get_cartesian_coords(coordinates)
#         ax.plot(xs, ys, c="red")
# fig.show()

# tf.keras.utils.array_to_img(predict)   
   
# model.load_weights('C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/model.h5', by_name=True)
# img=imread(path)
# results = model.detect([img], verbose=1) 
 
annotejs=[]
with open(annote_dir) as annotes:
    for i,annote in enumerate(annotes):
        annotej=json.loads(annote)#str to dict
        annotejs+=[annotej]