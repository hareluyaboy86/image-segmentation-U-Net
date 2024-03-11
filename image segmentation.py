# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:36:58 2023

@author: HZHONG
"""

# import json

# poly=open('C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/polygons.jsonl')

import pandas as pd
poly = pd.read_json(path_or_buf='C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/polygons.jsonl', lines=True)

import numpy as np
import json
# def load_ann_regions(maxr):
#     """
#     Load maxr annotation regions (for debug).
#     If maxr < 0, all records will be loaded (final).
#     """
#     reg_ann_df = pd.DataFrame({'iid':[], 'type':[], 'coord':[]})
#     ids = []
#     i = 0
#     last_id = None
#     with open('C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/polygons.jsonl', 'r') as f:
#         for line in f:
#             antn = json.loads(line)
#             img_id = antn['id']
#             img_antn = antn['annotations']
            
#             if(last_id != img_id):
#                 ids.append(img_id)
#                 i += 1
#                 last_id = img_id
            
#             if(i == maxr):
#                 break
                
#             for ant in img_antn:
#                 tp = ant['type']
#                 crd = ant['coordinates']
#                 reg_ann_df.loc[len(reg_ann_df.index)] = np.array([img_id, tp, crd], dtype = object)
#     return (ids, reg_ann_df)
from skimage.io import imread, imshow

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
    
d = res[500]
img_id = d["id"]
path = f"C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/train/{img_id}.tif"
img = imread(path)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(img)
for e in d["annotations"]:
    if e["type"] == "blood_vessel":
        coordinates = e["coordinates"]
        xs, ys = get_cartesian_coords(coordinates)
        ax.plot(xs, ys, c="red")
fig.show()


############################################################

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
#%matplotlib inline
    
    
# direction    
base_dir = 'C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature'
annote_dir = f'{base_dir}/polygons.jsonl'
images_dir = f'{base_dir}/train' 
test_images_dir = f'{base_dir}/test'

# image size
image_size = 512
input_image_size = (512,512)


images_listdir = os.listdir(images_dir)
random_images = np.random.choice(images_listdir, size = 9, replace = False)




def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    return img

# plot random img
rows = 3
cols = 3
fig, ax = plt.subplots(rows, cols, figsize = (12,12))

for i, ax in enumerate(ax.flat):
    if i < len(random_images):
        img = read_image(f"{images_dir}/{random_images[i]}")
        #print(img.shape)
        ax.set_title(f"{random_images[i]}")
        ax.imshow(img)
        ax.axis('off')
        
        
with open(annote_dir) as file:
    for i,line in enumerate(file):
        json_data = json.loads(line)
        # Process the individual JSON object here
        print(i,json_data.keys())
        print(i,json_data['annotations'][0].keys())
        print(i,json_data['annotations'][0]['type'])
        if i==3:
            break        

# check first annotation
with open(annote_dir) as annotes:
    for i,annote in enumerate(annotes):
        if i<1:
            # print(json.loads(annote))
            annotej=json.loads(annote)#str to dict
            print(annotej.keys())#error
            print(annotej['id'])
            print(len(annotej['annotations']))
            print(annotej['annotations'][0].keys())
            print(annotej['annotations'][0]['type'])



annotejs=[]
with open(annote_dir) as annotes:
    for i,annote in enumerate(annotes):
        annotej=json.loads(annote)#str to dict
        annotejs+=[annotej]
        
MASKS=np.zeros((1,image_size, image_size, 1), dtype=bool)
IMAGES=np.zeros((1,image_size, image_size, 3),dtype=np.uint8)        


# # preprocess mask and image
# for j,annotej in enumerate(annotejs[0:501]):## how many images to be used, the smaller, the faster
#     #print(j)
#     masks = np.zeros((len(annotej["annotations"]), image_size, image_size, 1), dtype=bool)

#     idi=annotej['id']
#     path=os.path.join(images_dir,idi+'.tif')
#     image=read_image(path)

# # polygon coordinate to Mask
#     for i,annotation in enumerate(annotej["annotations"]):
#         segmentation = annotation["coordinates"]
#         cur_mask = imantics.Polygons(segmentation).mask(*input_image_size).array
#         cur_mask = np.expand_dims(resize(cur_mask, (image_size, image_size), mode='constant', preserve_range=True), 2)
#         masks[i] = masks[i] | cur_mask 
        
#     mask2=np.sum(masks, axis=0) 
#     mask2_ex = np.expand_dims(mask2, axis=0)
#     image_ex = np.expand_dims(image, axis=0)

#     MASKS=np.vstack([MASKS, mask2_ex])
#     IMAGES=np.vstack([IMAGES, image_ex])
    
    
# images=np.array(IMAGES)[1:501]
# masks=np.array(MASKS)[1:501]
# print(images.shape,masks.shape)

# # split data to train and validation
# images_train, images_test, masks_train, masks_test = train_test_split(
#     images, masks, test_size=0.3, random_state=42)

# # check if train matchs masks
# print(len(images_train), len(masks_train))


# Data Generator class used in training to avoid memmory constraints

class DataGen(Sequence):
    def __init__(self, annotejs, images_dir, batch_size=8, image_size=512):
        self.annotejs = annotejs
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
    
    def __load__(self, annotej):
        # Preparing the mask array
        masks = np.zeros((self.image_size, self.image_size, 1), dtype=bool)

        idi=annotej['id']
        image_path = os.path.join(self.images_dir, idi+'.tif')

        # Read and resize image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        # Iterate over each annotation
        for i,annotation in enumerate(annotej["annotations"]):
            # print(annotation["type"])
            if annotation["type"]=="blood_vessel":
                segmentation = annotation["coordinates"]
                cur_mask = imantics.Polygons(segmentation).mask(*input_image_size).array
                cur_mask = np.expand_dims(resize(cur_mask, (self.image_size, self.image_size), mode='constant', preserve_range=True), 2)
                # cur_mask = np.expand_dims(cv2.resize(cur_mask, (self.image_size, self.image_size)), 2)
                masks = masks | cur_mask 
        
        ## Normalize 
        image = image/255.0
        # Binarize the mask
        masks = masks > 0

        return image, masks
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.annotejs):
            self.batch_size = len(self.annotejs) - index*self.batch_size

        # Select the right batch
        batch_annotejs = self.annotejs[index*self.batch_size : (index+1)*self.batch_size]

        images = []
        masks  = []

        # Load each image and mask pair
        for annotej in batch_annotejs:
            _img, _mask = self.__load__(annotej)
            images.append(_img)
            masks.append(_mask)

        # Return the batch
        return np.array(images), np.array(masks)
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.annotejs)/float(self.batch_size)))
    
    
    
# Build Model

def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def U_net_plus_plus(input_shape, num_classes):
    inputs = tf.keras.layers.Input(input_shape)

    # First part of U-Net
    c1 = conv_block(inputs, 64)
    drop1=tf.keras.layers.Dropout(0.1)(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(drop1)

    c2 = conv_block(p1, 128)
    drop2=tf.keras.layers.Dropout(0.1)(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(drop2)

    c3 = conv_block(p2, 256)
    drop3=tf.keras.layers.Dropout(0.1)(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(drop3)

    c4 = conv_block(p3, 512)
    drop4=tf.keras.layers.Dropout(0.1)(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(drop4)

    c5 = conv_block(p4, 1024)
    drop5=tf.keras.layers.Dropout(0.1)(c5)
    
    # Second part of U-Net++
    u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(drop5)
    u6 = tf.keras.layers.concatenate([u6, drop4])
    c6 = conv_block(u6, 512)

    u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, drop3])
    c7 = conv_block(u7, 256)

    u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, drop2])
    c8 = conv_block(u8, 128)

    u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, drop1], axis=3)
    c9 = conv_block(u9, 64)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)


    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

    return model


# Split the annotejs list into training and validation
annotejs_train, annotejs_val = train_test_split(annotejs, test_size=0.3, random_state=42)

# Instantiate the data generators
train_gen = DataGen(annotejs_train, images_dir, batch_size=2, image_size=512)
val_gen = DataGen(annotejs_val, images_dir, batch_size=2, image_size=512)

# Instantiate the model
model = U_net_plus_plus((512, 512, 3), 2)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Check model summary
model.summary()

# Save the model after every epoch
filepath="C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/model_{epoch:02d}-{val_accuracy:.2f}.h5"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)


# Train the model using the data generator
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=50,
                    callbacks=[checkpointer])

# Save the model after training
model.save("C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/model.h5")
# model.save("C:/Users/HZHONG/Documents/Python Scripts/hubmap-hacking-the-human-vasculature/model.keras")
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
# plt.savefig('C:/Users/HZHONG/Downloads/Los_Val.loss.png',dpi=600)
