# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:56:24 2018

@author: USER
"""

import tensorflow as tf
import os
from os import listdir
from os.path import isfile, isdir, join
import shutil
graph_def = tf.GraphDef()
labels = []
filename = "model.pb"
labels_filename = "labels.txt"
# Import the TF graph
with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Create a list of labels.
with open(labels_filename, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())


def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def crop_center(img,cropx,cropy):
    h, w = img.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if (h < 1600 and w < 1600):
        return image

    new_size = (1600 * w // h, 1600) if (h > w) else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation = cv2.INTER_LINEAR)

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation = cv2.INTER_LINEAR)

def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if (exif != None and exif_orientation_tag in exif):
            orientation = exif.get(exif_orientation_tag, 1)
            # orientation is 1 based, shift to zero based and flip/transpose based on 0-based values
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image




from PIL import Image
import numpy as np
import cv2
mypath = "../test/tmp/"
files = listdir(mypath)
# Load from a file
for f in files:  
  # 產生檔案的絕對路徑
    fullpath = join(mypath, f)
    if isfile(fullpath):
        imageFile = mypath + f
        image = Image.open(imageFile)

# Update orientation based on EXIF tags, if the file has orientation info.
        image = update_orientation(image)

# Convert to OpenCV format
        image = convert_to_opencv(image)


# If the image has either w or h greater than 1600 we resize it down respecting
# aspect ratio such that the largest dimension is 1600
        image = resize_down_to_1600_max_dim(image)

# We next get the largest center square
        h, w = image.shape[:2]
        min_dim = min(w,h)
        max_square_image = crop_center(image, min_dim, min_dim)

# Resize that square down to 256x256
        augmented_image = resize_to_256_square(max_square_image)
        #augmented_image = image



# The compact models have a network size of 227x227, the model requires this size.
        network_input_size = 227

# Crop the center for the specified network_input_Size
        augmented_image = crop_center(augmented_image, network_input_size, network_input_size)

        #cv2.namedWindow("Image")
        #cv2.imshow("Image", augmented_image)
        #cv2.waitKey (0)
  
# These names are part of the model and cannot be changed.
        output_layer = 'loss:0'
        input_node = 'Placeholder:0'
        #cv2.imwrite ("output1.jpg",augmented_image )
        with tf.Session() as sess:
            prob_tensor = sess.graph.get_tensor_by_name(output_layer)
            predictions, = sess.run(prob_tensor, {input_node: [augmented_image] })
            #predictions, = sess.run(prob_tensor, {input_node: [image] })
    
# Print the highest probability label
        highest_probability_index = np.argmax(predictions)
        #print('Classified as: ' + labels[highest_probability_index])
        print("辨識結果:"+str(labels[highest_probability_index]))
        
        if (labels[highest_probability_index] == "1"):
            
            shutil.move(mypath+f,mypath+"/1/"+f)
            
        if (labels[highest_probability_index] == "2"):
            shutil.move(mypath+f,mypath+"/2/"+f)
            
        if (labels[highest_probability_index] == "3"):
            shutil.move(mypath+f,mypath+"/3/"+f)
            
        if (labels[highest_probability_index] == "4"):
            shutil.move(mypath+f,mypath+"/4/"+f)
            
        if (labels[highest_probability_index] == "5"):
            shutil.move(mypath+f,mypath+"/5/"+f)
            
        if (labels[highest_probability_index] == "6"):
            shutil.move(mypath+f,mypath+"/6/"+f)
            
        if (labels[highest_probability_index] == "7"):
            shutil.move(mypath+f,mypath+"/7/"+f)
            
        if (labels[highest_probability_index] == "8"):
            shutil.move(mypath+f,mypath+"/8/"+f)
            
        if (labels[highest_probability_index] == "9"):
            shutil.move(mypath+f,mypath+"/9/"+f)
        