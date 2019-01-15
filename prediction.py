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

#------------------------------------------胡牌演算法
pais=list(range(1,10))+list(range(11,20))+list(range(21,30))+list(range(31,38,2))+list(range(41,46,2))

def hepai(a:list):
    '''Judge cards hepai. For excample:a=[1,2,3,4,4]

a=list,萬：1-9，條：11-19，餅：21-29，東西南北風：31,33,35,37，中發白：41,43,45。'''
    a=sorted(a)
    #print(a)

    #牌面檢查，是否屬於本函數規定的範圍內。
    #pais=list(range(1,10))+list(range(11,20))+list(range(21,30))+list(range(31,38,2))+list(range(41,46,2))
    #print(pais)
    for x in set(a):
        if a.count(x)>4:#某張牌的數量超過了4，是不正確的。
            return False
        if x not in pais: 
            #print('參數錯誤：輸入的牌型{}不在範圍內。\n萬：1-9，條：11-19，餅：21-29，東西南北風：31,33,35,37，中發白：41,43,45。'.format(x))
            return False

    #牌數檢查。
    if len(a)%3!=2:
        #print('和牌失敗：牌數不正確。')
        return False
    
    #是否有對子檢查。
    double=[]
    for x in set(a):
        if a.count(x)>=2:
            double.append(x)
    #print(double)
    if len(double)==0:
        #print('和牌失敗：無對子')
        return False
    
    #7對子檢查（由於不常見，可以放到後面進行判斷）
    #對子的檢查，特徵1：必須是14張；特徵2:一個牌型，有2張，或4張。特別注意有4張的情況。
    if len(a)==14:
        for x in set(a):
            if a.count(x) not in [2,4]:
                break
        else:
##            print('和牌:7對子。',a)
            return True

    #十三麼檢查。
    if len(a)==14:
        gtws=[1, 9, 11, 19, 21, 29, 31, 33, 35, 37, 41, 43, 45] #[1,9,11,19,21,29]+list(range(31,38,2))+list(range(41,46,2)) #用固定的表示方法，計算速度回加快。
        #print(gtws)
        for x in gtws:
            if 1<=a.count(x)<=2:
                pass
            else:
                break
        else:
            print('和牌：國土無雙，十三麼！')
            return True

    #常規和牌檢測。
    a1=a.copy()
    a2=[] #a2用來存放和牌後分組的結果。
    for x in double:
        #print('double',x)
        #print(a1[0] in a1 and (a1[0]+1) in a1 and (a1[0]+2) in a1)
        a1.remove(x)
        a1.remove(x)
        a2.append((x,x))
        for i in range(int(len(a1)/3)):
            #print('i-',i)
            if a1.count(a1[0])==3:
                #列表移除，可以使用remove,pop，和切片，這裏切片更加實用。
                a2.append((a1[0],)*3)
                a1=a1[3:]
                #print(a1)
            elif a1[0] in a1 and a1[0]+1 in a1 and a1[0]+2 in a1:#這裏注意，11,2222,33，和牌結果22,123,123，則連續的3個可能不是相鄰的。
                a2.append((a1[0],a1[0]+1,a1[0]+2))
                a1.remove(a1[0]+2)
                a1.remove(a1[0]+1)
                a1.remove(a1[0])
                #print(a1)

            else:
                a1=a.copy()
                a2=[]
                #print('重置')
                break
        else:
            #print('和牌成功,結果：',a2)
            return True
    
    #如果上述沒有返回和牌成功，這裏需要返回和牌失敗。
    else:
        #print('和牌失敗：遍歷完成。')
        return False



#------------------------------------------


def prediction(datapath):
    #------------------------------- TensorFlow預測
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
    #datapath = "tmp/2142045204.jpg"
    imageFile = datapath
    image = Image.open(imageFile)
    image = update_orientation(image)
    image = convert_to_opencv(image)
    image = resize_down_to_1600_max_dim(image)
    h, w = image.shape[:2]
    min_dim = min(w,h)
    max_square_image = crop_center(image, min_dim, min_dim)
    augmented_image = resize_to_256_square(max_square_image)
    network_input_size = 227
    augmented_image = crop_center(augmented_image, network_input_size, network_input_size)
    output_layer = 'loss:0'
    input_node = 'Placeholder:0'
    with tf.Session() as sess:
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
        predictions, = sess.run(prob_tensor, {input_node: [augmented_image] })    
    # Print the highest probability label
    highest_probability_index = np.argmax(predictions)
    print('Classified as: ' + labels[highest_probability_index])
    tf.reset_default_graph() 
    return labels[highest_probability_index]
        
  
#------------------------------- TensorFlow預測  





from PIL import Image
import numpy as np
import cv2
cap = cv2.VideoCapture(1)

while(True):
  # 從攝影機擷取一張影像
  ret, frame = cap.read()
  cv2.imshow('My Image1', frame)
  # 若按下 q 鍵則離開迴圈
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cv2.imwrite('output.jpg',frame)


x = os.system("mjMentor.exe "+ "output.jpg" + " cut_output.txt")
output=open("cut_output.txt",'r')
lines=output.readlines()
output_list = []
for line in lines:
    output_list.append(line)
    #print(output_list[0])
            
path_list = []
for path in output_list:
    path_list.append(path[2:-1])
            #print(path)
mj_list = []  
for path in path_list:
    print("path = "+path)
    if path.find("jpg") != -1:
        print(path)
        print(int(prediction(path)))
        #print("辨識結果:"+str(results.predictions[0].tag_name))
        #mj_list.append(int(results.predictions[0].tag_name))
print("牌辨識結果 : "+str(mj_list))

        
if(output_list[0] =="13\n"):  #聽模式
    print("聽模式")
    ting = ""
    for index in mj_list:
        ting += str(index) + " "
    ting_input = open('input.txt','w')
    ting_input.write(ting)
    ting_input.close()
    x = os.system("ting.exe")
    ting_output = open ('output.txt','r')
    ting_result=ting_output.readlines()
    ting_output.close()
            
    print("聽牌結果 = " + str(ting_result))
            #print("按下D以繼續")
            #while True:
            #  if ord(msvcrt.getch()) in [68, 100]:
            #      break
                  
elif (output_list[0] =="14\n"): #胡模式
    print("胡牌結果 = "+ str(hepai(mj_list)))
    count = 0
    if(hepai(mj_list)==True):
        count += 1
    print ("胡牌數:"+str(count))