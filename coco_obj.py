# -*- coding: utf-8 -*-
"""
Created on Sat May 15 15:42:14 2021

@author: reena
"""

#pip install opencv-python
#pip install numpy
#pip install argparse

import numpy as np
import cv2
import os
import argparse
import time

agp=argparse.ArgumentParser()
#pass argument for image
#agp.add_argument('-i','--image',required=True, help='C:/Users/reena/Desktop/python learning files/YOLO/images')
agp.add_argument('-i','--image',required=True, help='path to input image')
#pass argument for probability
agp.add_argument('-c','--confidence',default=0.5,type=float,help='minimum probability to filter weak detections,IoU threshold')
#pass argument for threshold
agp.add_argument('-t','--threshold',default=0.3,type=float,help='Threshold when applying non-maxima supression')
args=vars(agp.parse_args())

#load coco class labels on which YOLO is trained
Cocolabelpath='yolo-coco/coco.names'
LABELS=open(Cocolabelpath).read().strip().split('\n')

#initializing color list for representing each possible class label,since we have 80 labels we will get 80 colors
COLORS=np.random.randint(0,255,size=(len(LABELS),3),dtype='uint8')

#load coco yolo weights and and configuration files
Weights_path='yolo-coco/yolov3.weights'
config_path='yolo-coco/yolov3.cfg'

#load yolo object detector which is trained on coco dataset that has 80 classes
net=cv2.dnn.readNetFromDarknet(config_path,Weights_path)

#load the input image and convert it to array
image=cv2.imread(args['image'])

#get the spatial dimensions of the image
(height, width)=image.shape[:2]

#determining the Yolo's output layers name 
layer_names=net.getLayerNames()
layer_names=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

#create a blob od input image
blob= cv2.dnn.blobFromImage(image,1/255.0,(224,224),swapRB=True,crop=False)

#set the blob as input
net.setInput(blob)

#set the layer output
layer_outputs=net.forward(layer_names)

#initialize the list of bounded boxes,confidence and classIDs
boxes=[]
confidences=[]
classIDs=[]

for output in layer_outputs:
    for detection in output:
        scores=detection[5:]
        classID=np.argmax(scores)
        confidence=scores[classID]
        if confidence > args['confidence']:
            box=detection[0:4]*np.array([width,height,width,height])
            (centerX,centerY,Width,Height)=box.astype('int')
            x=int(centerX-(Width/2))
            y=int(centerY-(Height/2))
            boxes.append([x,y,int(Width),int(Height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
# Applying non-max suppression to avoid overlapping of bounding boxes and supress weak bpounding boxes
Nonmax_suppress=cv2.dnn.NMSBoxes(boxes,confidences,args['confidence'],args['threshold'])
if len(Nonmax_suppress) > 0:
    for i in Nonmax_suppress.flatten():
        (x,y)=(boxes[i][0],boxes[i][1])
        (w,h)=(boxes[i][2],boxes[i][3])
        color=[int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
        text='{}:{:.4f}'.format(LABELS[classIDs[i]],confidences[i])
        cv2.putText(image,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
cv2.imshow('Image',image)
cv2.waitKey(0)        
