#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 22:38:59 2017

@author: lijunyi
"""

import numpy as np
import os
import sys
import tensorflow as tf
from skimage.io import imread

# This is needed to display the images.
import csv
from glob import glob

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from utils import label_map_util

PATH_TO_CKPT = '../exported_graph_1/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'car_new_label_map.pbtxt')

NUM_CLASSES = 23

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '/media/bowenliu/Seagate JIONG/599/test'
image_files = glob(os.path.join(PATH_TO_TEST_IMAGES_DIR,'*/*_image.jpg'))

writer = csv.writer(open("./car", 'w'))
writer.writerow(['guid/image','N'])
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for j,file in enumerate(image_files):
        image = imread(file)
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        valid_car = (np.squeeze(scores)>0.16)
        car_num = valid_car.sum()
        invalid_car = 0
        boxes = np.squeeze(boxes)
	classes = np.squeeze(classes)
        boxes_center = boxes[0:car_num,0:2]
	valid_tag = np.ones((1,car_num))
        for i in range(car_num-1):
	    if(valid_tag[0,i]):
		if(classes[i]==1 or classes[i]==10 or classes[i]==12 or classes[i]==15 or classes[i]==16 or classes[i]==18 or classes[i]==17 or classes[i]==23):
			valid_tag[0,i]=0
		distance=np.sqrt(((boxes_center[i,:]-boxes_center[i+1:car_num,:])**2).mean(axis=1))
		#print(distance)
		valid_tag_1 = valid_tag[0,i+1:car_num]
            	valid_tag_1[distance<0.03]=0
		valid_tag[0,i+1:car_num]=valid_tag_1
	#print(car_num,invalid_car)
	valid_car=int(valid_tag.sum())
        file_1 = file.replace('_image.jpg', '')
        file_2 = file_1.replace(PATH_TO_TEST_IMAGES_DIR+'/', '')
        print(file_2,valid_car)
        writer.writerow([file_2,valid_car])
    
                
