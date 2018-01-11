## Description

The project implements Faster R-CNN for car detection.


## System setup

Ubuntu 16.04, Python3.5, OpenCV(Python3), Matlab, TensorFlow(Python3),
Numpy(Python3), Skimage(Python3), Scipy(Python3), Pillow(Python3), Cython(Python3).

## Inference

###Step1

Generate tensorflow object detection compatible file, i.e. tfrecord file with ‘create_my_tfrecord_1.py'.

###Step2

Upload corresponding files to google cloud:’car_new_label.pbkt,cloud.yml,faster_rcnn_resnet101_self.config’and the generated .tfrecord file.

###Step3

train model on google cloud and download the model after training.

###Step4

detect the number of cars with detection file.

## Dataset

 Dataset can be accessed from: [http://umich.edu/~fcav/rob599_dataset_deploy.zip]

## Group Members

* Junyi Li,
* Chen Wang,
* Jiong Zhu,
* Zhaoheng Zheng,
