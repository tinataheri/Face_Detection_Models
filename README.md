# Introduction

Computer vision has been a cornerstone of AI-based technologies since its inception. With the widespread availability of digital cameras, leveraging the vast amount of data they provide has become essential. This has led to the emergence of various sub-domains within computer vision, with primary objectives including detection, classification, segmentation, and recognition of images or specific parts thereof. Face recognition, a leading technology in computer vision, encompasses detection, alignment, representation, and verification.

Utilizing a state-of-the-art detector is crucial in the initial stages of a face recognition pipeline. Thus, exploring face detection models tailored to specific goals is as vital as other stages. These models can be categorized based on their approach to classifying an object as a face and accurately locating its coordinates.

In early models, classification relied on Haar-feature based cascades, which identify human faces by leveraging common features shared by certain facial areas. Histogram of Oriented Gradients (HOG) is another model that extracts features before classification using methods like Support Vector Machines (SVM). However, manual feature determination in these models leads to reduced accuracy with challenging images.

Recent years have seen the rise of deep learning models, addressing issues such as difficult poses, varied expressions, or occlusion. Convolutional Neural Network (CNN)-based models and Single Shot Detector (SSD) models dominate proposed papers, incorporating convolutional kernels for improved performance.

This paper is a comparative study of four widely used face detection methods provided by leading machine learning libraries.

## Overview of Models' Architectures

This section provides insights into the structure and theory behind each model, where images undergo machine learning algorithms such as CNN, SVM, or SSD. These algorithms result in the precise localization of existing faces through bounding boxes, relevant class labels, and confidence scores.

### 2.1 Histogram of Oriented Gradients

The HOG method, originally proposed for human body detection, is predominantly used for face detection. It involves extracting feature vectors and feeding them into a classification algorithm like SVM. These features consist of histograms of gradient directions, which help in retaining useful information while discarding unnecessary details from the image.

![Histogram of Oriented Gradients object detector chain](link to the image)

### 2.2 Multi-task Cascaded Convolutional Networks (MTCNN)

MTCNN is a framework capable of both face detection and alignment. It comprises three convolutional networks: P-Net, R-Net, and O-Net. The architecture utilizes an image pyramid to detect faces of various sizes. Non-Maximum Suppression is employed to filter overlapping bounding boxes, enhancing accuracy at each stage.

![Pipeline of the proposed Multi-task detector](link to the image)

### 2.3 Max-Margin Object Detection

### 2.4 Single Shot Detection (SSD)

SSD employs a base network for feature extraction, followed by multiscale feature map blocks to generate anchor boxes of different sizes. Grid-based division of images facilitates object detection within each grid cell using anchor boxes.

![SSD model structure](link to the image)

## Face Detection Datasets

### 3.1 Face Detection Dataset (FDDB)

FDDB comprises 5171 labeled faces in 2845 images, capturing diverse appearances and difficulties such as occlusion and low resolution. Each face is represented by an ellipse, aiding in performance evaluation using metrics like ROC curves.

### 3.2 WIDER FACE

WIDER FACE dataset contains 32,203 images with 393,703 faces, offering variability in scale, pose, and occlusion. It adopts evaluation metrics similar to PASCAL VOC dataset, essential for real-world system requirements.

## Evaluation Metrics

Performance evaluation involves comparing predictions with accurate face region coordinates. Commonly used methods include the Intersection over Union (IOU) metric and evaluation curves like ROC and Precision-Recall curves. These provide valuable insights into classifier performance across different thresholds and imbalanced classes.
