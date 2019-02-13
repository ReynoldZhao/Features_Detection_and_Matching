# Feature Detection and Matching

## Introduction

The goal of **feature detection** and **matching** is to identify a pairing between a point in one image and a corresponding point in another image. These correspondences can then be used to stitch multiple images together into a panorama.

## Features

* Feature detection: Identify points of interest in the image using the Harris corner detection method**
* Feature description: Come up with a *descriptor* for the feature centered at each interest point. Implement a simplified version of the MOPS descriptor.  Compute an 8x8 oriented patch sub-sampled from a 40x40 pixel region around the feature. Come up with a transformation matrix which transforms the 40x40 rotated window around the feature to an 8x8 patch rotated so that its keypoint orientation points to the right. Normalize the patch to have zero mean and unit variance. If the variance is very close to zero (less than 10-5 in magnitude) then just return an all-zeros vector to avoid a divide by zero error.
* Feature Matching: Have detected and described your features, the next step is to write code to match them.
  * Sum of squared differences (SSD): This is the the squared Euclidean distance between the two feature vectors.
  * The ratio test: Find the closest and second closest features by SSD distance. The ratio test distance is their ratio (i.e., SSD distance of the closest feature match divided by SSD distance of the second closest feature match).
* Complete features descriptor that has attribute Scale Invariant Feature Transform (SIFT) 

## Structure

| Name                   | Function                                                     |
| ---------------------- | ------------------------------------------------------------ |
| resources/             | available images to detect and match                         |
| src/benchmarck.py      | calculate ROC curve for the root image and matched image     |
| src/features.py        | include algorithms of features detecting, descriptor and matching |
| src/featuresUI.py      | GUI for debugging and executing the process of features detecting and matching |
| src/tests.py           | test whether function in features.py is same with task's command |
| src/transformations.py | transform images and enable the function scale invariant     |

## Usages

### Requirements

* Linux / Windows / MacOS
* python 2.7 / python 3.5
* cv2
* numpy
* pandas

### Compilation

``` python
cd python test.py
cd python featuresUI.py
```

## Examples

### Features detecting and descriptor

![](C:\Users\57844\Desktop\CV作业\实验原文件\实验1\Exp1_Hybrid_Images\resources\cat.jpg)

![](C:\Users\57844\Desktop\CV作业\实验原文件\实验1\Exp1_Hybrid_Images\resources\dog.jpg)

### Features matching

![](C:\Users\57844\Desktop\CV作业\实验原文件\实验1\Exp1_Hybrid_Images\resources\hybrid.png)