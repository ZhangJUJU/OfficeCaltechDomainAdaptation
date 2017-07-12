# OfficeCaltechDomainAdaptation

## What is this ?
This dataset is part of the Computer Vision problematic consisting in making machines learn to detect the presence of an object in an image. Here, we want to learn a classification model that takes as input an image and return the category of the object it contains.

The Office Caltech dataset contains four different domains: amazon, caltech10, dslr and webcam. These domains contain respectively 958, 1123, 295 and 157 images. Each image contains only one object among a list of 10 objects: backpack, bike, calculator, headphones, keyboard, laptop, monitor, mouse, mug and projector.

With this benchmark dataset in Domain Adaptation, we repeatedly take one of the four domains as Source domain S and one of the three remaining as target T. The aim is then to learn to classify images with the data from S to correctly classify the images in T.

## What is available in this repository ?
In addition to the images, we also give features that were extracted from the images to describe them. We give different sets of features that describe all the images in the corresponding folder.

We propose some code in python3 to show how to evaluate the benchmark. What is usually evaluated with this benchmark are Domain Adaptation algorithms. We provide code for a few of them.

## Dependencies
Python3 and some python3 libraries:
 - numpy
 - scipy
 - sklearn

## Example of execution

Program launched by executing the main.py script with python:
```
python3 main.py
```

For each adaptation problem among the 12 possible, each adaptation algorithm chosen at the beginning of the file is applied. Then are reported the mean accuracy and standard deviation. Results:
```
Feature used:  surf
Number of iterations:  10
Adaptation algorithms used:   NA  SA
A->C ..........   4.75s
     22.2  2.0 NA
     33.8  1.8 SA
A->D ..........   2.82s
     23.4  2.6 NA
     32.5  4.1 SA
A->W ..........   2.67s
     24.5  1.9 NA
     31.6  2.2 SA
C->A ..........   4.50s
     20.6  2.9 NA
     36.4  2.3 SA
C->D ..........   2.27s
     20.7  2.8 NA
     37.4  3.8 SA
C->W ..........   2.66s
     20.1  4.8 NA
     29.6  3.6 SA
D->A ..........   2.99s
     27.5  1.7 NA
     32.0  1.5 SA
D->C ..........   3.07s
     25.1  1.7 NA
     30.5  1.3 SA
D->W ..........   1.50s
     53.4  3.5 NA
     78.4  2.1 SA
W->A ..........   4.03s
     23.4  1.2 NA
     31.5  1.5 SA
W->C ..........   4.23s
     18.8  0.7 NA
     28.9  1.1 SA
W->D ..........   1.65s
     52.4  2.4 NA
     82.9  2.1 SA

Mean results:
     27.7  2.3 NA
     40.5  2.3 SA
```

By modifying the feature used in the script with CaffeNet features:
```
Feature used:  CaffeNet4096
Number of iterations:  10
Adaptation algorithms used:   NA  SA
A->C ..........  22.60s
     72.4  2.4 NA
     78.7  1.2 SA
A->D ..........   8.03s
     76.4  3.5 NA
     83.1  1.7 SA
A->W ..........   9.44s
     67.3  3.1 NA
     79.7  2.3 SA
C->A ..........  20.47s
     81.2  1.9 NA
     86.6  1.3 SA
C->D ..........   8.15s
     76.4  5.0 NA
     79.7  2.4 SA
C->W ..........   9.62s
     69.7  5.3 NA
     77.8  3.8 SA
D->A ..........  14.98s
     69.7  2.5 NA
     83.0  0.8 SA
D->C ..........  16.04s
     66.9  2.5 NA
     75.4  1.2 SA
D->W ..........   7.87s
     91.5  1.9 NA
     97.2  1.5 SA
W->A ..........  19.23s
     67.5  2.4 NA
     81.8  1.3 SA
W->C ..........  22.99s
     60.7  1.1 NA
     73.3  0.7 SA
W->D ..........   8.27s
     96.8  1.5 NA
     99.6  0.4 SA

Mean results:
     74.7  2.7 NA
     83.0  1.5 SA
```

and with GoogleNet features:
```
Feature used:  GoogleNet1024
Number of iterations:  10
Adaptation algorithms used:   NA  SA
A->C ..........   5.51s
     84.7  1.3 NA
     85.8  1.0 SA
A->D ..........   1.91s
     88.4  1.6 NA
     87.4  2.6 SA
A->W ..........   2.46s
     82.2  3.3 NA
     84.3  3.3 SA
C->A ..........   5.19s
     90.8  0.8 NA
     91.6  1.1 SA
C->D ..........   2.39s
     87.1  2.5 NA
     87.6  2.0 SA
C->W ..........   3.03s
     86.3  2.0 NA
     88.7  2.3 SA
D->A ..........   4.10s
     83.3  2.5 NA
     88.4  1.5 SA
D->C ..........   5.75s
     77.1  2.5 NA
     84.0  1.7 SA
D->W ..........   1.89s
     97.9  0.9 NA
     98.2  1.0 SA
W->A ..........   5.25s
     86.4  1.0 NA
     89.8  1.2 SA
W->C ..........   5.57s
     78.9  1.0 NA
     83.6  0.8 SA
W->D ..........   1.93s
     99.1  0.4 NA
     99.3  0.4 SA

Mean results:
     86.8  1.7 NA
     89.1  1.6 SA
```
