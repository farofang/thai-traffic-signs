# Image Analytics and Computer Vision
# By Asst.Prof.Thitirat Siriborvornratanakul, Ph.D.

# Example 4-2
# This example shows how to load the dictionary and the SVM model from Example 2-1 and do classification.
# Mostly this example does the same things as in Example 3.


import pickle
import cv2
import numpy as np
import os
import glob


# STEP1: Load all test images to disk (same codes as in Example 1)

pos_images = []
dataset_path = "object-classification/dataset/positive/*.jpg"
files = glob.glob(dataset_path)
for f in files:
    try:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)      # load as 1-channel image
        im = cv2.resize(im, (256, 256))
        pos_images.append(im)
    except:
        print("")

# 30 negative samples randomly chosen from UKBench dataset
# https://archive.org/details/ukbench
# This dataset contains 10,200 images used for building and evaluating Content-based Image Retrieval systems
# All images are 640x480 in dimension
neg_images = []
dataset_path = "object-classification/dataset/negative/*.jpg"
files = glob.glob(dataset_path)
for f in files:
    try:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)      # load as 1-channel image
        im = cv2.resize(im, (256, 256))
        neg_images.append(im)
    except:
        print("")

# Train/Test split:
# - Train = 20 positive , 20 negative
# - Test = 10 positive , 10 negative
n_pos = 20
pos_train_images = pos_images[:n_pos]
pos_test_images = pos_images[n_pos:]
neg_train_images = neg_images[:n_pos]
neg_test_images = neg_images[n_pos:]


# STEP2: Load a dictionary/vocabulary/codebook of visual words

# SIFT
detector = cv2.xfeatures2d.SIFT_create()
extractor = cv2.xfeatures2d.SIFT_create()
# SURF
#detector = cv2.xfeatures2d.SURF_create()
#extractor = cv2.xfeatures2d.SURF_create()
# ORB
#detector = cv2.KAZE_create()
#extractor = cv2.KAZE_create()
matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})


# Load the dictionary
with open('object-classification/dataset/model/my_dict.pickle', 'rb') as f:
    my_vocab = pickle.load(f)
print("Dictionary loaded successfully.")


bow_extract = cv2.BOWImgDescriptorExtractor(extractor, matcher)
bow_extract.setVocabulary(my_vocab)


# STEP3: Load a pre-trained SVM classifier

svm = cv2.ml.SVM_load('object-classification/dataset/model/my_svm.xml')
print("SVM model loaded successfully.")


# STEP4: Test the classifier with the test images (same codes as in Example 1)

font = cv2.FONT_HERSHEY_SIMPLEX
tp, tn, fp, fn, total = 0, 0, 0, 0, 0

print()
for i, img in enumerate(pos_test_images):
    total += 1
    keyp = detector.detect(img)
    bow_sig = bow_extract.compute(img, keyp)

    output = svm.predict(bow_sig)
    if output[1] == 1:
        tp += 1
        cv2.putText(img, 'True positive: Sign Detected', (10, 30),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    elif output[1] == -1:
        fn += 1
        cv2.putText(img, 'False negative: Sign Not Detected',
                    (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Test set: Positive sample', img)
    print('Press any key at the "Test set: Positive sample" window to continue ...')
    cv2.waitKey(0)

for i, img in enumerate(neg_test_images):
    total += 1
    keyp = detector.detect(img)
    bow_sig = bow_extract.compute(img, keyp)

    output = svm.predict(bow_sig)
    if output[1] == 1:
        fp += 1
        cv2.putText(img, 'False positive: Car Detected', (10, 30),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    elif output[1] == -1:
        tn += 1
        cv2.putText(img, 'True negative: Car Not Detected',
                    (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Test set: Negative sample', img)
    print('Press any key at the "Test set: Negative sample" window to continue ...')
    cv2.waitKey(0)

print('\n==== Conclusion ====')
print('Accuracy =', ((tp+tn)*100)/total, '%')
print('Error rate =', ((fp+fn)*100)/total, '%')
print('Precision =', ((tp)*100)/(tp+fp), '%')
print('Recall =', ((tp)*100)/(tp+fn), '%')

cv2.destroyAllWindows()
