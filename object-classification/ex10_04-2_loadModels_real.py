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
dataset_path = "object-classification/dataset/20210818_cut2/*.jpg"
files = glob.glob(dataset_path)
for f in files:
    try:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)      # load as 1-channel image
        im = cv2.resize(im, (256, 256))
        pos_images.append(im)
    except:
        print("")

# Train/Test split:
# - Train = 20 positive , 20 negative
# - Test = 10 positive , 10 negative
n_pos = 20
pos_train_images = pos_images[:n_pos]
pos_test_images = pos_images[n_pos:]

# STEP2: Load a dictionary/vocabulary/codebook of visual words

# SIFT
detector = cv2.xfeatures2d.SIFT_create()
extractor = cv2.xfeatures2d.SIFT_create()
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

for i, img in enumerate(pos_test_images):
    total += 1
    keyp = detector.detect(img)
    bow_sig = bow_extract.compute(img, keyp)
    # print(bow_sig == None)
    try:
        output = svm.predict(bow_sig)
        if output[1] == 1:
            tp += 1
            cv2.putText(img, 'True', (10, 30),
                        font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif output[1] == -1:
            fn += 1
            cv2.putText(img, 'False',
                        (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    except:
        fn += 1
        cv2.putText(img, 'False - None', (10, 30),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Test set: Positive sample', img)
    print('Press any key at the "Test set: Positive sample" window to continue ...')
    cv2.waitKey(0)

cv2.destroyAllWindows()
