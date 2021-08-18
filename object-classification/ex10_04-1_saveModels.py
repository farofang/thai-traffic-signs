# Image Analytics and Computer Vision
# By Asst.Prof.Thitirat Siriborvornratanakul, Ph.D.

# Example 4-1
# This example shows how to save the dictionary for further use.
# Mostly it does the same things as Example 3


import pickle
import cv2
import numpy as np
import os
import glob


# STEP1: Load all images to disk and divide them to train/test sets (same codes as in Example 1)

# 30 positive samples randomly chosen from Stanford Cars Dataset
# https://ai.stanford.edu/~jkrause/cars/car_dataset.html
# This dataset contains 8,144 train images and 8,041 test images of 196 classes of cars
# Images dimensions are not the same

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


# STEP2: Create a dictionary/vocabulary/codebook of visual words  (same codes as in Example 1)

# SIFT
detector = cv2.xfeatures2d.SIFT_create()
extractor = cv2.xfeatures2d.SIFT_create()
# SURF
# detector = cv2.xfeatures2d.SURF_create()
# extractor = cv2.xfeatures2d.SURF_create()
# ORB
# detector = cv2.KAZE_create()
# extractor = cv2.KAZE_create()
matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})

# the number of clusters in K-mean (= the total number of visual words)
vocab_size = 40
bow_kmeans_trainer = cv2.BOWKMeansTrainer(vocab_size)
print(type(bow_kmeans_trainer))
# Use 8 positive samples and 8 negative samples from the train set to construct a vocab
for i in range(20):
    # Add one floating-point descriptor extracted from one positive image
    _, desc = detector.detectAndCompute(pos_train_images[i], None)
    bow_kmeans_trainer.add(np.float32(desc))

    # Add one floating-point descriptor extracted from one negative image
    _, desc = detector.detectAndCompute(neg_train_images[i], None)
    bow_kmeans_trainer.add(np.float32(desc))
vocab = bow_kmeans_trainer.cluster()

bow_extract = cv2.BOWImgDescriptorExtractor(extractor, matcher)
bow_extract.setVocabulary(vocab)


# STEP3: Create and train the SVM classifier  (same codes as in Example 1)

# a set of fixed-size feature vectors resulting from SIFT descriptor + BOVW
train_data = []
train_labels = []   # a set of class labels corresponding to train_data

# Convert all positive images to BOVW signatures and append it to the lists
for img in pos_train_images:
    keyp = detector.detect(img)

    # Let the bow extractor find descriptors, and match them to the dictionary
    bow_sig = bow_extract.compute(img, keyp)

    train_data.extend(bow_sig)
    train_labels.append(1)      # 1 = label of positive image

# Convert all negative images to BOVW signatures and append it to the lists
for img in neg_train_images:
    keyp = detector.detect(img)

    # Let the bow extractor find descriptors, and match them to the dictionary
    bow_sig = bow_extract.compute(img, keyp)

    train_data.extend(bow_sig)
    train_labels.append(-1)     # -1 = label of negative image

svm = cv2.ml.SVM_create()
svm.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(train_labels))


# STEP4: Save the dictionary and the SVM model

# Save the dictionary
with open('object-classification/dataset/model/my_dict.pickle', 'wb') as f:
    pickle.dump(vocab, f)
svm.save('object-classification/dataset/model/my_svm.xml')

print("End of this program.")
