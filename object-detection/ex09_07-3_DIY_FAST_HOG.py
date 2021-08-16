# Image Analytics and Computer Vision
# By Asst.Prof.Thitirat Siriborvornratanakul, Ph.D.

# Example 7-3:
# A combination of FAST keypoints and HOG descriptors


import numpy as np
import cv2


def get_HOG_descriptors(img_gray, kps, win_size=(64, 64)):
    result = []
    for i, kp in enumerate(kps):
        # Read (x,y) coordinate from the current keypoint
        # https://docs.opencv.org/3.2.0/d2/d29/classcv_1_1KeyPoint.html
        x, y = kp.pt[0], kp.pt[1]

        # Create a win_size crop window for this keypoint
        left = int(x - (win_size[0]//2))
        right = int(x + (win_size[0]//2))
        if left < 0:
            left = 0
            right = left + win_size[0] + 1
        if right > img_gray.shape[1]-1:
            right = img_gray.shape[1]
            left = right - win_size[0] - 1
        top = int(y - (win_size[1]//2))
        bottom = int(y + (win_size[1]//2))
        if top < 0:
            top = 0
            bottom = top + win_size[1] + 1
        if bottom > img_gray.shape[0]-1:
            bottom = img_gray.shape[0]
            top = bottom - win_size[1] - 1

        # Crop the image with the window
        img_crop = img_gray[top:bottom, left:right]

        # Compute a HOG descriptor
        # Be careful, dimension of img_crop must match with winSize (as specified at hog initiation)
        desc = hog.compute(img_crop)

        # Add the descriptor to the resultant list
        result.append(desc)

    return np.asarray(result, np.float32)


# Initiate FAST detector
fast = cv2.FastFeatureDetector_create()

# Initiate HOG descriptor with default parameters
# size of the image sent to compute HOG
winSize = (16, 16)
# block size for block normalization
blockSize = tuple([i//2 for i in winSize])
blockStride = tuple([i//2 for i in blockSize])
# cell size (one cell for one 9-bin HOG)
cellSize = blockSize
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

# Initiate FLANN matching
# Since HOG descriptor is similar to SIFT/SURF descriptor, use the same matching technique
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


# Load a template image and find its keypoints and descriptors
dir = 'object-detection/starbucks.png'
template_bgr = cv2.imread(dir)
print(template_bgr)
template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
template_kps = fast.detect(template_gray, None)
template_descs = get_HOG_descriptors(template_gray, template_kps, winSize)


capture = cv2.VideoCapture(0)
while True:
    ret, cam_bgr = capture.read()
    if ret == False or cam_bgr is None:
        print('Cannot read camera input.')
        continue

    cam_gray = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2GRAY)

    # Find FAST keypoints with non-maximum supression
    cam_kps = fast.detect(cam_gray, None)

    # Find HOG descriptors
    cam_descs = get_HOG_descriptors(cam_gray, cam_kps, winSize)

    # Find two best matches
    matches = flann.knnMatch(template_descs,   # query set of descriptors
                             cam_descs,       # train set of descriptors
                             k=2)             # only find two best matches for each query descriptor

    # Ratio test for eliminating poor matches
    matchesMask = [[0, 0] for i in range(len(matches))]
    counter = 0
    for i, mn in enumerate(matches):
        if (len(mn) == 2):                # prevent the case when only one match is found
            m = mn[0]
            n = mn[1]        # 'm' is the best match, 'n' is the second-best match
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
                counter += 1
    print('The number of good matches =', counter)

    # Draw matching result
    # cv2.drawMatchesKnn expects list of lists as matches.
    result = cv2.drawMatchesKnn(template_bgr, template_kps,
                                cam_bgr, cam_kps,
                                matches,
                                None,                          # output image
                                # use random colors to draw the matching line
                                matchColor=(-1, -1, -1),
                                # use random colors to draw each keypoint
                                singlePointColor=(-1, -1, -1),
                                matchesMask=matchesMask,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('FAST+HOG: Feature Matching', result)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()
capture.release()
