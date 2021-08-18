import numpy as np
import cv2
import sys
import glob
from matplotlib import pyplot as plt
import time
from datetime import datetime

ksize = 7

# noise removal


def remove_noise(image, k):
    return cv2.GaussianBlur(image, (k, k), 0)
    # return cv2.medianBlur(image,k)

# dilation


def dilate(image, cnt_dilate):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=cnt_dilate)

# erosion


def erode(image, cnt_erode):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=cnt_erode)

# canny edge detection


def canny(image):
    return cv2.Canny(image, 100, 255)


def mark_red(img):
    src_bgr = img.copy()
    # Get current values from trackbars
    th1 = 0
    th2 = 144

    # Convert to HSV and split each channel
    src_hsv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV)
    src_h = src_hsv[..., 0]
    src_s = src_hsv[..., 1]
    src_v = src_hsv[..., 2]

    # Choose NOT only pixels whose HUE values are in our preferred range
    mask_h = (src_h > th1) & (src_h < th2)

    # Choose only pixels whose SAT values exceed our minimum requirement
    min_sat = 0.2
    mask_s = (src_s >= min_sat*255)

    # Combine two masks into one final mask
    mask_hs = (~mask_h) & mask_s

    # Change from 1-channel mask_hs to 3-channel mask_hs_3ch
    h, w = mask_hs.shape[:2]
    mask_hs_3ch = np.zeros((h, w, 3), dtype=mask_hs.dtype)
    mask_hs_3ch[..., 0] = mask_hs
    mask_hs_3ch[..., 1] = mask_hs
    mask_hs_3ch[..., 2] = mask_hs

    # Change from True-False image to 0-255 image
    mask_hs_3ch = (mask_hs_3ch.astype(np.uint8)) * 255

    return mask_hs_3ch


def mark_yellow(src_bgr):
    th1 = 20
    th2 = 30
    # Convert to HSV and split each channel
    src_hsv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV)
    src_h = src_hsv[..., 0]
    src_s = src_hsv[..., 1]
    src_v = src_hsv[..., 2]

    # Choose only pixels whose HUE values are in our preferred range
    mask_h = (src_h >= th1) & (src_h <= th2)
    print(mask_h.shape, mask_h.dtype)

    # Choose only pixels whose SAT values exceed our minimum requirement
    min_sat = 0.2
    mask_s = (src_s >= min_sat*255)
    print(mask_s.shape, mask_s.dtype)

    # Combine two masks into one final mask
    # mask_hs = mask_h & mask_s
    mask_hs = (src_h >= th1) & (src_h <= th2) & (src_s >= 0.2 * 255)
    print(mask_hs.shape, mask_hs.dtype)

    # Change from 1-channel mask_hs to 3-channel mask_hs_3ch
    h, w = mask_hs.shape[:2]
    mask_hs_3ch = np.zeros((h, w, 3), dtype=mask_hs.dtype)
    mask_hs_3ch[..., 0] = mask_hs
    mask_hs_3ch[..., 1] = mask_hs
    mask_hs_3ch[..., 2] = mask_hs
    print(mask_hs_3ch.shape, mask_hs_3ch.dtype)

    # Change from True-False image to 0-255 image
    mask_hs_3ch = (mask_hs_3ch.astype(np.uint8)) * 255

    # return src_segment
    return mask_hs_3ch


def mark_blue(src_bgr):
    th1 = 100
    th2 = 135
    # Convert to HSV and split each channel
    src_hsv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV)
    src_h = src_hsv[..., 0]
    src_s = src_hsv[..., 1]
    src_v = src_hsv[..., 2]

    # Choose only pixels whose HUE values are in our preferred range
    mask_h = (src_h >= th1) & (src_h <= th2)
    print(mask_h.shape, mask_h.dtype)

    # Choose only pixels whose SAT values exceed our minimum requirement
    min_sat = 0.2
    mask_s = (src_s >= min_sat*255)
    print(mask_s.shape, mask_s.dtype)

    # Combine two masks into one final mask
    # mask_hs = mask_h & mask_s
    mask_hs = (src_h >= th1) & (src_h <= th2) & (src_s >= 0.2 * 255)
    print(mask_hs.shape, mask_hs.dtype)

    # Change from 1-channel mask_hs to 3-channel mask_hs_3ch
    h, w = mask_hs.shape[:2]
    mask_hs_3ch = np.zeros((h, w, 3), dtype=mask_hs.dtype)
    mask_hs_3ch[..., 0] = mask_hs
    mask_hs_3ch[..., 1] = mask_hs
    mask_hs_3ch[..., 2] = mask_hs
    print(mask_hs_3ch.shape, mask_hs_3ch.dtype)

    # Change from True-False image to 0-255 image
    mask_hs_3ch = (mask_hs_3ch.astype(np.uint8)) * 255

    # return src_segment
    return mask_hs_3ch


def mark_white(img):
    global_thres = 60
    img_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_white = cv2.threshold(img.copy(), global_thres,
    #                           255, cv2.THRESH_BINARY)[1]
    # img_white = remove_noise(img_white, 13)
    return img_white


def get_contours(mark):
    # candy
    img_candy = canny(mark)

    # Find all contours
    contours, hierarchy = cv2.findContours(img_candy,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    contours_mark = []
    for i, c in enumerate(contours):
        if (cv2.contourArea(c) > 100):
            x, y, w, h = cv2.boundingRect(c)
            if h > w:
                w = h
            if w >= 15 and w <= 80:
                contours_mark.append(c)
    return contours_mark


def get_mark(img):
    # img = cv2.imread(dir)
    # print('img', img.shape)
    # resize
    # h, w, _ = img.shape
    # if h > w:
    #     img = cv2.resize(img, (540, round(540 * h / w)))
    # else:
    #     img = cv2.resize(img, (round(540 * w / h), 540))
    # rgb for present
    img_org = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = remove_noise(img, 3)
    # yellow
    img_yellow = mark_yellow(img.copy())
    # red
    img_red = mark_red(img.copy())
    # blue
    img_blue = mark_blue(img.copy())
    # white
    img_white = mark_white(img.copy())
    # contours
    contours_red = get_contours(img_red)
    contours_yellow = get_contours(img_yellow)
    contours_blue = get_contours(img_blue)
    # contours_white = get_contours(img_white)
    contours = contours_red + contours_yellow + contours_blue
    # contours = contours_white

    img_box = img.copy()
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if h > w:
            w = h
        if w >= 15 and w <= 80:
            cropped_image = img.copy()
            cropped_image = cropped_image[y:y+w, x:x+w]
            cropped_image = cv2.resize(cropped_image, (256, 256))
            # check feature
            # Initiate FAST detector
            fast = cv2.FastFeatureDetector_create()
            img_kps = fast.detect(cropped_image, None)
            # if len(img_kps) > 0:
            now = datetime.now()
            filename = now.strftime("%H_%M_%S_") + str(i) + '.jpg'
            cv2.imwrite(
                'object-detection/output/20210818_cut4/'+filename, cropped_image)
            cv2.rectangle(img_box, (x, y), (x+w, y+w), (0, 0, 255), 2)

    img_box = cv2.drawContours(img_box.copy(), contours, -1,
                               (0, 255, 0),
                               thickness=1)

    return img_box
    # return img_mark
    # return img_white
    # return img_bright


test_path = "object-detection/dataset/video/input1_cut4.mp4"
test_files = glob.glob(test_path)
print('test_files', test_files)

dir = test_files[0]
video_capture = cv2.VideoCapture(dir)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_out = cv2.VideoWriter(
    'object-detection/output_video/demo_output_cut4.mp4', fourcc, 25, (frame_width, frame_height))

# Initialize the timer
n_frames = 0              # the number of frames
start_time = time.time()  # the start time

delay = 0       # delay in millisecond for better observation regarding tracking results
while True:
    retval, img = video_capture.read()
    img_org = img.copy()
    img = get_mark(img)
    # print(img.shape)
    # Draw a text label specifying the current fps
    n_frames += 1
    total_time = time.time() - start_time
    fps = n_frames / total_time
    h, w, c = img.shape
    caption_h = 40
    frame_fps = np.zeros((h+caption_h, w, c), dtype=np.uint8)
    frame_fps[caption_h:, ...] = img
    cv2.putText(frame_fps,                  # image to draw text
                f'fps={fps:.2f}',           # text to be written
                (10, 25),                    # position coordinates of the text
                cv2.FONT_HERSHEY_COMPLEX,   # font type
                0.75,                       # font scale
                (255, 255, 255),              # font BGR color
                1,                          # font thickness
                cv2.LINE_AA)
    cv2.imshow('Orginal', img_org)
    cv2.imshow('MultiTracker: ', frame_fps)

    video_out.write(img)
    # Wait 1 millisecond for any key press
    if (cv2.waitKey(max(1, delay)) == 27):       # press ESC to quit
        break
cv2.destroyAllWindows()
if video_capture.isOpened():
    video_capture.release()
