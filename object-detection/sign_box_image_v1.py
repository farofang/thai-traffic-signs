import numpy as np
import cv2
import sys
import glob
from matplotlib import pyplot as plt
import time

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


def get_mark(img):
    # img = cv2.imread(dir)
    print('img', img.shape)
    # resize
    # h, w, _ = img.shape
    # if h > w:
    #     img = cv2.resize(img, (360, round(360 * h / w)))
    # else:
    #     img = cv2.resize(img, (round(360 * w / h), 360))
    # print('img-resize', img.shape)
    # rgb for present
    img_org = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # gray
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    img_gray = remove_noise(img_gray, 3)

    img_candy = canny(img_gray)

    # Find all contours
    contours, hierarchy = cv2.findContours(img_candy,
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)

    contours_mark = []
    max_area = 0
    max_length = 0
    img_box = img.copy()
    for c in contours:
        if cv2.contourArea(c) > max_area:
            max_area = cv2.contourArea(c)
        if cv2.arcLength(c, False) > max_length:
            max_length = cv2.arcLength(c, False)
        if (cv2.contourArea(c) > 100 and cv2.contourArea(c) < 500) or (cv2.arcLength(c, False) > 500 and cv2.arcLength(c, False) < 1500):
            contours_mark.append(c)
            x, y, w, h = cv2.boundingRect(c)
            print('w,h', w, h, w/h)
            is_square = False

            if w/h > 0.8 and w/h < 1.2:
                is_square = True
            if is_square and w > h:
                cv2.rectangle(img_box, (x, y), (x+w, y+w), (0, 0, 255), 2)
            elif is_square and h >= w:
                cv2.rectangle(img_box, (x, y), (x+h, y+h), (0, 0, 255), 2)
        # print(cv2.contourArea(c),cv2.arcLength(c,False))
    print('max', max_area, max_length)

    img_box = cv2.drawContours(img_box.copy(), contours_mark, -1,
                               (0, 255, 0),
                               thickness=1)

    return img_box


test_path = "object-detection/dataset/video/input4.mp4"
test_files = glob.glob(test_path)
print('test_files', test_files)

dir = test_files[0]
capture = cv2.VideoCapture(dir)

# Initialize the timer
n_frames = 0              # the number of frames
start_time = time.time()  # the start time

delay = 50       # delay in millisecond for better observation regarding tracking results
while True:
    retval, img = capture.read()

    img = get_mark(img)
    print(img.shape)
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

    cv2.imshow('MultiTracker: ', frame_fps)

    # Wait 1 millisecond for any key press
    if (cv2.waitKey(max(1, delay)) == 27):       # press ESC to quit
        break
cv2.destroyAllWindows()
if capture.isOpened():
    capture.release()
