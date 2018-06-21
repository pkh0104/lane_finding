import numpy as np
import cv2
from line import Line

def birds_eye(img, M):
    img_size = (img.shape[1], img.shape[0])
    wraped = cv2.warpPerspective(img, M, img_size)
    return wraped

def apply_thresholds(img):
    s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2]
    l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:,:,0]
    b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:,:,2]

    s_thresh_min = 180
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    b_thresh_min = 155
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    l_thresh_min = 225
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    combined_binary = np.zeros_like(s_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

    return combined_binary


def binary_image(img, binary):
    thresholds = np.zeros_like(img)
    thresholds[binary == 1] = [255, 255, 255]

    return thresholds 

cap = cv2.VideoCapture('/home/kihyeon/Study/python/lane_finding/project_video.mp4')

src = np.float32([[ 490,  482], [ 810,  482], [1250,  720], [  40,  720]])
dst = np.float32([[   0,    0], [1280,    0], [1250,  720], [  40,  720]])
M = cv2.getPerspectiveTransform(src, dst)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is False:
        print('read fail')
        break
   #hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    warped = birds_eye(frame, M)
    binary = apply_thresholds(warped)
    thresholds = binary_image(warped, binary)
    x, y = np.nonzero(np.transpose(binary))
    Left = Line('left')
    leftx, lefty, Left.found = Left.blind_search(x, y, binary)
    cv2.imshow('frame', thresholds)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
