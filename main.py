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

def extract_points(xi, yi, line):
    if line.found == True:
        sidex, sidey, line.found = line.found_search(xi, yi)
    if line.found == False:
        sidex, sidey, line.found = line.blind_search(xi, yi, binary)

    sidey = np.array(sidey).astype(np.float32)
    sidex = np.array(sidex).astype(np.float32)
    
    side_fit = np.polyfit(sidey, sidex, 2)
    sidex_int, side_top = line.get_intercepts(side_fit)

    line.x_int.append(sidex_int)
    line.top.append(side_top)
    sidex_int = np.mean(line.x_int)
    side_top = np.mean(line.top)
    line.lastx_int = sidex_int
    line.last_top  = side_top

    sidex = np.append(sidex, sidex_int)
    sidey = np.append(sidey, 720)
    sidex = np.append(sidex, side_top)
    sidey = np.append(sidey, 0)

    sidex, sidey = line.sort_vals(sidex, sidey)
    line.X = sidex
    line.Y = sidey

    side_fit = np.polyfit(sidey, sidex, 2)
    line.fit0.append(side_fit[0])
    line.fit1.append(side_fit[1])
    line.fit2.append(side_fit[2])
    side_fit = [np.mean(line.fit0),
                np.mean(line.fit1),
                np.mean(line.fit2)]

    side_fitx = side_fit[0]*sidey**2 + side_fit[1]*sidey + side_fit[2]
    line.fitx = side_fitx

    if line.pos == 'left':
        pts = np.array([np.flipud(np.transpose(np.vstack([line.fitx, line.Y])))])
    else:
        pts = np.array([np.transpose(np.vstack([line.fitx, line.Y]))])

    return pts


src = np.float32([[ 490,  482], [ 810,  482], [1250,  720], [  40,  720]])
dst = np.float32([[   0,    0], [1280,    0], [1250,  720], [  40,  720]])

M    = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

Left  = Line('left')
Right = Line('right')

cap = cv2.VideoCapture('/home/kihyeon/Study/python/lane_finding/project_video.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is False:
        print('read fail')
        break

    warped = birds_eye(frame, M)
    binary = apply_thresholds(warped)
    #thresholds = binary_image(warped, binary)

    x, y = np.nonzero(np.transpose(binary))
    left_pts   = extract_points(x, y, Left)
    right_pts  = extract_points(x, y, Right)

    warp_zero = np.zeros_like(binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts = np.hstack((left_pts, right_pts))

    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(255,0,0), thickness=20)
    cv2.fillPoly(color_warp, np.int_(pts), (34, 255, 34))
    newwarp = cv2.warpPerspective(color_warp, Minv, (frame.shape[1], frame.shape[0]))
    result = cv2.addWeighted(frame, 1, newwarp, 0.5, 0)

    cv2.imshow('frame', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
