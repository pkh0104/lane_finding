import numpy as np
import cv2
from line import Line
from moviepy.editor import VideoFileClip

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

    b_thresh_min = 150
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    l_thresh_min = 220
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

   # combined_binary = np.zeros_like(s_binary)
    combined_binary = np.zeros_like(b_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

    return combined_binary


def binary_image(img, binary):
    thresholds = np.zeros_like(img)
    thresholds[binary == 1] = [255, 255, 255]

    return thresholds 

def extract_points(binary, xi, yi, line):
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
    line.count += 1

    return pts

#src = np.float32([[ 490,  482], [ 810,  482], [1250,  720], [  40,  720]])
#dst = np.float32([[   0,    0], [1280,    0], [1250,  720], [  40,  720]])
#src = np.float32([[ 605,  500], [ 777,  500], [1250,  620], [  140,  670]])
src = np.float32([[ 610,  500], [ 742,  500], [1100,  630], [  240,  665]])
dst = np.float32([[   0,    0], [1280,    0], [1280,  715], [    0,  720]])

roi = np.float32([[  240,  665], [ 610,  500], [ 742,  500], [1100,  630] ])

M    = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

Left  = Line('left')
Right = Line('right')

def process_video(img):
    resized = cv2.resize(img, (1280,720), cv2.INTER_AREA)
    warped = birds_eye(resized, M)
    binary = apply_thresholds(warped)
  ##thresholds = binary_image(warped, binary)

    x, y = np.nonzero(np.transpose(binary))
    left_pts   = extract_points(binary, x, y, Left)
    right_pts  = extract_points(binary, x, y, Right)

    warp_zero = np.zeros_like(binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts = np.hstack((left_pts, right_pts))

   #cv2.polylines(resized, np.int_([roi]), isClosed=False, color=(255,0,0), thickness=2)
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness=20)
    cv2.fillPoly(color_warp, np.int_(pts), (34, 255, 34))
    newwarp = cv2.warpPerspective(color_warp, Minv, (resized.shape[1], resized.shape[0]))
    result = cv2.addWeighted(resized, 1, newwarp, 0.5, 0)

   #return resized
    return result
  ##return thresholds


video_output = '/home/kihyeon/Study/python/lane_finding/line_detected.mp4'
clip1 = VideoFileClip('/home/kihyeon/Study/python/lane_finding/20180531_075532_I_A.avi').subclip(0,35)
#clip1 = VideoFileClip('/home/kihyeon/Study/python/lane_finding/20180531_075357_I_A.avi').subclip(0,10)
#clip1 = VideoFileClip('/home/kihyeon/Study/python/lane_finding/20180531_075457_I_A.avi').subclip(0,4)
#clip1 = VideoFileClip('/home/kihyeon/Study/python/lane_finding/20180531_075614_I_A.avi').subclip(0,44)
white_clip = clip1.fl_image(process_video)
white_clip.write_videofile(video_output, audio=False)

"""
#cap = cv2.VideoCapture('/home/kihyeon/Study/python/lane_finding/project_video.mp4')
cap = cv2.VideoCapture('/home/kihyeon/Study/python/lane_finding/20180531_075532_I_A.avi')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is False:
        print('read fail')
        break

    processed = process_video(frame, M, Minv, Left, Right)

    cv2.imshow('frame', processed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""
