#!/usr/bin/python
# -*- coding: UTF-8 -*-

# from threading import local
# from cv2 import imshow
import dlib
# import imutils
import cv2 as cv
from imutils import face_utils
# import math
import time
import argparse
import os
# import multiprocessing as mp
import numpy as np
import ffmpeg
from imutils.video import FileVideoStream

start_time = time.time()


def merge_audio(afile, vfile, outfile):
    currentPwd = os.getcwd() + '/'
    command = "ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental {}".format(currentPwd + vfile,
                                                                                     currentPwd + afile,
                                                                                     currentPwd + outfile)
    print('command -> ' + command)
    os.system(command)
    # os.system('rm -f {}'.format(currentPwd + vfile))


def merge_audio_py(infile, videofile, outfile):
    input_video = ffmpeg.input(infile)
    audio = input_video.audio
    output_video = ffmpeg.input(videofile)
    video = output_video.video
    stream = ffmpeg.output(audio, video, outfile)
    ffmpeg.run(stream)


def cv_show(name, file):
    cv.imshow(name, file)
    cv.waitKey(0)
    cv.destroyAllWindows()


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha_inv * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha * img[y1:y2, x1:x2, c])


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('dataset/shape_predictor_68_face_landmarks.dat')


def find_face(inputImg):
    cap = inputImg
    frame = cap.copy()

    face_Gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    boundary = detector(face_Gray, 1)

    for (index, rectangle) in enumerate(boundary):
        shape = predictor(face_Gray, rectangle)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rectangle)
    # x1 = shape[0][0]
    #     x2 = shape[16][0]
    #     y1 = shape[44][1]
    #     y2 = shape[46][1]

    #     x_end = x2 - x1
    #     y_end = y2 - y1 + 30

    return shape


feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=3,
                      blockSize=5)  # ShiTomasi corner detection parameters
lk_params = dict(winSize=(25, 25), maxLevel=2, criteria=(
cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))  # Lucas Kanade optical flow method parameters

# def get_first(path):
#     cap = cv.VideoCapture(path) #It is necessary to recapture the video here, otherwise.read() will not get the first frame correctly
#     ret, old_frame = cap.read() #Get the first frame
#     old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

#     shape = find_face(old_frame)
#     s_img = cv.imread('images/glass6.jpg')
#     #把眼镜的大小定义好，大小不改变
#     x1 = shape[0][0] 
#     x2 = shape[16][0] 
#     y1 = shape[44][1]
#     y2 = shape[46][1]
#     x_end = x2 - x1
#     y_end = y2 - y1 + 30
#     s_img = cv.resize(s_img, (x_end, y_end))

#     #定义一个搜索角点的范围
#     left_x = shape[0][0]
#     right_x = shape[16][0]
#     top_y = shape[19][1] 
#     bottom_y = shape[17][1]
#     roi = old_gray[top_y:bottom_y,left_x:right_x]

#     p0 = cv.goodFeaturesToTrack(roi, mask = None, **feature_params)

#     return p0,old_gray,shape,top_y,bottom_y,left_x,right_x,s_img
# p0,old_gray,shape,top_y,bottom_y,left_x,right_x,s_img = get_first(path)
# standard_x = shape[0][0]
# standard_y = shape[17][1]
# save_x = 0
parser = argparse.ArgumentParser()
parser.add_argument("--input", help="display a square of a given number", default='test_video/front_0.5.mp4', type=str)
parser.add_argument("--size", help="display size of picture", default=0.5, type=float)
parser.add_argument("--output", help="display a square of a given number", default='result_video/front_1.mp4', type=str)
parser.add_argument("--glass", help="display a square of a given number", default="glass/20.jpg", type=str)
args = parser.parse_args()

path = args.input
output = args.output
if not output:
    output = args.input.split('.')[0] + '_out.' + args.input.split('.')[1]
cap1 = cv.VideoCapture(path)

# custom
fvs = FileVideoStream(path).start()

fourcc = cv.VideoWriter_fourcc(*'mp4v')
print(int(cap1.get(3)), int(cap1.get(4)), int(cap1.get(5)))
temp_out = output + '.tmp.mp4'
out = cv.VideoWriter(temp_out, fourcc, int(cap1.get(5)), (int(cap1.get(3)), int(cap1.get(4))))

frame_idx = -1
detect_interval = 0
tracks = []
frames_num = int(cap1.get(cv.CAP_PROP_FRAME_COUNT))
# hile cap1.isOpened():
while fvs.more():

    print("视频帧数:{} / {}".format(frame_idx, frames_num))
    # ret,frame = cap1.read()
    frame = fvs.read()
    frame_idx += 1
    # if not ret:
    if frame_idx >= frames_num:
        out.release()
        print("处理完毕, 即将合成音频和视频", path, temp_out)
        time.sleep(5.0)
        merge_audio(path, temp_out, output)
        print("[INFO] 总耗时: {:.2f}s".format(time.time() - start_time))
        break
    if len(tracks) > 0:
        shape = find_face(frame)
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # 光流的搜索范围比角点搜索范围更大，因为考虑到角点的运动
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray[top_y - 50:bottom_y + 50, left_x + 30:right_x - 30],
                                              frame_gray[top_y - 50:bottom_y + 50, left_x + 30:right_x - 30], p0, None,
                                              **lk_params)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        total_x = 0
        total_y = 0

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            total_x += int(a) - int(c)
            total_y += int(b) - int(d)
            # # mask = cv.line(frame, (int(a) + 200,int(b)),(int(c)+ 200,int(d)), (0,0,255), 2) #Added an optical flow track diagram for this frame
            # frame = cv.circle(frame,(int(a)+left_x + 30,int(b) +top_y - 50),5,(255,255,255),-1)

        total_x = int(total_x / len(good_new))
        total_y = int(total_y / len(good_new))
        # print(total_x,total_y)

        # 根据平均位移减第一帧位移，归0化
        overlay_image_alpha(frame, s_img, (standard_x + total_x, standard_y + total_y), s_img[:, :, 2] / 255.0)
        # cv.imshow('lk_track', frame)
        # a = 0
        # if len(good_new) != 0:
        #     for i in range(len(good_new)):
        #         if good_new[i][0] != 0 or good_new[i][1] !=0:
        #             a +=1
        #             total_x += good_new[i][0]
        #             total_y += good_new[i][1]

        #     #平均位移
        #     total_x = int(total_x/len(good_new))
        #     total_y = int(total_y/len(good_new))

        #     #s_xy是第一帧的位移
        #     if frame_idx == 1:
        #         s_x = total_x
        #         s_y = total_y

        #     #根据平均位移减第一帧位移，归0化
        #     loc_x = total_x - s_x
        #     loc_y = total_y - s_y

        #     catch_change_one = 0
        #     # if abs(loc_x - save_x ) > 10:
        #     #     #如果位移过大，需要重新归零，因为这是突然变大的，所以此时不能根据第一帧归零
        #     #     #需要根据前一帧归零
        #     #     catch_change_one += 1

        #     #     loc_x = total_x - st_x
        #     #     loc_y = total_y  - st_y
        #     #     if catch_change_one == 1:
        #     #         st_x = total_x
        #     #         st_y = total_y

        #     # save_x = loc_x
        #     # save_y = loc_y
        #     old_gray = frame_gray
        #     overlay_image_alpha(frame, s_img, (standard_x + loc_x , standard_y + loc_y), s_img[:, :, 2] / 255.0)
        # cv.imshow('lk_track', frame)

        # old_gray = frame_gray.copy()
        # roi = old_gray[top_y:bottom_y,left_x:right_x]
        # p0 = good_new.reshape(-1,1,2)
    if frame_idx == 0:  # 每250帧检测一次特征点

        shape = find_face(frame)
        old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        s_img = cv.imread(args.glass)
        # 把眼镜的大小定义好，大小不改变
        x1 = shape[0][0]
        x2 = shape[16][0]
        y1 = shape[44][1]
        y2 = shape[46][1]
        if args.size == 0.6:
            x_end = x2 - x1
            # y_end = y2 - y1 + 60
            y_end = y2 - y1 + 50
        elif args.size == 0.5:
            x_end = x2 - x1
            y_end = y2 - y1 + 40
        elif args.size == 0.4:
            x_end = x2 - x1
            y_end = y2 - y1 + 30
        elif args.size == 0.3:
            x_end = x2 - x1
            y_end = y2 - y1 + 20
        s_img = cv.resize(s_img, (x_end, y_end))
        # 定义一个搜索角点的范围
        left_x = shape[0][0]
        right_x = shape[16][0]
        top_y = shape[19][1]
        bottom_y = shape[17][1]
        # 定义一个roi区域
        standard_x = shape[0][0]
        standard_y = shape[37][1] - 20
        if args.size == 0.6 or args.size == 0.5:
            roi = old_gray[top_y + 40:bottom_y + 80, left_x + 30:right_x - 30]
            # 60%:[147:196, 1466:1598]
        elif args.size == 0.4:
            roi = old_gray[top_y + 40:bottom_y + 80, left_x + 30:right_x - 30]
            # roi = old_gray[top_y:bottom_y + 230, left_x - 50:right_x + 70]
        # 40%: [431:487, 1404:1586] 30%: [363:407, 1665:1651]
        p = cv.goodFeaturesToTrack(roi, mask=None, **feature_params)  # 像素级别角点检测
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中
    else:
        out.write(frame.astype('uint8'))
    if cv.waitKey(1) == ord('q'):
        break

# parser = argparse.ArgumentParser()
# parser.add_argument("input", help="display a square of a given number",
#                     type=str)
# parser.add_argument("output", help="display a square of a given number",
#                     type=str)
# args = parser.parse_args()
# num_cores = int(mp.cpu_count())
# print("本地计算机有: " + str(num_cores) + " 核心")
# pool = mp.Pool(processes = num_cores)

# for filename in os.listdir(args.input):
#     print(filename)
# #    add_glasses_to_face(os.path.join(args.input, filename), os.path.join(args.output, filename))    
#     pool.apply_async(add_glasses_to_face, (os.path.join(args.input, filename), os.path.join(args.output, filename)))

# pool.close()
# pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
# print("Successfully")
