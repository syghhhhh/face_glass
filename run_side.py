#!/usr/bin/python
# -*- coding: UTF-8 -*-

# from shutil import move
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

start_time = time.time()

###基础目录
base_path = os.path.dirname(__file__)


###此代码为侧面代码

def merge_audio(afile, vfile, outfile):
    # currentPwd = os.getcwd() + '/'
    currentPwd = ''
    command = "ffmpeg -y -i {} -i {} -c:v copy -c:a aac -strict experimental -crf 0 {} -loglevel quiet".format(currentPwd + vfile, currentPwd + afile, currentPwd + outfile)
    print('command -> ' + command)
    os.system(command)
    # os.system('rm -f {}'.format(currentPwd + vfile))


def cv_show(name, file):
    cv.imshow(name, file)
    cv.waitKey(0)
    cv.destroyAllWindows()


def add_alpha_channel(img):
    b_channel, g_channel, r_channel = cv.split(img)  # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道

    img_new = cv.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
    return img_new


def merge_img(jpg_img, png_img, y1, y2, x1, x2):
    # y1 = -1 y2 =
    # 判断jpg图像是否已经为4通道
    if jpg_img.shape[2] == 3:
        jpg_img = add_alpha_channel(jpg_img)

    yy1 = 0
    yy2 = png_img.shape[0]
    xx1 = 0
    xx2 = png_img.shape[1]

    alpha_png = png_img[yy1:yy2, xx1:xx2, 3] / 255.0

    alpha_jpg = 1 - alpha_png
    # print(png_img[yy1:yy2 ,xx1:xx2,0].shape)
    # print(alpha_png.shape)
    # print(jpg_img[y1:y2,x1:x2,0].shape)

    # # 开始叠加
    for c in range(0, 3):
        jpg_img[y1:y2, x1:x2, c] = ((alpha_jpg * jpg_img[y1:y2, x1:x2, c]) + (alpha_png * png_img[yy1:yy2, xx1:xx2, c]))

    return jpg_img


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(base_path, 'dataset/shape_predictor_68_face_landmarks.dat'))


def find_face(inputImg):
    cap = inputImg
    frame = cap.copy()

    face_Gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    boundary = detector(face_Gray, 1)

    for (index, rectangle) in enumerate(boundary):
        shape = predictor(face_Gray, rectangle)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rectangle)

    return shape


def change_glasses_side(img, move, ratio):
    img_copy = img.copy()

    width = img.shape[0]
    length = img.shape[1]
    n = []
    n.append((0, 0))
    n.append((length, 0))
    n.append((length, width))
    n.append((0, width))
    # print(width, length)
    p1 = np.array(n, dtype=np.float32)

    p2 = np.array([(move, (width * (1 - ratio)) / 2), (length * ratio, 0), (length * ratio, width),
                   (move, (width * (1 + ratio)) / 2)], dtype=np.float32)
    M = cv.getPerspectiveTransform(p1, p2)  # 变换矩阵
    # 使用透视变换
    result = cv.warpPerspective(img_copy, M, (0, 0), borderValue=(255, 255, 255))

    # 重新截取
    result = result[:1501, :1001]

    # cv.imwrite("ocr1.png", result)
    return result


def affine_change(img, move_x, move_y, move_y2):
    rows, cols, channels = img.shape
    # print(rows, cols)
    p1 = np.float32([[81, 91], [76, 228], [310, 96]])
    p2 = np.float32(
        [[81 + move_x, 91 + int(1.5 * move_y)], [76 + move_x, 228 + int(1.5 * move_y)], [310, 96 - move_y2]])
    M = cv.getAffineTransform(p1, p2)
    dst = cv.warpAffine(img, M, (cols, rows))
    return dst


feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=3,
                      blockSize=5)  # ShiTomasi corner detection parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))  # Lucas Kanade optical flow method parameters


def run_video(path, output, size, glass):
    cap1 = cv.VideoCapture(path)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    print(int(cap1.get(3)), int(cap1.get(4)), int(cap1.get(5)))
    temp_out = output + '.tmp.mp4'
    out = cv.VideoWriter(temp_out, fourcc, int(cap1.get(5)), (int(cap1.get(3)), int(cap1.get(4))))

    frame_idx = -1
    detect_interval = 0
    tracks = []
    frames_num = int(cap1.get(cv.CAP_PROP_FRAME_COUNT))
    while cap1.isOpened():

        # print("视频帧数:{} / {}".format(frame_idx, frames_num))
        ret, frame = cap1.read()
        frame_idx += 1
        if not ret:
            out.release()
            print("处理完毕, 即将合成音频和视频", path, temp_out)
            # time.sleep(5.0)
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
            total_xl = 0
            total_yl = 0
            total_xr = 0
            total_yr = 0
            average_count = 0
            r_aver = 0
            l_aver = 0
            for i in range(len(good_new)):
                average_count += good_new[i][0]
            average_col = average_count / len(good_new)

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                if a > average_col:
                    total_xr += int(a) - int(c)
                    total_yr += int(b) - int(d)
                    r_aver += 1
                else:
                    total_xl += int(a) - int(c)
                    total_yl += int(b) - int(d)
                    l_aver += 1
                # mask = cv.line(frame, (int(a) + 200,int(b)),(int(c)+ 200,int(d)), (0,0,255), 2) #Added an optical flow track diagram for this frame
                # frame = cv.circle(frame,(int(a)+left_x + 30,int(b) +top_y - 50),1,(255,255,255),-1)

            total_xl = int(total_xl / l_aver)
            total_yl = int(total_yl / l_aver)

            total_xr = int(total_xr / r_aver)
            total_yr = int(total_yr / r_aver)
            total_xx = int((total_xl + total_xr) / 2)
            total_yy = int((total_yl + total_yr) / 2)

            if total_xl > 8:
                gs_img = change_glasses_side(s_img, int(total_xl / 3), 0.99)

            else:
                gs_img = change_glasses_side(s_img, 0, 1)

            # if total_x>5:
            #     gs_img = change_glasses_side(s_img,int(total_x/2),0.75)

            # elif total_x<-5:
            #     gs_img = change_glasses_side(s_img,int(total_x/2),0.75)
            # else:
            #     gs_img = change_glasses_side(s_img,0,0.75)
            # 根据平均位移减第一帧位移，归0化
            # if int(total_x/3) >2 or int(total_y/3) >2:
            # s_x = standard_x.copy()
            # s_y = standard_y.copy()

            # if abs(int(total_x/3)) <2 or abs(int(total_y/3)) <2:
            #     standard_x = shape[0][0] -110
            #     standard_y = shape[17][1] - 100
            # print(total_yr)
            # 判断镜腿和镜片的相对位置
            if abs(total_yl - total_yr) >= 0:

                if total_yr > 1:
                    c_img = affine_change(s_img, 0, total_yl, +total_yr)
                    x22 = standard_x + c_img.shape[1] + total_xl
                    y22 = standard_y + c_img.shape[0]
                    frame = merge_img(frame, c_img, standard_y, y22, standard_x + total_xl, x22)
                else:
                    c_img = affine_change(s_img, 0, total_yl, 0)
                    x22 = standard_x + c_img.shape[1] + total_xl
                    y22 = standard_y + c_img.shape[0]
                    # frame = merge_img(frame, c_img, standard_y,y22,standard_x+total_xl,x22)
                    if size == 0.3:
                        frame = merge_img(frame, c_img, standard_y + 35, y22 + 35, standard_x + total_xl + 55, x22 + 55)
                    elif size == 0.4:
                        frame = merge_img(frame, c_img, standard_y + 32, y22 + 32, standard_x + total_xl + 48, x22 + 48)
                    elif size == 0.5:
                        frame = merge_img(frame, c_img, standard_y + 20, y22 + 20, standard_x + total_xl + 30, x22 + 30)
                    elif size == 0.6:
                        frame = merge_img(frame, c_img, standard_y + 5, y22 + 5, standard_x + total_xl + 12, x22 + 12)

            else:
                c_img = gs_img
                x22 = standard_x + c_img.shape[1] + total_xx
                y22 = standard_y + c_img.shape[0] + total_yy
                frame = merge_img(frame, c_img, standard_y + total_yy, y22, standard_x + total_xx, x22)

            # else:
            #     c_img = s_img
            #     x22 = standard_x + c_img.shape[1] +total_xx
            #     y22 = standard_y+ c_img.shape[0] +total_yy
            #     frame = merge_img(frame, c_img, standard_y +total_yy,y22,standard_x+total_xx,x22)

            b_channel, g_channel, r_channel, a_channel = cv.split(frame)
            end_frame = cv.merge((b_channel, g_channel, r_channel))
            # cv.imshow("!1",end_frame)
            if frame_idx > 0:
                out.write(end_frame.astype('uint8'))

            # p0 = good_new.reshape(-1,1,2)
        if frame_idx == 0:  # 每250帧检测一次特征点

            shape = find_face(frame)
            old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            s_img = cv.imread(glass, -1)

            s_img = change_glasses_side(s_img, 0, 0.95)

            # s_img = change_glasses_side(s_img)
            # 把眼镜的大小定义好，大小不改变
            x1 = shape[0][0]
            x2 = shape[16][0]
            y1 = shape[44][1]
            y2 = shape[46][1]

            # 第二个参数决定再旋转 第三个参数越大 左侧高越小 第四个参数越大 右侧宽越大（整体越小）

            # x_end = x2 - x1 +250
            # y_end = y2 - y1 +180
            if size == 0.3:
                x_end = x2 - x1 + int(250 * 0.5)
                y_end = y2 - y1 + int(180 * 0.5)
            elif size == 0.4:
                x_end = x2 - x1 + int(250 * 0.57)
                y_end = y2 - y1 + int(180 * 0.57)
            elif size == 0.5:
                x_end = x2 - x1 + int(250 * 0.74)
                y_end = y2 - y1 + int(180 * 0.74)
            elif size == 0.6:
                x_end = x2 - x1 + int(250 * 0.9)
                y_end = y2 - y1 + int(180 * 0.9)
            # 单位 行列
            s_img = cv.resize(s_img, (x_end, y_end))

            # 定义一个搜索角点的范围
            left_x = shape[0][0]
            right_x = shape[16][0] + 40
            top_y = shape[19][1]
            bottom_y = shape[14][1]
            # 定义一个roi区域
            standard_x = shape[0][0] - 105
            standard_y = shape[17][1] - 75
            roi = old_gray[top_y + 40:bottom_y + 80, left_x + 30:right_x - 30]

            p = cv.goodFeaturesToTrack(roi, mask=None, **feature_params)  # 像素级别角点检测
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

        if cv.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="display a square of a given number", type=str)
    parser.add_argument("--size", help="display size of picture", type=float)
    parser.add_argument("--output", help="display a square of a given number", default=None, type=str)
    parser.add_argument("--glass", help="display a square of a given number",
                        default=os.path.join(base_path, "glass/22s.png"), type=str)
    args = parser.parse_args()

    path = args.input
    output = args.output
    if not output:
        output = args.input.split('.')[0] + '_out.' + args.input.split('.')[1]

    run_video(path, output, args.size, args.glass)
