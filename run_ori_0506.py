# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 18:57
# @Author  : 施昀谷
# @File    : run_ori_0506.py


import os
# import cv2
import json
import time
import math
import ffmpeg
import shutil
import argparse
# import threading
import multiprocessing
# import subprocess
# import moviepy.editor as mp
from run_ori import run_ori

time_start = int(time.time())
# 基础目录
base_path = os.path.dirname(__file__)
print(f'base_path -> {base_path}')


# # ----------------------------用不到的函数--------------------------------
# def get_video_duration_ffprobe(video_path):
#     """
#     subprocess调用ffprobe获取视频时长
#
#     :param video_path: 视频路径
#     :return: 视频时长，单位为秒
#     """
#     result = subprocess.run(
#         ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
#          video_path, '-loglevel', 'quiet'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#     duration = float(result.stdout)
#     return duration
#
#
# def video_split_moviepy(video_path, split_num, save_path):
#     """
#     用moviepy库将视频切割成split_num段，保存到save_path目录下
#
#     :param video_path:
#     :param split_num:
#     :param save_path:
#     :return:
#     """
#     # 创建保存目录
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     # 打开视频文件
#     video = mp.VideoFileClip(video_path, verbose=False)
#
#     # 获取视频时长
#     duration = video.duration
#
#     # 切割时间，单位为秒
#     split_time = math.ceil(duration / split_num)
#
#     for i in range(split_num):
#         start = i * split_time
#         end = min(duration, (i + 1) * split_time)
#         sub_video = video.subclip(start, end)
#         sub_video.write_videofile(os.path.join(save_path, f"split_{i + 1}.mp4"))
#     return [os.path.join(save_path, f"split_{i + 1}.mp4") for i in range(split_num)]
#
#
# def video_split_opencv(video_path, split_num, save_path):
#     """
#     用cv2库将视频切割成split_num段，保存到save_path目录下
#
#     :param video_path:
#     :param split_num:
#     :param save_path:
#     :return:
#     """
#     # 读取视频
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
#     img_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 图像大小
#
#     # 计算每一段需要抽取的帧数
#     frames_per_part = total_frames // split_num
#
#     # 循环读取视频帧，抽取部分帧并将其保存
#     for i in range(split_num):
#         start_frame = i * frames_per_part  # 起始帧
#         end_frame = (i + 1) * frames_per_part if i < split_num-1 else total_frames  # 终止帧
#
#         # 设置帧的位置
#         cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
#
#         # 创建VideoWriter对象
#         save_name = os.path.join(save_path, f"split_{i + 1}.mp4")
#         out = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, img_size)
#
#         # 读取帧并保存
#         for j in range(start_frame, end_frame):
#             ret, frame = cap.read()
#             if ret:
#                 out.write(frame)
#             else:
#                 break
#
#         # 释放资源
#         out.release()
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     return [os.path.join(save_path, f"split_{i + 1}.mp4") for i in range(split_num)]
# # ------------------------------------------------------------------------


# ----------------------------用到的函数--------------------------------
def get_video_information(video_path):
    """
    获取视频信息

    :param video_path: 视频路径
    :return: 视频信息
    """
    # 获取视频属性
    probe = ffmpeg.probe(video_path)
    video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    return video_info


def get_video_information_os(video_path):
    """
    获取视频信息

    :param video_path: 视频路径
    :return: 视频信息
    """
    # 获取视频属性
    cmd = f'ffprobe -v quiet -print_format json -show_format -show_streams {video_path}'
    video_info = os.popen(cmd).read()

    # 字符串转字典
    video_info = json.loads(video_info)
    return video_info


def video_split_ffmpeg(video_path, split_num, save_path):
    """
    用ffmpeg将视频切割成split_num段，保存到save_path目录下。发现ffmpeg切割的视频会有问题，视频会有卡顿现象，所以弃用。

    :param video_path: 原视频路径
    :param split_num: 切割数量
    :param save_path: 保存路径
    :return:
    """
    # 创建保存目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # # 获取视频时长
    # duration = get_video_duration_ffprobe(video_path)
    # print(f'duration -> {duration}')

    # # 获取视频信息
    # video_info = get_video_information(video_path)
    # duration = float(video_info['duration'])  # 视频时长，单位为秒
    # video_bitrate = video_info['bit_rate']  # 视频码率
    # video_codec = video_info['codec_name']  # 视频编码格式
    # print(f'duration -> {duration}')  # 70.32
    # print(f'video_bitrate -> {video_bitrate}')  # '1982950'
    # print(f'video_codec -> {video_codec}')  # 'h264'

    # 获取视频信息
    video_info = get_video_information_os(video_path)
    duration = float(video_info['format']['duration'])  # 视频时长，单位为秒
    video_bitrate = video_info['format']['bit_rate']  # 视频码率
    video_codec = video_info['streams'][0]['codec_name']  # 视频编码格式
    print(f'duration -> {duration}')  # 70.32
    print(f'video_bitrate -> {video_bitrate}')  # '2057219'
    print(f'video_codec -> {video_codec}')  # 'h264'

    # 切割时间，单位为秒
    split_time = duration / split_num
    print(f'split_time -> {split_time}')

    for i in range(split_num):
        start = round(i * split_time)
        print(f'{i}:start -> {start}')
        end = min(math.ceil(duration), round((i + 1) * split_time))
        print(f'{i}:end -> {end}')

        # subprocess.run(['ffmpeg', '-i', video_path, '-ss', str(start), '-to', str(end), '-c:v', 'copy',
        #                 os.path.join(save_path, f"split_{i + 1}.mp4"), '-loglevel', 'quiet']
        #                , stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        command = f'ffmpeg -i {video_path} -ss {start} -to {end} -c:v {video_codec} -b:v {video_bitrate} -c:a copy ' \
                  f'{os.path.join(save_path, f"split_{i + 1}.mp4")} -loglevel quiet'
        print(f'command -> {command}')
        os.system(command)
    return [os.path.join(save_path, f"split_{i + 1}.mp4") for i in range(split_num)]
# ------------------------------------------------------------------------


def main(video_path, save_path, glass_path, multithreading_thread_count):
    """
    主函数

    :param video_path: 视频路径
    :param save_path: 保存路径
    :param glass_path: 眼镜图片路径
    :param multithreading_thread_count: 多线程数量
    :return:
    """
    # 新建临时工作目录
    temp_folder = os.path.join(base_path, f"temp_{time_start}")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    # 切割视频
    video_path_list = video_split_ffmpeg(video_path, multithreading_thread_count, temp_folder)  # 10s
    # video_path_list = video_split_moviepy(video_path, multithreading_thread_count, temp_folder)  # 26s
    # video_path_list = video_split_opencv(video_path, multithreading_thread_count, temp_folder)  # 28s
    print("video_path_list -> " + str(video_path_list))

    # # 多线程执行run_ori
    # threads = []
    # for i in range(multithreading_thread_count):
    #     print(f"run_ori_{i + 1}.mp4")
    #     t = threading.Thread(target=run_ori, args=(video_path_list[i], os.path.join(temp_folder, f"merge_{i + 1}.mp4"), glass_path))
    #     threads.append(t)
    #     t.start()
    # print("threads -> " + str(threads))
    #
    # # 等待所有线程执行完毕
    # for t in threads:
    #     t.join()
    # print("all threads done")

    # 多进程执行run_ori
    for i in range(multithreading_thread_count):
        print(f"run_ori.py split_{i + 1}.mp4")
        p = multiprocessing.Process(target=run_ori,
                                    args=(video_path_list[i], os.path.join(temp_folder, f"result_{i + 1}.mp4"),
                                          glass_path))
        p.start()

    # 等待所有进程执行完毕
    for p in multiprocessing.active_children():
        p.join()
        print(f'child process {p.pid} done')
    print("all processes done")

    # 生成list.txt
    with open(os.path.join(temp_folder, 'list.txt'), 'w') as f:
        for i in range(multithreading_thread_count):
            f.write(f"file '{os.path.join(temp_folder, f'result_{i + 1}.mp4')}'\n")
    print("list.txt done")

    # 合并视频
    command = f"ffmpeg -f concat -safe 0 -i {os.path.join(temp_folder, 'list.txt')} -c copy {save_path} -loglevel quiet"
    print('command -> ' + command)
    os.system(command)

    # 删除临时工作目录
    shutil.rmtree(temp_folder)
    print("temp folder removed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the model')
    parser.add_argument('-i', "--input", type=str)
    parser.add_argument('-o', "--output", default=None, type=str)
    parser.add_argument('-g', "--glass", default=os.path.join(base_path, "glass/20.jpg"), type=str)
    parser.add_argument('-m', "--multithreading_thread_count", default=2, type=int,
                        help="Number of threads to use for multithreading.")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(base_path, "output.mp4")
    print(f'args -> {args}')

    main(args.input, args.output, args.glass, args.multithreading_thread_count)
    print(f'time cost -> {time.time() - time_start}')

    # python run.py -i input.mp4 -o output.mp4 -m 4
