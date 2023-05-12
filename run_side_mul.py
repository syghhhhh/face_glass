# -*- coding: utf-8 -*-
# @Time    : 2023/5/12 16:19
# @Author  : 施昀谷
# @File    : run_side_mul.py

import os
import json
import time
import math
import shutil
import argparse
import multiprocessing
from run_side import run_video

time_start = int(time.time())
# 基础目录
base_path = os.path.dirname(__file__)
print(f'base_path -> {base_path}')


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

    # 获取视频信息
    video_info = get_video_information_os(video_path)
    duration = float(video_info['format']['duration'])  # 视频时长，单位为秒
    video_bitrate = video_info['format']['bit_rate']  # 视频码率
    video_codec = video_info['streams'][0]['codec_name']  # 视频编码格式
    print(f'duration -> {duration}')  # 70.32
    print(f'video_bitrate -> {video_bitrate}')  # '2057219'
    print(f'video_codec -> {video_codec}')  # 'h264'

    # 切割时间，单位为秒
    print(f'split_num -> {split_num}')
    split_time = duration / split_num
    print(f'split_time -> {split_time}')

    for i in range(split_num):
        start = round(i * split_time)
        print(f'{i}:start -> {start}')
        end = round((i + 1) * split_time)
        if i == split_num - 1:
            end = math.ceil(duration)
        print(f'{i}:end -> {end}')

        command = f'ffmpeg -i {video_path} -ss {start} -to {end} -c:v {video_codec} -b:v {video_bitrate} -c:a copy ' \
                  f'{os.path.join(save_path, f"split_{i + 1}.mp4")} -loglevel quiet'
        print(f'command -> {command}')
        os.system(command)
    return [os.path.join(save_path, f"split_{i + 1}.mp4") for i in range(split_num)]
# ------------------------------------------------------------------------


def main(video_path, save_path, glass_path, multithreading_thread_count, size):
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
    video_path_list = video_split_ffmpeg(video_path, multithreading_thread_count, temp_folder)
    print("video_path_list -> " + str(video_path_list))

    # 多进程执行run_ori
    for i in range(multithreading_thread_count):
        print(f"run_ori.py split_{i + 1}.mp4")
        p = multiprocessing.Process(target=run_video,
                                    args=(video_path_list[i], os.path.join(temp_folder, f"result_{i + 1}.mp4"),
                                          size, glass_path))
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
    parser.add_argument('-g', "--glass", default=os.path.join(base_path, "glass/22s.png"), type=str)
    parser.add_argument('-m', "--multithreading_thread_count", default=3, type=int,
                        help="Number of threads to use for multithreading.")
    parser.add_argument('-s', "--size", help="display size of picture", type=float)
    args = parser.parse_args()

    print(f'args -> {args}')

    main(args.input, args.output, args.glass, args.multithreading_thread_count, args.size)
    print(f'time cost -> {time.time() - time_start}')

    # python run_side_mul.py -i test_video/side_0.5_2.mp4 -o result_video/side_0.5_3.mp4 -s 0.5
