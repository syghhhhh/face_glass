# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 14:59
# @Author  : 施昀谷
# @File    : run_ori_multithreading.py
# @Description :

import os
import time
import math
import shutil
import argparse
import threading
import moviepy.editor as mp
from run_ori import run_ori

time_start = time.time()
# 基础目录
base_path = os.path.dirname(__file__)


def video_split(video_path, split_num, save_path):
    """
    将视频切割成split_num段，保存到save_path目录下

    :param video_path:
    :param split_num:
    :param save_path:
    :return:
    """
    # 创建保存目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 打开视频文件
    video = mp.VideoFileClip(video_path, verbose=False)

    # 获取视频时长
    duration = video.duration

    # 切割时间，单位为秒
    split_time = math.ceil(duration / split_num)

    for i in range(split_num):
        start = i * split_time
        end = min(duration, (i + 1) * split_time)
        sub_video = video.subclip(start, end)
        sub_video.write_videofile(os.path.join(save_path, f"{i + 1}.mp4"))
    return [os.path.join(save_path, f"{i + 1}.mp4") for i in range(split_num)]


class MyThread(threading.Thread):
    def __init__(self, name, sem, video_temp_list, time_start, glass):
        threading.Thread.__init__(self)
        self.name = name
        self.sem = sem
        self.video_temp = video_temp_list[int(name)]
        self.time_start = time_start
        self.glass = glass

    def run(self):
        print('Thread ' + self.name + ' started')
        # 执行任务
        result_temp = os.path.join(base_path, f"temp_{self.time_start}", f"result_{int(self.name) + 1}.mp4")
        run_ori(self.video_temp, result_temp, self.glass)
        print('Thread ' + self.name + ' finished')
        self.sem.release()


def main(args):
    mul = args.multithreading_thread_count
    video_temp_list = video_split(args.input, mul,
                                  os.path.join(base_path, f"temp_{time_start}"))
    sem = threading.Semaphore(0)
    threads = []
    for j in range(mul):
        thread = MyThread(str(j), sem, video_temp_list, time_start, args.glass)
        thread.start()
        threads.append(thread)

    for thread in threads:
        sem.acquire()
        thread.join()

    # 合并视频
    if args.output is None:
        args.output = os.path.join(base_path, f"output_{time_start}.mp4")
    video_clip_list = [mp.VideoFileClip(os.path.join(base_path, f"temp_{time_start}", f"result_{i + 1}.mp4")) for i in
                       range(mul)]
    video_clip = mp.concatenate_videoclips(video_clip_list)
    video_clip.write_videofile(args.output)
    shutil.rmtree(os.path.join(base_path, f"temp_{time_start}"))
    print(f"总耗时：{time.time() - time_start}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", type=str)
    parser.add_argument('-o', "--output", default=None, type=str)
    parser.add_argument('-g', "--glass", default=os.path.join(base_path, "glass/20.jpg"), type=str)
    parser.add_argument('-m', "--multithreading_thread_count", default=4, type=int,
                        help="Number of threads to use for multithreading.")
    args1 = parser.parse_args()
    # print(args)
    main(args1)


