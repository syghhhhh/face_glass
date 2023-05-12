import cv2
import os
import argparse
import time
import json
import shutil

# 基础目录
base_path = os.path.dirname(__file__)


# # ----------------------------用不到的函数--------------------------------
# def extract_frames(video_path, dst_folder, index):
#     # 实例化视频对象
#     video = cv2.VideoCapture(video_path)
#
#     # 循环遍历视频中的所有帧
#     while True:
#         # 逐帧读取
#         _, frame = video.read()
#         if frame is None:
#             break
#         # 设置保存文件名
#         save_path = "{}/{:>03d}.jpg".format(dst_folder, index)
#         # 保存图片
#         cv2.imwrite(save_path, frame)
#         index += 1  # 保存图片数＋1
#     video.release()
#
#
# def get_mp4(folder, outfile):
#     """
#
#     :param folder:
#     :param outfile:
#     :return:
#     """
#     command = "ffmpeg -f image2 -i {}%3d.jpg -crf 0 {} -loglevel quiet".format(f'{folder}/', outfile)
#     os.system(command)
#
#
# def get_wav(infile, wavname):
#     command = 'ffmpeg -i {} -f wav -ar 44100 {} -loglevel quiet'.format(infile, wavname)
#     os.system(command)
#
#
# def merger(mp4, wav, outfile):
#     command = 'ffmpeg -i {} -i {} -c:v copy -c:a aac -strict experimental -crf 0 -y {} -loglevel quiet'.format(mp4, wav, outfile)
#     os.system(command)
#
#
# def merge(picture_folder, audio_path, output_path, arg_dict):
#     """
#     合并视频
#
#     :param picture_folder:
#     :param audio_path:
#     :param output_path:
#     :param arg_dict:
#     :return:
#     """
#     command = f'ffmpeg -framerate {arg_dict["fps"]} -i {picture_folder}/%05d.png -i {audio_path} ' \
#               f'-vcodec {arg_dict["codec"]} -r {arg_dict["fps"]} -s {arg_dict["width"]}x{arg_dict["height"]} ' \
#               f'-pix_fmt {arg_dict["pix_fmt"]} -acodec {arg_dict["audio_codec"]} -shortest {output_path} -loglevel quiet'
#     os.system(command)
# # ------------------------------------------------------------------------


# 覆盖图像
def overlay_img(img, img_over, img_over_x, img_over_y):
    img_h, img_w, img_p = img.shape  # 背景图像宽、高、通道数
    img_over_h, img_over_w, img_over_c = img_over.shape  # 覆盖图像高、宽、通道数
    if img_over_c == 3:  # 通道数小于等于3
        img_over = cv2.cvtColor(img_over, cv2.COLOR_BGR2BGRA)  # 转换成4通道图像
    for w in range(0, img_over_w):  # 遍历列
        for h in range(0, img_over_h):  # 遍历行
            if img_over[h, w, 3] != 0:  # 如果不是全透明的像素
                for c in range(0, 3):  # 遍历三个通道
                    x = img_over_x + w  # 覆盖像素的横坐标
                    y = img_over_y + h  # 覆盖像素的纵坐标
                    if x >= img_w or y >= img_h:  # 如果坐标超出最大宽高
                        break  # 不做操作
                    img[y, x, c] = img_over[h, w, c]  # 覆盖像素
    return img  # 完成覆盖的图像


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
    # # 视频帧率
    # fps = int(video_info['streams'][0]['r_frame_rate'].split('/')[0]) / int(
    #     video_info['streams'][0]['r_frame_rate'].split('/')[1])
    # # 音频编码器
    # audio_codec = video_info['streams'][1]['codec_name']
    # # 视频分辨率
    # width = video_info['streams'][0]['width']
    # height = video_info['streams'][0]['height']
    # 视频编码器
    codec = video_info['streams'][0]['codec_name']
    # 像素格式
    pix_fmt = video_info['streams'][0]['pix_fmt']
    args_dict = {
        # 'fps': fps,
        # 'audio_codec': audio_codec,
        # 'width': width,
        # 'height': height,
        'codec': codec,
        'pix_fmt': pix_fmt
    }
    return args_dict


def extract_frames_ffmpeg(video_path, save_folder):
    command = f'ffmpeg -i {video_path} -vf "select=not(mod(n\\,1))" -vsync 0 {save_folder}/%05d.png -loglevel quiet'
    os.system(command)
    return os.listdir(save_folder)


def merge_new(input_video, frames_folder, output_video, codec, pix_fmt):
    # command = f'ffmpeg -i {input_video} -i {frames_folder}/merged_%05d.png -y -threads $(nproc) ' \
    #           f'-filter_complex "[0:v][1:v] overlay=shortest=1 [v]" -map "[v]" -map 0:a -c:v {codec} -crf 0 ' \
    #           f'-preset medium -pix_fmt {pix_fmt} -f mp4 {output_video}'
    command = f'ffmpeg -i {input_video} -i {frames_folder}/merged_%05d.png -y ' \
              f'-filter_complex "[0:v][1:v] overlay=shortest=1 [v]" -map "[v]" -map 0:a -c:v {codec} -crf 0 ' \
              f'-preset medium -pix_fmt {pix_fmt} -f mp4 {output_video} -loglevel quiet'
    os.system(command)


def run_front(input_path, output_path, start_time, glass):
    # 创建工作目录
    folder_work = os.path.join(base_path, 'work' + str(start_time))
    os.makedirs(folder_work, exist_ok=True)
    # folder_merge = os.path.join(base_path, 'merge' + str(start_time))
    # os.makedirs(folder_merge, exist_ok=True)
    # EXTRACT_FREQUENCY = 25

    # 视频逐帧拆成图片保存
    # extract_frames(input_path, folder_work, 1)
    picture_list = extract_frames_ffmpeg(input_path, folder_work)

    glass_img = cv2.imread(glass, cv2.IMREAD_UNCHANGED)  # 读取眼镜图像，保留图像类型
    height, width, channel = glass_img.shape  # 获取眼镜图像高、宽、通道数

    # 每张图片添加眼镜
    for picture in picture_list:
        # print(picture + " is processing")
        k = 0.5
        frame = cv2.imread(os.path.join(folder_work, picture))
        face_cascade = cv2.CascadeClassifier(os.path.join(base_path, "haarcascade_frontalface_default.xml"))
        garyframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(garyframe, 1.1, 5)  # 识别人脸
        # for (x, y, w, h) in faces:
        #     print(x, y, w, h)
        # break
        for (x, y, w, h) in faces:  # 遍历所有人脸的区域
            # gw = int(0.8 * w)  # 眼镜缩放之后的宽度
            # gh = int(0.8 * height * w / width)  # 眼镜缩放之后的高度度
            if picture == picture_list[0]:
                x0 = x
                y0 = y
                gw = int(0.8 * w)  # 眼镜缩放之后的宽度
                gh = int(0.8 * height * w / width)  # 眼镜缩放之后的高度度
                glass_img = cv2.resize(glass_img, (gw, gh))  # 按照人脸大小缩放眼镜
                overlay_img(frame, glass_img, x0 + int(0.1 * w), y0 + int(h * 1 / 3))  # 将眼镜绘制到人脸上
                break
            else:
                if abs(x - x0) / x0 < 0.1 and abs(y - y0) / y0 < 0.1:
                    x0 = int(x0 * (1 - k) + x * k)
                    y0 = int(y0 * (1 - k) + y * k)
                    overlay_img(frame, glass_img, x0 + int(0.1 * w), y0 + int(h * 1 / 3))  # 将眼镜绘制到人脸上
                    break
        # (x, y, w, h) = faces[0]  # 取置信度最高的人脸
        # gw = int(0.8 * w)  # 眼镜缩放之后的宽度
        # gh = int(0.8 * height * w / width)  # 眼镜缩放之后的高度度
        # glass_img = cv2.resize(glass_img, (gw, gh))  # 按照人脸大小缩放眼镜
        # overlay_img(frame, glass_img, x + int(0.1 * w), y + int(h * 1 / 3))  # 将眼镜绘制到人脸上
        cv2.imwrite(os.path.join(folder_work, 'merged_' + picture), frame)

    # 视频信息
    args_dict = get_video_information_os(input_path)
    # 合成视频
    # path_mp4 = os.path.join(folder_merge, f'result_{start_time}.mp4')
    # get_mp4(folder_work, path_mp4)
    # path_wav = os.path.join(folder_merge, f'{start_time}.wav')
    # get_wav(input_path, path_wav)
    # merger(path_mp4, path_wav, output_path)
    merge_new(input_path, folder_work, output_path, args_dict['codec'], args_dict['pix_fmt'])

    # 删除中间文件
    # os.remove(path_mp4)
    # os.remove(path_wav)
    # for picture in os.listdir(folder_work):
    #     os.remove(os.path.join(folder_work, picture))
    # os.rmdir(folder_work)
    shutil.rmtree(folder_work)
    # shutil.rmtree(folder_merge)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="display a square of a given number", default='test_video/front_0.3.mp4', type=str)
    parser.add_argument("-o", "--output", help="display a square of a given number", default='result_video/front_3.mp4', type=str)
    parser.add_argument('-g', "--glass", default=os.path.join(base_path, "glass"), type=str)
    args = parser.parse_args()

    start_time1 = int(time.time())
    run_front(args.input, args.output, start_time1, os.path.join(args.glass, "20.png"))
    print(f'cost time: {int(time.time()) - start_time1}')

    # python run_front.py -i test_video/front_0.3.mp4 -o result_video/front_3.mp4
