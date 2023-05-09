# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 18:06
# @Author  : 施昀谷
# @File    : mul_test.py

import time
import threading


def fun_a(i1):
    # 执行一些操作
    time.sleep(i1 * 0.5)
    print(f"function {i1}")


def fun_b():
    # 执行一些操作
    print("function b")


def main():
    # 开始多线程执行 fun_a
    threads = []
    for i in range(10):
        t = threading.Thread(target=fun_a, args=(i,))
        threads.append(t)
        t.start()

    # 等待所有线程执行完毕
    for t in threads:
        t.join()

    # 执行 fun_b
    fun_b()


if __name__ == '__main__':
    main()
