import cv2
import numpy as np
import random


def cv2_imread(file_path, flag=1):
    """解决包含中文的路径cv2.imread无法打开的问题的函数"""
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)


def generate_random_string() -> str:
    # random length of the random string
    length = random.randint(1, 100)
    rand_str = ''
    for _ in range(length):
        rand_str += chr(random.randint(33, 127))
    print(rand_str)
    return rand_str
