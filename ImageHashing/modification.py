from pathlib import Path
from PIL import Image
import cv2
from utils import cv2_imread
import numpy as np
from typing import List


def jpeg_compress(file: Path, dst_dir: Path, param_list: List[int]):
    """
    Perform JPEG compression on image using given parameters, and save it in the given directory.
    :param file: src image file location.
    :param dst_dir: directory to save the image after operation.
    :param param_list: list of qualities of JPEG compression.
    :return: None.
    """
    img = Image.open(file)
    img = img.convert('RGB')
    prefix, suffix = file.name.split('.')
    for i, param in enumerate(param_list):
        save = dst_dir.joinpath(f'{prefix}_{i}_{str(param)}.{suffix}')
        img.save(save, "JPEG", quality=param)


def gamma_correction(file: Path, dst_dir: Path, param_list: List[float]):
    """
    Perform gamma correction on image using given parameters, and save it in the given directory.
    :param file: src image file location.
    :param dst_dir: directory to save the image after operation.
    :param param_list: list of gamma values for gamma correction.
    :return: None.
    """
    img = cv2_imread(file)
    img = img / 255.0  # project values from [0, 255] to [0, 1] for gamma correction
    prefix, suffix = file.name.split('.')
    for i, gamma in enumerate(param_list):
        save = dst_dir.joinpath(f'{prefix}_{i}_{str(gamma)}.{suffix}')
        cor = np.copy(img)
        cor = cor ** gamma
        cor = cor * 255.0
        cor = cor.astype(np.uint8)
        cv2.imwrite(str(save), cor)


def scale_image(file: Path, dst_dir: Path, param_list: List[float]):
    """
    Scale an image to different sizes using given parameters, and save it in the given directory.
    :param file: src image file location.
    :param dst_dir: directory to save the image after operation.
    :param param_list: list of ratios for image scaling.
    :return: None.
    """
    img = cv2_imread(file)
    prefix, suffix = file.name.split('.')
    for i, ratio in enumerate(param_list):
        save = dst_dir.joinpath(f'{prefix}_{i}_{str(ratio)}.{suffix}')
        scale = cv2.resize(img, (0, 0), fx=ratio, fy=ratio)
        cv2.imwrite(str(save), scale)


def Gaussian_filter(file: Path, dst_dir: Path, param_list: List[float]):
    """
    Perform Gaussian filtering using given parameters, and save it in the given directory.
    :param file: src image file location.
    :param dst_dir: directory to save the image after operation.
    :param param_list: value list of standard deviation for Gaussian filtering.
    :return: None.
    """
    img = cv2_imread(file)
    prefix, suffix = file.name.split('.')
    for i, std in enumerate(param_list):
        save = dst_dir.joinpath(f'{prefix}_{i}_{std}.{suffix}')
        blur = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=std)
        cv2.imwrite(str(save), blur)


def rotate_image(file: Path, dst_dir: Path, param_list: List[float]):
    """
    Firstly the image is rotated by given angles, and then cropped to remove padding pixels introduced by rotation,
    and lastly rescaled to the original size. Save the modified version in the given directory.
    :param file: src image file location.
    :param dst_dir: directory to save the image after operation.
    :param param_list: list of angles for rotation.
    :return: None.
    """
    img = Image.open(file)
    img = img.convert('RGB')
    W, H = img.size
    prefix, suffix = file.name.split('.')
    for i, angle in enumerate(param_list):
        save = dst_dir.joinpath(f'{prefix}_rotate_{str(angle)}.{suffix}')
        # calculate areas to preserve for image cropping
        theta = np.abs(angle)
        sin_theta = np.sin(np.pi * theta / 180.0)
        cos_theta = np.cos(np.pi * theta / 180.0)
        x_dis = (H * sin_theta + (cos_theta - 1) * W) / (2 * cos_theta)
        y_dis = (W * sin_theta + (cos_theta - 1) * H) / (2 * cos_theta)
        x_dis = int(np.ceil(x_dis))
        y_dis = int(np.ceil(y_dis))
        box = (x_dis, y_dis, W - x_dis, H - y_dis)
        # rotate image
        sample = img.rotate(angle)
        sample = sample.crop(box)
        sample = sample.resize(img.size)
        sample.save(save)


def add_text(file: Path, dst_dir: Path, text: str):
    """
    Perform Gaussian filtering using given parameters, and save it in the given directory.
    :param file: src image file location.
    :param dst_dir: directory to save the image after operation.
    :param text: text to be added on the image.
    :return: None.
    """
    img = cv2_imread(file)
    prefix, suffix = file.name.split('.')
    save = dst_dir.joinpath(f'{prefix}_text.{suffix}')
    blur = cv2.putText(img, text, (0, img.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                       0.75, (0, 0, 0), 2)
    cv2.imwrite(str(save), blur)


if __name__ == '__main__':
    src = Path('data/COREL/query_database')
    dst = Path('data/COREL/test_images')
    for file in src.iterdir():
        add_text(file, dst, "Copyright in 2019")
    pass
