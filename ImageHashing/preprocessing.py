import numpy as np
import cv2
import utils


def biliner_interpolation(image: np.ndarray, std_size: int) -> np.ndarray:
    dst_size = (std_size, std_size)
    # create the return array
    dst = cv2.resize(image, dst_size, interpolation=cv2.INTER_LINEAR)
    return dst


def Gaussian_filtering(src: np.ndarray, ksize: int) -> np.ndarray:
    # if ksize is not odd, raise Error
    if ksize % 2 == 0:
        raise ValueError("The value of ksize must be odd!")
    # values of ksize and sigma are not mentioned, I choose to use the 3*3 kernel here
    # sigmaX and sigmaY are set to be 0 here, so they will be computed accordingly
    return cv2.GaussianBlur(src, (ksize, ksize), sigmaX=0)


def get_intensity(src: np.ndarray) -> np.ndarray:
    r = src[:, :, 2]
    g = src[:, :, 1]
    b = src[:, :, 0]
    return np.array((r + g + b) / 3, dtype=np.int)


def get_Lab(src: np.ndarray) -> np.ndarray:
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    L = src_lab[:, :, 0]
    return L


def get_YCbCr(src: np.ndarray) -> np.ndarray:
    src_YCbCr = cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)
    Y = src_YCbCr[:, :, 0]
    return Y


def preprocess(src: np.ndarray, B: int, ksize: int = 3, color_space: str = 'HSI') -> np.ndarray:
    """
    Pre-process the input image and get the intensity component of it.
    :param src: image read from file.
    :param B: the image would be resized to B * B.
    :param ksize: kernel size of Gaussian filter.
    :param color_space: convert the image to the target color space to get component.
    :return: intensity component.
    """
    ret = biliner_interpolation(src, B)
    ret = Gaussian_filtering(ret, ksize=ksize)
    intensity = None
    if color_space == 'HSI':
        intensity = get_intensity(ret)
    elif color_space == 'YCbCr':
        intensity = get_YCbCr(ret)
    elif color_space == 'Lab':
        intensity = get_Lab(ret)
    return intensity
