import modification as md
import os
from pathlib import Path
from typing import List, Union


OPERATION_PARAM_DCT = {
    'jpeg_compression': [30, 40, 50, 60, 70, 80, 90, 100],
    # 'watermarking': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'speckle_noise': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
    'salt_and_pepper_noise': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
    'brightness_adjustment': [-20, -10, 10, 20],
    'contrast_adjustment': [-20, -10, 10, 20],
    'gamma_correction': [0.75, 0.9, 1.1, 1.25],
    'Gaussian_filtering': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'image_scaling': [0.5, 0.75, 0.9, 1.1, 1.5, 2.0],
    'rotation_cropping_rescaling': [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
}


def operate(root: Path, operation: str, param_list: List[Union[int, float]]):
    src = root.joinpath('raw')
    dst = root.joinpath(operation)
    if not os.path.exists(dst):
        os.mkdir(dst)
    for file in src.iterdir():
        if operation == 'jpeg_compression':
            md.jpeg_compress(file, dst, param_list)
        elif operation == 'gamma_correction':
            md.gamma_correction(file, dst, param_list)
        elif operation == 'image_scaling':
            md.scale_image(file, dst, param_list)
        elif operation == 'Gaussian_filtering':
            md.Gaussian_filter(file, dst, param_list)
        elif operation == 'rotation_cropping_rescaling':
            md.rotate_image(file, dst, param_list)


if __name__ == '__main__':
    root = Path('data/COREL/query_database')
    for k, v in OPERATION_PARAM_DCT.items():
        operate(root, k, v)
