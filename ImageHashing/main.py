from preprocessing import preprocess
from feature_extraction import extract_features
from hashing import generate_hashing
from evaluation import calculate_similarity
import utils
from pathlib import Path
import numpy as np
from dataset_generation import OPERATION_PARAM_DCT
from plot_data import plot_line_chart, plot_key_sensitivity, plot_roc, plot_pr_curve
import csv
import pandas as pd
from typing import List, Union

# defined constants here
B = 512
b = 64
SECRET_KEY = 'test'


def cal_hashcode(file: Path, key: str = SECRET_KEY,
                 norm_img: int = B, sub_img: int = b, color_space: str = 'HSI') -> np.ndarray:
    """
    Aggregate operations for calculating hashcode from a given image file.
    :param file: path of the image.
    :param key: secret key for secret stream.
    :param norm_img: image size after normalization.
    :param sub_img: sub-image size when extracting features.
    :param color_space: the color space where the algorithm extracts features.
    :return: hashcode of the image.
    """
    print('Processing...')
    img = utils.cv2_imread(file)
    intensity = preprocess(img, norm_img, color_space=color_space)
    features = extract_features(intensity, sub_img)
    hashcode = generate_hashing(features, key)
    print(f'Get hashcode for image {file.name} now!')
    return hashcode


def validate_perceptual_robustness_on_USC_SIPI():
    # source for read image
    dataset = Path('data/USC-SIPI')
    raw_data = dataset.joinpath('raw')
    out = Path('out')
    # dict to save benchmark hash codes to avoid redundant calculation
    bench_hash_dict = {}

    for op_dir in dataset.iterdir():
        if (not op_dir.is_dir()) or op_dir.name == 'raw':
            continue

        # variables for plotting line chart
        legends = []
        rhos_list = []
        op = op_dir.name
        for file in raw_data.iterdir():
            # calculate hash code for the benchmark image
            if file.name not in bench_hash_dict.keys():
                hash_bench = cal_hashcode(file)
            else:
                hash_bench = bench_hash_dict[file.name]

            # calculate correlation coefficients between benchmark image and its copy versions
            name = file.name.split('.')[0]
            legends.append(name)

            rhos = []
            for img in op_dir.glob(f'{name}*'):
                hashcode = cal_hashcode(img)
                rho = calculate_similarity(hash_bench, hashcode)
                rhos.append(rho)
            rhos_list.append(rhos)
        plot_line_chart(x=OPERATION_PARAM_DCT[op], ys=rhos_list, legend=legends,
                        operation=op, path=out.joinpath(op + '.png'))
        print(f'\n{op_dir.name}: {np.min(rhos_list)}\n\n\n')


def validate_perceptual_robustness_on_Copydays(
        sub_size: int = b,
        save: Path = Path('out/copydays_result_64.csv'),
        color: str = 'HSI'):
    # source for read image
    dataset = Path('data/Copydays')
    raw_data = dataset.joinpath('raw')
    # dict to save benchmark hash codes to avoid redundant calculation
    bench_hash_dict = {}

    # open a file to save the result
    with open(save, 'a+') as res:
        writer = csv.writer(res)
        # write column names
        writer.writerow(['type', 'rho'])

        for op_dir in dataset.iterdir():
            if (not op_dir.is_dir()) or op_dir.name == 'raw':
                continue

            # variables for recording digital attack types
            op = op_dir.name
            for file in raw_data.iterdir():
                # calculate hash code for the benchmark image
                if file.name not in bench_hash_dict.keys():
                    hash_bench = cal_hashcode(file, sub_img=sub_size, color_space=color)
                else:
                    hash_bench = bench_hash_dict[file.name]

                # calculate correlation coefficients between benchmark image and its copy versions
                name = file.name.split('.')[0]

                for img in op_dir.glob(f'{name}*'):
                    hashcode = cal_hashcode(img, sub_img=sub_size, color_space=color)
                    rho = calculate_similarity(hash_bench, hashcode)
                    writer.writerow([op, rho])


def get_statistics_on_Copydays():
    data = pd.read_csv('out/copydays_result_64.csv')
    types = data.type
    rhos = data.rho
    dct = {}
    for i in range(len(rhos)):
        if types[i] not in dct.keys():
            dct[types[i]] = [rhos[i]]
        else:
            dct[types[i]].append(rhos[i])

    for k, v in dct.items():
        _min = np.min(v)
        _max = np.max(v)
        _median = np.median(v)
        _mean = np.mean(v)
        print(f'{k}: {_min}(min)\t{_max}(max)\t{_median}(median)\t{_mean}(mean)')


def analyze_discrimination_on_UCID(sub_size: int = b, save: Path = Path('out/ucid_result_64.csv'),
                                   color: str = 'HSI'):
    # source for read image
    dataset = Path('data/UCID')
    # dict to save benchmark hash codes to avoid redundant calculation
    bench_hash_list = []

    # calculate and save all the hashcodes of the images
    for file in dataset.iterdir():
        bench_hash_list.append(cal_hashcode(file, sub_img=sub_size, color_space=color))

    # open a file to save the result
    with open(save, 'a+') as res:
        writer = csv.writer(res)
        # write column names
        writer.writerow(['rho'])

        for i in range(len(bench_hash_list) - 1):
            for j in range(i + 1, len(bench_hash_list)):
                rho = calculate_similarity(bench_hash_list[i], bench_hash_list[j])
                writer.writerow([rho])


def get_statistics_on_UCID():
    data = pd.read_csv('out/ucid_result_64.csv')
    rhos = data.rho
    _min = np.min(rhos)
    _max = np.max(rhos)
    _median = np.median(rhos)
    _mean = np.mean(rhos)
    print(f'UCID: {_min}(min)\t{_max}(max)\t{_median}(median)\t{_mean}(mean)')


def get_fpr_tpr():
    copydays = pd.read_csv('out/copydays_result_64.csv')
    c_rhos = copydays.rho
    print(len(c_rhos))
    ucid = pd.read_csv('out/ucid_result_64.csv')
    u_rhos = ucid.rho
    print(len(u_rhos))

    Ts = np.linspace(0.84, 0.97, 14)
    print(Ts)
    for T in Ts:
        # print(f'n1: {len(np.where(u_rhos >= T)[0])}')
        fpr = len(np.where(u_rhos >= T)[0]) / len(u_rhos)
        # print(f'n2: {len(np.where(c_rhos >= T)[0])}')
        tpr = len(np.where(c_rhos >= T)[0]) / len(c_rhos)
        print(f'T: {T}\tFPR: {fpr}\tTPR: {tpr}')


def analyze_key_sensitivity():
    bench_image = Path('data/USC-SIPI/raw/airplane.tiff')
    bench_hash = cal_hashcode(bench_image)

    rhos = []
    for _ in range(100):
        rand_key = utils.generate_random_string()
        hash_code = cal_hashcode(bench_image, rand_key)
        rho = calculate_similarity(bench_hash, hash_code)
        rhos.append(rho)
        print(rho)

    plot_key_sensitivity(rhos, Path('out/key_sensitivity_figure.png'))


def effect_of_dominant_parameters():
    for sub_size in [16, 32, 128]:
        copydays_dst = Path(f'out/copydays_result_{sub_size}.csv')
        validate_perceptual_robustness_on_Copydays(sub_size, copydays_dst)
        ucid_dst = Path(f'out/ucid_result_{sub_size}.csv')
        analyze_discrimination_on_UCID(sub_size, ucid_dst)
    pass


def get_roc(params: List[Union[int, str]], name: str):
    ul = []
    cl = []
    lgd = []
    for i in params:
        ul.append(Path(f'out/ucid_result_{i}.csv'))
        cl.append(Path(f'out/copydays_result_{i}.csv'))
        # lgd.append(f'{i}×{i}')
        lgd.append(f'{i}')
    plot_roc(ul, cl, lgd, Path(f'out/roc_{name}.png'))


def analyze_color_space():
    for color_space in ['YCbCr', 'Lab']:
        copydays_dst = Path(f'out/copydays_result_{color_space}.csv')
        validate_perceptual_robustness_on_Copydays(save=copydays_dst, color=color_space)
        ucid_dst = Path(f'out/ucid_result_{color_space}.csv')
        analyze_discrimination_on_UCID(save=ucid_dst, color=color_space)
    pass


def copy_detection_precision_recall():
    src_database = Path('data/COREL/query_database')
    test_database = Path('data/COREL/test_images')
    query_database = {}
    for file in src_database.iterdir():
        hash_code = cal_hashcode(file)
        query_database[file.name] = hash_code
    with open('out/pr_result.csv', 'a+') as txt:
        writer = csv.writer(txt)
        # write column names
        writer.writerow(['rho', 'label'])
        for file in test_database.iterdir():
            hash_code = cal_hashcode(file)
            for name, bench in query_database.items():
                rho = calculate_similarity(hash_code, bench)
                label = 1 if name.split('.')[0] == file.name.split('_')[0] or name == file.name else 0
                writer.writerow([rho, label])
    pass


if __name__ == "__main__":
    # exp 1
    # validate_perceptual_robustness_on_USC_SIPI()

    # exp 2
    # validate_perceptual_robustness_on_Copydays()
    # get_statistics_on_Copydays()

    # exp 3
    # analyze_discrimination_on_UCID()
    # get_statistics_on_UCID()
    # get_fpr_tpr()

    # exp 4
    # effect_of_dominant_parameters()
    # get_roc([16, 32, 64, 128], 'subimage')

    # exp 5
    # analyze_key_sensitivity()

    # exp 6
    # analyze_color_space()
    # get_roc(['HSI', 'Lab', 'YCbCr'], 'ColorSpace')

    # exp 7
    # copy_detection_precision_recall()
    plot_pr_curve()

    # hash1 = cal_hashcode(Path('data/COREL/query_database/71.jpg'))
    # hash2 = cal_hashcode(Path('data/COREL/test_images/920.jpg'))
    # print(hash1)
    # print(np.mean(hash1))
    # print(hash2)
    # print(np.mean(hash2))
    # print(calculate_similarity(hash1, hash2))
    pass
