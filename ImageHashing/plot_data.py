from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
import numpy as np
from typing import List, Union
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

OPERATION_TITLE_DICT = {
    'jpeg_compression': 'Quality factor',
    'watermarking': 'Strength',
    'speckle_noise': 'Variance',
    'salt_and_pepper_noise': 'Density',
    'brightness_adjustment': "Photoshop's scale",
    'contrast_adjustment': "Photoshop's scale",
    'gamma_correction': r'$\gamma$',
    'Gaussian_filtering': 'Standard deviation',
    'image_scaling': 'Ratio',
    'rotation_cropping_rescaling': 'Rotation angle'
}
LEGEND_STYLE = {
    'airplane': ('red', '*'),
    'baboon': ('pink', 'o'),
    'house': ('black', 's'),
    'lena': ('blue', '^')
}


def plot_line_chart(x: List[Union[int, float]], ys: List[List[float]],
                    legend: List[str], operation: str, path: Path = None):
    """
    Plot line chart with given values of the x axis and multiple values of different images of y axis.
    If needed, save the figure to a given directory.
    :param x: value series of x axis.
    :param ys: multiple value series (different image groups) of y axis.
    :param legend: names of each line.
    :param operation: the operation of the image to determine label of x axis.
    :param path: path to save the figure(optional).
    :return: None.
    """
    for i, y in enumerate(ys):
        lgd = legend[i]
        plt.plot(x, y, label=lgd.capitalize(),
                 color=LEGEND_STYLE[lgd][0],
                 marker=LEGEND_STYLE[lgd][1])
    plt.xlabel(OPERATION_TITLE_DICT[operation])
    plt.xticks(x)
    plt.ylabel('Correlation coefficient')
    # dynamically adjust ylim
    diff = np.max(ys) - np.min(ys)
    print('diff:', 1 - diff * 30)
    plt.ylim((max(np.round(1 - diff * 30, 3), 0), 1.0))
    plt.legend(loc='lower center')
    if path is not None:
        plt.savefig(path)
    # plt.show()
    plt.clf()


def plot_key_sensitivity(y: List[float], path: Path = None):
    """
    Plot line chart with given values of the x axis and multiple values of different images of y axis.
    If needed, save the figure to a given directory.
    :param y: value series (index of wrong keys) of y axis.
    :param path: path to save the figure(optional).
    :return: None.
    """

    x = np.arange(1, 101)
    plt.plot(x, y, marker='o', markerfacecolor='none')
    plt.xlabel('Index of wrong keys')
    plt.xticks(np.arange(0, 101, 10))
    plt.ylabel('Correlation coefficient')
    # determine height for the horizontal line
    hline = 0
    max_rho = np.max(y)
    for v in np.arange(0, 1, 0.05):
        if max_rho <= v:
            hline = v
            break
    plt.hlines(y=hline, xmin=0, xmax=100, colors='r', linestyles='dashed')

    if path is not None:
        plt.savefig(path)
    plt.show()
    plt.clf()


def plot_roc(ucid: List[Path], copydays: List[Path], legends: List[str], out: Path = None):
    """
    Draw ROC curve based on results on UCID and Copydays datasets.
    :param ucid: list of paths of UCID result files.
    :param copydays: list of paths of Copydays result files. The file at the same position should be the results
    under the same parameters correspondingly.
    :param legends: legends for the graph.
    :param out: path to save the figure.
    :return: None.
    """
    if len(ucid) != len(copydays):
        raise ValueError("The list of result files should have the same length!")
    style_lst = [('red', 'o'), ('blue', '*'), ('green', 's'), ('pink', '^')]

    fig, ax = plt.subplots()
    FPR_lst = []
    TPR_lst = []
    for i in range(len(ucid)):
        u_data = pd.read_csv(ucid[i]).rho
        c_data = pd.read_csv(copydays[i]).rho
        fprs = []
        tprs = []
        for T in np.arange(0.97, 0, step=-0.03):
            fpr = np.sum(len(np.where(u_data >= T)[0])) / len(u_data)
            tpr = np.sum(len(np.where(c_data >= T)[0])) / len(c_data)
            fprs.append(fpr)
            tprs.append(tpr)
        ax.plot(fprs, tprs, color=style_lst[i][0], label=legends[i], linestyle='-',
                marker=style_lst[i][1], linewidth=1, markersize=5, markerfacecolor='none')
        FPR_lst.append(fprs)
        TPR_lst.append(tprs)
    ax.set_xlabel('False positive rate')
    ax.set_xlim(0, 1)
    ax.set_ylabel('True positive rate')
    ax.set_ylim(0.4, 1)
    ax.legend()

    axins = inset_axes(ax, width="45%", height="45%", loc='lower left',
                       bbox_to_anchor=(0.5, 0.1, 1, 1),
                       bbox_transform=ax.transAxes)
    for i in range(len(FPR_lst)):
        print(f'{legends[i]}: {auc(FPR_lst[i], TPR_lst[i])}')
        axins.plot(FPR_lst[i], TPR_lst[i], color=style_lst[i][0], label=legends[i], linestyle='-',
                   marker=style_lst[i][1], linewidth=1, markersize=5, markerfacecolor='none')
    axins.set_xlim(1e-5, 0.1)
    axins.set_ylim(0.9, 1)
    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)

    if out is not None:
        plt.savefig(out)
    else:
        plt.show()
    plt.clf()


def plot_pr_curve():
    data = pd.read_csv('out/pr_result.csv')
    rhos = data.rho
    labels = data.label
    pre, rec, thr = precision_recall_curve(labels, rhos)
    pre = pre[:-1]
    rec = rec[:-1]

    fig, ax = plt.subplots()
    ax.plot(thr, pre, label='precision', color='green', linestyle='-',
            linewidth=1, markersize=2, markerfacecolor='none')
    ax.plot(thr, rec, label='recall', color='blue', linestyle='-',
            linewidth=1, markersize=2, markerfacecolor='none')
    ax.set_xlabel('threshold')
    ax.set_xlim(thr[0], 1)
    ax.set_ylabel('precision/recall')
    ax.set_ylim(0, 1)
    ax.legend()

    idx = np.argwhere(np.diff(np.sign(rec - pre))).flatten()
    ax.axvline(thr[idx[0]], 0, pre[idx[0]], linestyle='--', color='red', linewidth=2)
    print(f'thr: {thr[idx[0]]}\tpre: {pre[idx[0]]}')
    plt.savefig('out/pr_result.png')
    plt.clf()
