import numpy as np


def calculate_similarity(hash1: np.ndarray, hash2: np.ndarray):
    """
    Calculate correlation coefficient between hash codes of two different images.
    :param hash1: hash code of one image.
    :param hash2: hash code of the other image.
    :return: similarity.
    """
    mu1 = np.mean(hash1)
    mu2 = np.mean(hash2)
    epsilon = 1e-20

    numerator = np.sum((hash1 - mu1) * (hash2 - mu2))
    denominator = np.sqrt(np.sum((hash1 - mu1) ** 2) * np.sum((hash2 - mu2) ** 2)) + epsilon
    # denominator = np.sqrt(np.sum((hash1 - mu1) ** 2) * np.sum((hash2 - mu2) ** 2))
    similarity = numerator / denominator

    return similarity
