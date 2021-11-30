import numpy as np
import hashlib


def generate_secret_stream(key: str, length: int = 80) -> np.ndarray:
    """
    This function is used to generate pseudo-random secret stream corresponding to the input key.
    :param key: secret key.
    :param length: length of hash sequence. The default value is 80.
    :return: pseudo-random stream that has fixed length 80, the elements of which are all between
    0 and 1.
    """
    # map key to an integer using hashing to generate seed for pseudo-random stream generator
    h = hashlib.blake2b(digest_size=4)
    h.update(key.encode())
    seed = int(h.hexdigest(), 16)

    # generate secret stream
    np.random.seed(seed)
    stream = np.random.rand(length)
    h = None

    return stream


def generate_hashing(feature: np.ndarray, key: str) -> np.ndarray:
    """
    Generate image hash code according to features extracted from the image and a given secret key.
    :param feature: features extracted from the image.
    :param key: secret key.
    :return: hash code of the image containing len(features) integer values.
    """
    stream = generate_secret_stream(key, len(feature))
    vector_z = np.argsort(stream)
    hash_code = np.round(feature[vector_z])
    return hash_code


if __name__ == '__main__':
    generate_hashing(feature=np.array([]), key='test')
