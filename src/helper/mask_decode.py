# -*- coding: utf-8 -*-


import numpy as np


def mask_decode(mask_list, height=240, width=320):
    """
    Decode a mask list to a mask image
    https://github.com/ccvl/clevr-refplus-dataset-gen
    """

    img = []
    cur = 0
    for num in mask_list.strip().split(','):
        num = int(num)
        img += [cur] * num
        cur = 1 - cur
    object_mask = np.array(img)

    return object_mask.reshape((height, width))

