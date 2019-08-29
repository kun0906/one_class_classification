import os
from collections import Counter
from copy import deepcopy
import numpy as np


def balance_dict(value_dict={}):
    """

    :param value_dict: {'X':, 'y'}
    :return:
    """

    new_dict = deepcopy(value_dict)
    d_tmp = [v for k, v in Counter(value_dict['y']).items()]  # (k, v)
    print(f'', Counter(value_dict['y']))
    min_v = min(d_tmp)
    x_value = value_dict['X']
    y_value = value_dict['y']

    new_x = {'x_norm': [], 'x_attack': []}
    new_y = {'y_norm': [], 'y_attack': []}
    for x, y in zip(x_value, y_value):
        if int(y) == 1:
            new_x['x_norm'].append(x)
            new_y['y_norm'].append(y)
        elif int(y) == 0:
            new_x['x_attack'].append(x)
            new_y['y_attack'].append(y)

    x_train = np.concatenate((np.asarray(new_x['x_norm'], dtype=float)[:min_v, :],
                              np.asarray(new_x['x_attack'], dtype=float)[:min_v, :]), axis=0)
    y_train = np.concatenate((np.asarray(new_y['y_norm'], dtype=int)[:min_v],
                              np.asarray(new_y['y_attack'], dtype=int)[:min_v]), axis=0)
    new_dict['X'] = x_train
    new_dict['y'] = y_train

    return new_dict


def concat_path(output_dir = '', file_path=''):

    return os.path.join(output_dir, file_path)