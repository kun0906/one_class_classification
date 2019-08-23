# -*- coding: utf-8 -*-
"""
    useful tools

    several standard data normalization techniques such as min-max, softmax, z-score, decimal scaling, box-cox and etc
"""
import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from utils.CSV_Dataloader import csv_dataloader, open_file


def normalize_data(np_arr, eplison=10e-4):
    """

    :param np_arr:
    :param eplison: handle with 0.
    :return:
    """
    min_val = np.min(np_arr, axis=0)  # X
    max_val = np.max(np_arr, axis=0)
    range_val = (max_val - min_val)
    if not range_val.all():  # Returns True if all elements evaluate to True.
        for i in range(len(range_val)):
            if range_val[i] == 0.0:
                range_val[i] += eplison
    print('range_val is ', range_val)
    norm_data = (np_arr - min_val) / range_val

    return norm_data


def normalizate_data_with_sigmoid(np_arr, eplison=10e-4):
    """

    :param np_arr:
    :param eplison: handle with 0.
    :return:
    """
    min_val = np.min(np_arr, axis=0)  # X
    max_val = np.max(np_arr, axis=0)
    range_val = (max_val - min_val)
    if not range_val.all():  # Returns True if all elements evaluate to True.
        for i in range(len(range_val)):
            if range_val[i] == 0.0:
                range_val[i] += eplison
    print('range_val is ', range_val)
    norm_data = []
    for i in range(np_arr.shape):
        norm_data.append(list(map(lambda x: 1 / (1 + np.exp(-(x))), np_arr)))
    # norm_data = list(map(lambda x: np.exp(-(x)), np_arr))
    norm_data = np.asarray(norm_data, dtype=float)

    return norm_data

def normalizate_data_with_u_std(np_arr, u_std_dict={'u': 0.5, 'std': 1.0}):
    """

    :param np_arr:
    :param u_std_dict: {'u':0.5,'std':1.0}
    :return:
    """
    # u_val = np.mean(np_arr, axis=0)  # X
    # std_val = np.std(np_arr, axis=0)

    norm_data = (np_arr - u_std_dict['u']) / u_std_dict['std']

    return norm_data

def split_data():
    # train_tset_split()
    pass

