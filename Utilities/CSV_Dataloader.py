# -*- coding: utf-8 -*-
"""
    load Data from csv file
"""
import os
from collections import Counter
import numpy as np


def csv_dataloader(input_file):
    """

    :param input_file:
    :return:
    """
    X = []
    y = []
    with open(input_file, 'r') as f_in:
        line = f_in.readline()
        while line:
            if line.startswith('Flow'):
                line = f_in.readline()
            line_arr = line.split(',')
            X.append(line_arr[7:40])
            if line_arr[-1] == '2\n':
                y.append('1')
            else:
                y.append('0')

            line = f_in.readline()

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    print(Counter(y))

    return (X, y)


def mix_normal_attack_and_label(normal_file, attack_file, out_file='./mix_data.txt'):
    """

    :param normal_file: 0
    :param attack_file: 1
    :param out_file   :
    :return: (X,y), out_file
    """
    assert os.path.exists(normal_file)
    assert os.path.exists(attack_file)
    X = []
    y = []
    with open(normal_file, 'r') as file_in:
        line = file_in.readline()
        while line:
            if line.strip().startswith('ts'):
                print(line)
                line = file_in.readline()

            line_arr = line.split(',')
            X.append(line_arr[2:])
            y.append('0')

    with open(attack_file, 'r') as file_in:
        line = file_in.readline()
        while line:
            if line.strip().startswith('ts'):
                print(line)
                line = file_in.readline()

            line_arr = line.split(',')
            X.append(line_arr[2:])
            y.append('1')

    with open(out_file, 'w') as file_out:
        for i in range(len(y)):
            line = ','.join(X[i]) + y[i] + '\n'
            file_out.write(line)

    return (X, y), out_file
