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
                continue
            line_arr = line.split(',')
            X.append(line_arr[:-1])
            # X.append(line_arr[7:40])
            # if line_arr[-1] == '2\n':
            #     y.append('1')
            # else:
            #     y.append('0')
            y.append(line_arr[-1].strip())
            line = f_in.readline()

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    print('input_data size is ', Counter(y))

    return (X, y)


def open_file(input_file, label='0'):
    """

    :param input_file:
    :param label:
    :return:
    """
    X = []
    y = []
    with open(input_file, 'r') as file_in:
        line = file_in.readline()
        while line:
            if line.strip().startswith('ts'):
                print(line.strip())
                line = file_in.readline()
                continue

            line_arr = line.strip().split(',')
            X.append(line_arr[3:])
            y.append(label)
            line = file_in.readline()

    return X, y


def mix_normal_attack_and_label(normal_file, attack_file, out_file='./mix_data.csv'):
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
    X_normal, y_normal = open_file(normal_file, label='0')
    X.extend(X_normal)
    y.extend(y_normal)

    X_attack, y_attack = open_file(attack_file, label='1')
    X.extend(X_attack)
    y.extend(y_attack)

    with open(out_file, 'w') as file_out:
        for i in range(len(y)):
            line = ','.join(X[i]) + ',' + y[i] + '\n'
            file_out.write(line)

        file_out.flush()

    return (X, y), out_file
