# -*- coding: utf-8 -*-
"""
    useful tools
"""
import os
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from Utilities.CSV_Dataloader import csv_dataloader


def normalizate_data(np_arr, eplison=10e-4):
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


def split_data():
    # train_tset_split()
    pass


def load_data(input_data='', norm_flg=True, train_val_test_percent=[0.7 * 0.9, 0.7 * 0.1, 0.3]):
    """

    :param input_data:
    :param norm_flg: default True
    :param train_val_test_percent: train_set = 0.7*0.9, val_set = 0.7*0.1, test_set = 0.3
    :return:
    """
    if 'mnist' in input_data:
        from Utilities.Mnist_data_loader import MNIST_DataLoader
        # load Data with Data loader
        dataset = MNIST_DataLoader(ad_experiment=1)
        train_set, val_set, test_set = (dataset._X_train, dataset._y_train), (dataset._X_val, dataset._y_val), (
            dataset._X_test, dataset._y_test)
    elif 'csv' in input_data:
        # train_set, val_set, test_set = csv_dataloader(input_data,norm_flg)
        (X, y) = csv_dataloader(input_data)
        if norm_flg:
            X = normalizate_data(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_val_test_percent[-1], random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=train_val_test_percent[1],
                                                          random_state=1)
        train_set, val_set, test_set = (X_train, y_train), (X_val, y_val), (X_test, y_test)

    else:
        print('error dataset.')
        return -1

    # if norm_flg:
    #     train_set = (normalizate_data(train_set[0]),train_set[1]) # X, y
    #     val_set=(normalizate_data(val_set[0]),val_set[1])
    #     test_set=(normalizate_data(test_set[0]),test_set[1])

    return train_set, val_set, test_set


def load_data_with_new_principle(input_data='', norm_flg=True, train_val_test_percent=[0.7 * 0.9, 0.7 * 0.1, 0.3]):
    """
    Case1:
        sess_normal_0 + sess_TDL4_HTTP_Requests_0
    Case2:
        sess_normal_0  + sess_Rcv_Wnd_Size_0_0

    Case1 and Case 2:
        Train set : (0.7 * all_normal_data)*0.9
        Val_set: (0.7*all_normal_data)*0.1 + 0.1*all_abnormal_data
        Test_set: 0.3*all_normal_data+ 0.9*all_abnormal_data

    :param input_data:
    :param norm_flg: default True
    :param train_val_test_percent: train_set = 0.7*0.9, val_set = 0.7*0.1, test_set = 0.3
    :return:
    """
    if 'mnist' in input_data:
        from Utilities.Mnist_data_loader import MNIST_DataLoader
        # load Data with Data loader
        dataset = MNIST_DataLoader(ad_experiment=1)
        train_set, val_set, test_set = (dataset._X_train, dataset._y_train), (dataset._X_val, dataset._y_val), (
            dataset._X_test, dataset._y_test)
    elif 'csv' in input_data:
        # train_set, val_set, test_set = csv_dataloader(input_data,norm_flg)
        (X, y) = csv_dataloader(input_data)
        if norm_flg:
            X = normalizate_data(X)

        lab = Counter(y)
        len_normal, len_abnormal = lab[0], lab[1]
        # X_normal=[]
        # y_normal=[]
        # X_abnormal=[]
        # y_abnormal=[]
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        X_test = []
        y_test = []
        train_set_size = 0
        val_set_size = 0
        test_set_size = 0
        for i in range(len(y)):
            if y[i] == 1:
                # X_abnormal.append(X[i])
                # y_abnormal.append(y[i])
                if test_set_size < int(len_abnormal * 0.9):
                    X_test.append(X[i])
                    y_test.append(y[i])
                    test_set_size += 1
                else:
                    X_val.append(X[i])
                    y_val.append(y[i])
            elif y[i] == 0:
                # X_normal.append(X[i])
                # y_normal.append(y[i])
                if train_set_size < int(len_normal * 0.7 * 0.9):
                    X_train.append(X[i])
                    y_train.append(y[i])
                    train_set_size += 1
                elif val_set_size < int(len_normal * 0.7 * 0.1):
                    X_val.append(X[i])
                    y_val.append(y[i])
                    val_set_size += 1
                else:
                    X_test.append(X[i])
                    y_test.append(y[i])
            else:
                pass
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=int)
        X_val = np.asarray(X_val, dtype=float)
        y_val = np.asarray(y_val, dtype=int)
        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test, dtype=int)
        #
        # len_train_set = int(len(y_normal)*0.7)
        # len_val_set = int(len_train_set * 0.1)
        # X_train = np.asarray(X_normal[:len_train_set-len_val_set],dtype=float)
        # y_train = np.asarray(y_normal[:len_train_set-len_val_set], dtype = int)
        # len_test_set = int(len(y_abnormal)*0.9)
        # X_test = np.asarray(X_abnormal[:len_test_set].extend(X_normal[len_train_set:]), dtype=float)
        # y_test = np.asarray(y_abnormal[:len_test_set].extend(y_normal[len_train_set:]),dtype=int)
        # X_val = np.asarray(X_normal[len_train_set-len_val_set:len_train_set].extend(X_abnormal[len_test_set:]),dtype=float)
        # y_val = np.asarray(y_normal[len_train_set-len_val_set:len_train_set].extend(y_abnormal[len_test_set:]), dtype=int)

        # X_train, X_test, y_train, y_test = train_test_split(X_normal, y_normal, test_size=train_val_test_percent[-1], random_state=1)
        # train_set = (X_train,y_train)
        # X_val, X_test, y_val, y_test = train_test_split(X_abnormal, y_abnormal, test_size=0.9,random_state=1)
        # test_set=(test_set+)

        train_set, val_set, test_set = (X_train, y_train), (X_val, y_val), (X_test, y_test)

    else:
        print('error dataset.')
        return -1

    # if norm_flg:
    #     train_set = (normalizate_data(train_set[0]),train_set[1]) # X, y
    #     val_set=(normalizate_data(val_set[0]),val_set[1])
    #     test_set=(normalizate_data(test_set[0]),test_set[1])

    return train_set, val_set, test_set

def dump_model(model, out_file):
    """
        save model to disk
    :param model:
    :param out_file:
    :return:
    """
    out_dir = os.path.split(out_file)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(out_file, 'wb') as f:
        pickle.dump(model, f)

    print("Model saved in %s" % out_file)


def load_model(input_file):
    """

    :param input_file:
    :return:
    """
    print("Loading model...")
    with open(input_file, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded.")

    return model


def show_data(data, x_label='epochs', y_label='y', fig_label='', title=''):
    plt.figure()
    plt.plot(data, 'r', alpha=0.5, label=fig_label)
    plt.legend(loc='upper right')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def evaluate_model(model, test_set, iters=10,
                   fig_params={'x_label': 'evluation times', 'y_label': 'accuracy on val set',
                               'fig_label': 'acc', 'title': 'accuracy on val set'}):
    """

    :param model:
    :param test_set:
    :param iters: evluation times
    :return:
    """
    assert len(test_set) > 0

    if iters == 1:
        thres = model.T.data
        print("\nEvaluation:%d/%d threshold = %f" % (1, iters, thres))
        test_set_acc, test_set_cm = model.evaluate(test_set, threshold=thres)
        max_acc_thres = (test_set_acc, thres)
    else:
        i = 0
        test_acc_lst = []
        thres_lst = []
        max_acc_thres = (0.0, 0.0)  # (acc, thres)
        for thres in np.linspace(start=10e-3, stop=(model.T.data) * 10, num=iters, endpoint=True):
            i += 1
            print("\nEvaluation:%d/%d threshold = %f" % (i, iters, thres))
            test_set_acc, test_set_cm = model.evaluate(test_set, threshold=thres)
            test_acc_lst.append(test_set_acc)
            thres_lst.append(thres)
            if test_set_acc > max_acc_thres[0]:
                max_acc_thres = (test_set_acc, thres)

        if model.show_flg:
            show_data(data=test_acc_lst, x_label=fig_params['x_label'], y_label=fig_params['y_label'], fig_label='acc',
                      title=fig_params['title'])
            show_data(data=thres_lst, x_label='evluation times', y_label='threshold', fig_label='thresholds',
                      title='thresholds variation')

    return max_acc_thres

def get_variable_name(data_var):
    """
        get variable name as string
    :param data_var:
    :return:
    """
    name = ''
    keys = locals().keys()
    for key, val in locals().items():
        # if id(key) == id(data_var):
        print(key, id(key), id(data_var), key is data_var)
        # if id(key) == id(data_var):
        if val == data_var:
            name = key
            break

    return name
