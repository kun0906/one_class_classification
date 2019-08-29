"""
    load features data from txt
"""
import os
from collections import OrderedDict, Counter
from copy import deepcopy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# from utils.visualization import vis_high_dims_data_umap


def balance_data(x_norm_train_DT, y_norm_train_DT, x_attack_train_DT, y_attack_train_DT, random_state=42):
    min_size = x_norm_train_DT.shape[0]
    if min_size > x_attack_train_DT.shape[0]:
        min_size = x_attack_train_DT.shape[0]
        x_train_DT = np.concatenate([shuffle(x_norm_train_DT, random_state=random_state)[:min_size], x_attack_train_DT])
        y_train_DT = np.concatenate([y_norm_train_DT[:min_size], y_attack_train_DT])
    else:
        x_train_DT = np.concatenate([x_norm_train_DT, shuffle(x_attack_train_DT, random_state=random_state)[:min_size]])
        y_train_DT = np.concatenate([y_norm_train_DT, y_attack_train_DT[:min_size]])
    print(f'\nWith data balance, x_train.shape: {x_train_DT.shape}')
    print(
        f' in which, x_norm_train_DT.shape: {x_norm_train_DT[:min_size].shape}, and x_attack_train_DT.shape: {x_attack_train_DT[:min_size].shape}')
    return x_train_DT, y_train_DT


def discretize_features(features_arr=[]):
    # features=[]
    # for idx, feat in enumerate(features_arr):
    #     if idx

    features = []
    if features_arr[0] == '6':  # 6: tcp
        features.extend([1, 0])  # one hot: tcp and udp
    else:  # features_arr[0] == '17':  # 17: udp
        features.extend([0, 1])

    features.extend(features_arr[1:])

    return features


def print_dict(train_set_dict):
    """

    :param train_set_dict:  train_set_dict = {'train_set': {'X': x_train, 'y': y_train}}
    :return:
    """
    for key, value in train_set_dict.items():
        print(f'{key}:', end='')
        x = value['X']
        y = value['y']
        print(f'x.shape: {x.shape}, y.shape: {Counter(y.reshape(-1,))}')


def load_data_from_txt(input_file, data_range=(0, -1), label=1, discretize_flg=True):
    """
        1) load data from txt
        2) label data
        3) discretize feature if discretize_flg == True.

        features: ts, sip, dip, sport, dport, proto, dura, orig_pks, reply_pks, orig_bytes, reply_bytes, orig_min_pkt_size, orig_max_pkt_size, reply_min_pkt_size, reply_max_pkt_size, orig_min_interval, orig_max_interval, reply_min_interval, reply_max_interval, orig_min_ttl, orig_max_ttl, reply_min_ttl, reply_max_ttl, urg, ack, psh, rst, syn, fin, is_new, state, prev_state
            idx : 0    1    2    3      4      5      6      7         8          9            10            11                 12                  13                  14                       15            16                      17               18                19          20             21           22            23   24   25   26   27   28   29      30      31
    :param input_file:
    :param range: ('start=0', 'end=77989'):
    :param discretize_flg: label all data, default: 1.
    :return:
    """
    print(f'input_file: {input_file}, data_range: {data_range}, label: {label}, discretize_flg: {discretize_flg}')
    start, end = data_range
    x = []
    cnt = 0
    with open(input_file, 'r') as hdl:
        line = hdl.readline()
        while line != '' and cnt < end:
            if line.startswith('ts'):
                line = hdl.readline()
                continue
            if (cnt >= start) and (cnt < end):
                if discretize_flg:
                    arr = line.split(',')[5:]
                    features = discretize_features(arr)
                    x.append(features)  # without : "ts, sip, dip, sport, dport"
                else:
                    x.append(line.split(',')[5:])  # without : "ts, sip, dip, sport, dport"
            line = hdl.readline()
            cnt += 1

    y = np.ones(len(x)) * int(label)  # if label = 0, y = 0
    data_dict = {'X': np.asarray(x, dtype=float), 'y': y}

    return data_dict


def sample_from_data(arr='', sample_num=10, shuffle_flg=True, random_state=42):
    """

    :param arr:
    :param sample_num:
    :param shuffle_flg:
    :param random_state:
    :return:
    """
    if sample_num > len(arr):
        print(f'sample_num ({sample_num}) is larger than array size ({len(arr)}), please modify it.')
        return -1

    if shuffle_flg:
        shuffled_arr = shuffle(arr, random_state=random_state)
        sampled_arr = shuffled_arr[:sample_num, :]
        remained_arr = shuffled_arr[sample_num:, :]
    else:
        sampled_arr = arr[:sample_num, :]
        remained_arr = arr[sample_num:, :]

    return sampled_arr, remained_arr


def sample_from_attack(x_attack_11, x_attack_train_size, x_attack_val_size, x_attack_test_size, all_attack_size,
                       shuffle_flg=True, random_state=42):
    x_attack_11_train_size = int((len(x_attack_11) / all_attack_size) * x_attack_train_size)
    x_attack_11_train, x_attack_11_remained = sample_from_data(x_attack_11, sample_num=x_attack_11_train_size,
                                                               shuffle_flg=shuffle_flg,
                                                               random_state=random_state)

    x_attack_11_val_size = int((len(x_attack_11) / all_attack_size) * x_attack_val_size)
    x_attack_11_val, x_attack_11_remained = sample_from_data(x_attack_11_remained, sample_num=x_attack_11_val_size,
                                                             shuffle_flg=shuffle_flg,
                                                             random_state=random_state)
    x_attack_11_test_size = int((len(x_attack_11) / all_attack_size) * x_attack_test_size)
    x_attack_11_test, x_attack_11_remained = sample_from_data(x_attack_11_remained, sample_num=x_attack_11_test_size,
                                                              shuffle_flg=shuffle_flg, random_state=random_state)

    return x_attack_11_train, x_attack_11_val, x_attack_11_test, x_attack_11_remained


def sample_from_attacks_with_equal_ratio(x_attack_11, x_attack_12, x_attack_13, x_attack_14, x_norm_test,
                                         x_norm_test_size, x_norm_train,
                                         x_norm_train_size, x_norm_val, x_norm_val_size, y_norm_test, y_norm_train,
                                         y_norm_val, attack_percent=0.01):
    '''
      # insert 5% attack to train set, val set. normal:attack= 95:5
    '''
    x_attack_train_size = int(x_norm_train_size / (1 - attack_percent) * attack_percent)
    x_attack_val_size = int(x_norm_val_size / (1 - attack_percent) * attack_percent)
    x_attack_test_size = int(x_norm_test_size / (1 - attack_percent) * attack_percent)
    all_attack_size = len(x_attack_11) + len(x_attack_12) + len(x_attack_13) + len(x_attack_14)
    x_attack_11_train, x_attack_11_val, x_attack_11_test, x_attack_11_remained = sample_from_attack(x_attack_11,
                                                                                                    x_attack_train_size,
                                                                                                    x_attack_val_size,
                                                                                                    x_attack_test_size,
                                                                                                    all_attack_size,
                                                                                                    shuffle_flg=True,
                                                                                                    random_state=42)
    x_attack_12_train, x_attack_12_val, x_attack_12_test, x_attack_12_remained = sample_from_attack(x_attack_12,
                                                                                                    x_attack_train_size,
                                                                                                    x_attack_val_size,
                                                                                                    x_attack_test_size,
                                                                                                    all_attack_size,
                                                                                                    shuffle_flg=True,
                                                                                                    random_state=42)
    x_attack_13_train, x_attack_13_val, x_attack_13_test, x_attack_13_remained = sample_from_attack(x_attack_13,
                                                                                                    x_attack_train_size,
                                                                                                    x_attack_val_size,
                                                                                                    x_attack_test_size,
                                                                                                    all_attack_size,
                                                                                                    shuffle_flg=True,
                                                                                                    random_state=42)
    x_attack_14_train, x_attack_14_val, x_attack_14_test, x_attack_14_remained = sample_from_attack(x_attack_14,
                                                                                                    x_attack_train_size,
                                                                                                    x_attack_val_size,
                                                                                                    x_attack_test_size,
                                                                                                    all_attack_size,
                                                                                                    shuffle_flg=True,
                                                                                                    random_state=42)
    x_attack_train = np.concatenate(
        [x_attack_11_train, x_attack_12_train, x_attack_13_train, x_attack_14_train], axis=0)
    x_attack_val = np.concatenate([x_attack_11_val, x_attack_12_val, x_attack_13_val, x_attack_14_val], axis=0)
    x_attack_test = np.concatenate([x_attack_11_test, x_attack_12_test, x_attack_13_test, x_attack_14_test],
                                   axis=0)
    x_attack_test_remained = np.concatenate(
        [x_attack_11_remained, x_attack_12_remained, x_attack_13_remained, x_attack_14_remained], axis=0)
    x_train = np.concatenate([x_norm_train, x_attack_train], axis=0)
    y_attack_train = np.zeros(shape=[x_attack_train.shape[0], 1])
    y_train = np.concatenate([y_norm_train, y_attack_train], axis=0)
    x_val = np.concatenate([x_norm_val, x_attack_val], axis=0)
    y_attack_val = np.zeros(shape=[x_attack_val.shape[0], 1])
    y_val = np.concatenate([y_norm_val, y_attack_val], axis=0)
    x_test = np.concatenate([x_norm_test, x_attack_test], axis=0)
    y_attack_test = np.zeros(shape=[x_attack_test.shape[0], 1])
    y_test = np.concatenate([y_norm_test, y_attack_test], axis=0)
    x_remained_attack = x_attack_test_remained
    y_remained_attack = np.zeros(shape=[x_remained_attack.shape[0], 1])
    return x_remained_attack, x_test, x_train, x_val, y_remained_attack, y_test, y_train, y_val


def discrete_feature(data):
    """
        discrete categorical features (such as protocol (Only TCP and UDP)).
    :param data:
    :return:
    """


def sampling_from_dict(data_dict, test_set_percent=0.1, shuffle_flg=False, random_state=42):
    """

    :param data_dict: {'X':x, 'y':y}
    :param test_set_percent:
    :param shuffle_flg:
    :param random_state:
    :return:
    """

    x_train, x_test, y_train, y_test = train_test_split(data_dict['X'], data_dict['y'], test_size=test_set_percent,
                                                        shuffle=shuffle_flg, random_state=random_state)

    data_train_dict = {'X': x_train, 'y': y_train}
    data_test_dict = {'X': x_test, 'y': y_test}

    return data_train_dict, data_test_dict


def merge_dict(dict_1, dict_2):
    """

    :param dict_1:  {'X':, 'y':}
    :param dict_2: {'X':, 'y':}
    :return:
    """

    new_dict = OrderedDict(deepcopy(dict_1))

    for key, value in dict_2.items():
        if key == 'y':
            dict_1[key] = dict_1[key].reshape(-1, )
            dict_2[key] = dict_2[key].reshape(-1, )
        new_dict[key] = np.concatenate([dict_1[key], dict_2[key]], axis=0)  # concatenate by rows.

    return new_dict


def get_each_dataset(norm_dict, attack_dict_lst=[], name='synt', test_set_percent=0.2, random_state=42,
                     shuffle_flg=False):
    """

    :param norm_dict:
    :param attack_dict_lst:  [attack_11_dict, attack_12_dict, attack_13_dict, attack_14_dict]
    :param test_set_percent:
    :param random_state:
    :return:
    """
    ### shuffle_flg = False for normal data, means split data without shuffling them.
    norm_train_dict, norm_test_dict = sampling_from_dict(data_dict=norm_dict,
                                                         test_set_percent=test_set_percent, shuffle_flg=False,
                                                         random_state=random_state)

    remained_size = len(norm_train_dict['y'])
    test_set_percent = 1 - 1 / (7 + 1)  # # train:val:test=7:1:2
    norm_train_dict, norm_val_dict = sampling_from_dict(data_dict=norm_train_dict,
                                                        test_set_percent=test_set_percent,
                                                        shuffle_flg=False, random_state=random_state)

    if len(attack_dict_lst) == 0:
        train_set_dict = norm_train_dict
        val_set_dict = norm_val_dict
        test_set_dict = norm_test_dict
        train_set_dict = {f'{name}_train_set': train_set_dict}  # train set only has one (1).
        val_set_dict = {f'{name}_val_set': val_set_dict}  # val set only has one (1).
        test_set_dict = {f'{name}_test_set': test_set_dict}  # test set only has one (1).
    else:
        ###########################################################################################################
        ### test set 1
        ### shuffle_flg = True for attack data
        attack_train_dict = {}
        attack_val_dict = {}
        attack_test_dict = {}
        shuffle_flg = shuffle_flg
        for idx, value_dict in enumerate(attack_dict_lst):
            train_tmp_dict, test_tmp_dict = sampling_from_dict(data_dict=value_dict,
                                                               test_set_percent=test_set_percent,
                                                               shuffle_flg=True, random_state=random_state)

            remained_tmp_size = len(train_tmp_dict['y'])
            test_set_percent = 1 - ((remained_tmp_size * 0.875) / remained_tmp_size)  # train_set:val_set = (7:3)*0.8
            train_tmp_dict, val_tmp_dict = sampling_from_dict(data_dict=train_tmp_dict,
                                                              test_set_percent=test_set_percent,
                                                              shuffle_flg=True, random_state=random_state)

            if idx == 0:
                attack_train_dict = train_tmp_dict
                attack_val_dict = val_tmp_dict
                attack_test_dict = test_tmp_dict
            else:
                attack_train_dict.update(train_tmp_dict)  # append, not overalp
                attack_val_dict.update(val_tmp_dict)  # append, not overalp
                attack_test_dict.update(test_tmp_dict)  # append, not overalp

        ### test set 2.
        # might has more than more test set 1

        # concatenate normal and attack.
        train_dict = merge_dict(norm_train_dict, attack_train_dict)
        train_set_dict = {f'{name}_train_set': train_dict}  # train set only has one (1).

        val_dict = merge_dict(norm_val_dict, attack_val_dict)
        val_set_dict = {f'{name}_val_set': val_dict}  # val set might has more than 1.

        test_1_dict = merge_dict(norm_test_dict, attack_test_dict)
        test_set_dict = OrderedDict({f'{name}_test_set': test_1_dict})  # test sets might has more than 1.

    return train_set_dict, val_set_dict, test_set_dict


def select_attack_percent(attack_dict_lst='', total_attack_size='', norm_train_dict='', norm_val_dict='',
                          norm_test_dict='', insert_attack_percent='', random_state=42):
    """
        insert attack to train_set and val_set
    :param attack_dict_lst:
    :param total_attack_size:
    :param norm_train_dict:
    :param norm_val_dict:
    :param norm_test_dict:
    :param insert_attack_percent:
    :param random_state:
    :return:
    """
    norm_train_num = len(norm_train_dict['y'])
    norm_val_num = len(norm_val_dict['y'])
    norm_test_num = len(norm_test_dict['y'])
    print(
        f'insert_attack_percent:{insert_attack_percent}, i.e., attack samples is {insert_attack_percent} normal samples,\n'
        f'\tinsert_attack_train: {int(norm_train_num * insert_attack_percent)} = {norm_train_num} * {insert_attack_percent}: {int(norm_train_num * insert_attack_percent) == int(norm_train_num * insert_attack_percent)},\n'
        f'\tinsert_attack_val: {int(norm_val_num * insert_attack_percent)} = {norm_val_num} * {insert_attack_percent}: {int(norm_val_num * insert_attack_percent) == int(norm_val_num * insert_attack_percent)},\n'
        f'\tinsert_attack_test: {int(norm_test_num * insert_attack_percent)} = {norm_test_num} * {insert_attack_percent}: {int(norm_test_num * insert_attack_percent) == int(norm_test_num * insert_attack_percent)}')

    attack_train = 0
    attack_train_str = ''
    attack_val = 0
    attack_val_str = ''
    attack_test = 0
    attack_test_str = ''
    remained_attack = 0
    remained_attack_str = ''
    for idx, value_dict in enumerate(attack_dict_lst):
        # insert attack_sample_percent = 0.01 attack to train set.

        len_size = len(value_dict['y'])
        ### step 1. get attack for train
        ratio = (len(value_dict['y']) / total_attack_size)  # each attack set are selected with the correspending ratio.
        test_set_num = int((insert_attack_percent * len(norm_train_dict['y'])) * ratio)  #
        # print(f'attack train set num: {test_set_num}')
        if test_set_num == 0:
            test_set_num = 1
        test_set_percent = test_set_num / len_size  #
        train_all_tmp_dict, test_tmp_dict = sampling_from_dict(data_dict=value_dict,
                                                               test_set_percent=test_set_percent,
                                                               shuffle_flg=True, random_state=random_state)
        attack_train_num = len(test_tmp_dict['y'])
        print(
            f'\tidx:{idx}, ratio:{ratio:.3f}, (attack_train_num({attack_train_num})/norm_train_num({norm_train_num}))')
        if idx == 0:
            attack_train_dict = test_tmp_dict  # insert attack to train set.
        else:
            attack_train_dict = merge_dict(attack_train_dict, test_tmp_dict)  # append, not overalp
        attack_train += attack_train_num
        attack_train_str += str(attack_train_num) + '+'

        ### step 2. get attack for validation
        len_size = len_size - len(test_tmp_dict['y'])
        test_set_num = int((insert_attack_percent * len(norm_val_dict['y'])) * ratio)  # attack_percent = 0.01
        if test_set_num == 0:
            test_set_num = 1
        test_set_percent = test_set_num / len_size
        train_all_tmp_dict, test_tmp_dict = sampling_from_dict(data_dict=train_all_tmp_dict,
                                                               test_set_percent=test_set_percent,
                                                               shuffle_flg=True, random_state=random_state)
        attack_val_num = len(test_tmp_dict['y'])
        print(
            f'\tidx:{idx}, ratio:{ratio:.3f}, (attack_val_num({attack_val_num})/norm_val_num({norm_val_num}))')
        if idx == 0:
            attack_val_dict = test_tmp_dict  # insert attack to train set.
            attack_test_2_dict = train_all_tmp_dict  # remained attack for set
        else:
            attack_val_dict = merge_dict(attack_val_dict, test_tmp_dict)  # append, not overalp
            attack_test_2_dict = merge_dict(attack_test_2_dict, train_all_tmp_dict)  # append, not overalp
        attack_val += attack_val_num
        attack_val_str += str(attack_val_num) + '+'
        remained_attack += len(train_all_tmp_dict['y'])
        remained_attack_str += str(len(train_all_tmp_dict['y'])) + '+'

        ### step 3. get attack for test 1
        len_size = len_size - len(test_tmp_dict['y'])
        test_set_num = int((insert_attack_percent * len(norm_test_dict['y'])) * ratio)  #
        if test_set_num == 0:
            test_set_num = 1
        test_set_percent = test_set_num / len_size
        train_all_tmp_dict, test_tmp_dict = sampling_from_dict(data_dict=train_all_tmp_dict,
                                                               test_set_percent=test_set_percent,
                                                               shuffle_flg=True, random_state=random_state)
        attack_test_num = len(test_tmp_dict['y'])
        print(
            f'\tidx:{idx}, ratio:{ratio:.3f}, (attack_test_num({attack_test_num})/norm_test_num({norm_test_num}))')
        if idx == 0:
            attack_test_dict = test_tmp_dict  # insert attack to train set.
        else:
            attack_test_dict = merge_dict(attack_test_dict, test_tmp_dict)  # append, not overalp
        attack_test += attack_test_num
        attack_test_str += str(attack_test_num) + '+'

    print(f'after split, insert_attack_percent:{insert_attack_percent},\n'
          f'\tinsert_attack_train: {attack_train}({attack_train_str}) = {norm_train_num} * {insert_attack_percent},\n'
          f'\tinsert_attack_val: {attack_val}({attack_val_str}) = {norm_val_num} * {insert_attack_percent},\n'
          f'\tinsert_attack_test: {attack_test}({attack_test_str}) = {norm_test_num} * {insert_attack_percent},\n'
          f'\tremained_attack_test-attack_test: {remained_attack-attack_test}({remained_attack_str}-{attack_test_str}) = {total_attack_size - (attack_train+attack_val+attack_test)}, total_attack_size:{total_attack_size}')

    return attack_train_dict, attack_val_dict, attack_test_dict, attack_test_2_dict


def get_each_dataset_for_experiment_1(norm_dict='', attack_dict_lst=[], name='synt', insert_attack_percent=0.02,
                                      test_set_percent=0.2, random_state=42, shuffle_flg=False):
    """

    :param norm_dict:
    :param attack_dict_lst:  [attack_11_dict, attack_12_dict, attack_13_dict, attack_14_dict]
    :param test_set_percent:
    :param random_state:
    :return:
    """
    print(f'train:val:test=7:1:2. It is for normal data')
    ### shuffle_flg = False for normal data
    norm_train_dict, norm_test_dict = sampling_from_dict(data_dict=norm_dict,
                                                         test_set_percent=test_set_percent, shuffle_flg=False,
                                                         random_state=random_state)
    test_set_percent = 1 / (7 + 1)  # train:val = 7:1
    norm_train_dict, norm_val_dict = sampling_from_dict(data_dict=norm_train_dict,
                                                        test_set_percent=test_set_percent,
                                                        shuffle_flg=False, random_state=random_state)

    if len(attack_dict_lst) == 0:
        train_set_dict = norm_train_dict
        val_set_dict = norm_val_dict
        test_set_dict = norm_test_dict
        train_set_dict = {f'{name}_train_set': train_set_dict}  # train set only has one (1).
        val_set_dict = {f'{name}_val_set': val_set_dict}  # val set only has one (1).
        test_set_dict = {f'{name}_test_set': test_set_dict}  # test set only has one (1).
    else:
        ###########################################################################################################

        if insert_attack_percent > 0 and insert_attack_percent < 1:
            ### test set 1
            ### shuffle_flg = True for attack data
            shuffle_flg = shuffle_flg
            total_attack_size = 0
            line_str = ''
            tt = 0
            for idx, value_dict in enumerate(attack_dict_lst):
                tt += len(value_dict['y'])

            line_str_2 = ''
            for idx, value_dict in enumerate(attack_dict_lst):
                total_attack_size += len(value_dict['y'])
                tmp_len = len(value_dict['y'])
                if idx == len(attack_dict_lst) - 1:  # last value.
                    line_str += str(len(value_dict['y']))
                    line_str_2 += f'({tmp_len/tt:.4f})'
                else:
                    line_str += str(len(value_dict['y'])) + '+'
                    line_str_2 += f'({tmp_len/tt:.3f})' + ':'

            print(f'total_attack_size: {total_attack_size} ({line_str}=>{line_str_2}) of \'{name}\'')

            attack_train_dict, attack_val_dict, attack_test_dict, attack_test_2_dict = select_attack_percent(
                attack_dict_lst=attack_dict_lst, total_attack_size=total_attack_size, norm_train_dict=norm_train_dict,
                norm_val_dict=norm_val_dict, norm_test_dict=norm_test_dict, insert_attack_percent=insert_attack_percent,
                random_state=random_state)

            # concatenate normal and attack.
            train_dict = merge_dict(norm_train_dict, attack_train_dict)
            train_set_dict = {f'{name}_train_set': train_dict}  # train set only has one (1).

            val_dict = merge_dict(norm_val_dict, attack_val_dict)
            val_set_dict = {f'{name}_val_set': val_dict}  # val set might has more than 1.

            test_1_dict = merge_dict(norm_test_dict, attack_test_dict)  # attack_percent (0.01) attack samples
            test_set_dict = OrderedDict({f'{name}_test_set': test_1_dict})  # test sets might has more than 1.

            test_2_dict = merge_dict(norm_test_dict, attack_test_2_dict)
            test_set_dict.update({f'{name}_test_set_2': test_2_dict})  # all the remained attack samples

        elif insert_attack_percent == 0:
            print(f'insert_attack_percent: {insert_attack_percent}, so using all attack as test set')
            train_set_dict = norm_train_dict
            val_set_dict = norm_val_dict

            for idx, value_dict in enumerate(attack_dict_lst):
                # for ith, (key, value) in enumerate(value_dict.items()):
                if idx == 0:
                    attack_test_dict = value_dict
                else:
                    attack_test_dict = merge_dict(attack_test_dict, value_dict)

            test_set_dict = merge_dict(norm_test_dict, attack_test_dict)

            train_set_dict = {f'{name}_train_set': train_set_dict}  # train set only has one (1).
            val_set_dict = {f'{name}_val_set': val_set_dict}  # val set only has one (1).
            test_set_dict = {f'{name}_test_set': test_set_dict}  # test set only has one (1).
        else:
            print(f'insert_attack_percent:{insert_attack_percent} is not correct.')

    return train_set_dict, val_set_dict, test_set_dict


def get_dataset(case='uSc1C2_z-score_20_14', input_dir="input_data/dataset", shuffle_flg=False,
                random_state=42, test_set_percent=0.2, label_dict={'norm': 1, 'attack': 0}):
    """
        1) load data from txt, and label samples (Attack: 0, Normal: 1) or not.
        2) discretize feature or not.
        3) split data or not.

    :param case:
    :param input_dir:
    :param shuffle_flg:
    :param random_state:
    :param split_ratio_dict: {'train_set_percent':0.7, 'val_set_percent':0.1, 'test_set_percent':0.2}
    :return: datasets_dict = {'train_set_dict':, 'val_set_dict':, 'test_set_dict':}
    """

    '''
        case[0] = u/s for Unsupervised or Supervised
        case[3] = Scenario
        case[5] = Source
    This load data function is a  bit complicated.

    start = index of first data_point
    end = index of last data_point
    example : start = 10, end = 100 returns data points from [10,100]
    '''
    print(f'label: \'normal:1, attack:0\'')
    if case[3] == '1':  # Experiment 1
        if case[5] == '1':  # training and testing on SYNT (simulated data)
            ### all norm data
            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_normal_0.txt")
            norm_dict = load_data_from_txt(input_file=input_file, data_range=(0, 77989), label=label_dict['norm'])

            ### all attack data
            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt")
            attack_11_dict = load_data_from_txt(input_file=input_file, data_range=(0, 36000),
                                                label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Recursive_GET.dms")
            attack_12_dict = load_data_from_txt(input_file=input_file, data_range=(0, 37000),
                                                label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms")
            attack_13_dict = load_data_from_txt(input_file=input_file, data_range=(0, 243), label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms")
            attack_14_dict = load_data_from_txt(input_file=input_file, data_range=(0, 1000), label=label_dict['attack'])

            attack_dict_lst = [attack_11_dict, attack_12_dict, attack_13_dict, attack_14_dict]

            name = 'SYNT'

        elif case[5] == '2':  # training and testing on unb
            ### all norm data
            input_file = os.path.join(input_dir, "unb/Normal_UNB.txt")
            norm_dict = load_data_from_txt(input_file=input_file, data_range=(0, 59832), label=label_dict['norm'])

            ### all attack data
            input_file = os.path.join(input_dir, "unb/DoSHulk_UNB.txt")
            attack_21_dict = load_data_from_txt(input_file=input_file, data_range=(0, 11530),
                                                label=label_dict['attack'])

            input_file = input_file = os.path.join(input_dir, "unb/DOSSlowHttpTest_UNB.txt")
            attack_22_dict = load_data_from_txt(input_file=input_file, data_range=(0, 6414), label=label_dict['attack'])

            input_file = os.path.join(input_dir, "unb/UNB_DosGoldenEye_UNB_IDS2017.txt")
            attack_23_dict = load_data_from_txt(input_file=input_file, data_range=(0, 1268), label=label_dict['attack'])

            input_file = input_file = os.path.join(input_dir, "unb/UNB_DoSSlowloris_UNB_IDS2017.txt")
            attack_24_dict = load_data_from_txt(input_file=input_file, data_range=(0, 16741),
                                                label=label_dict['attack'])

            attack_dict_lst = [attack_21_dict, attack_22_dict, attack_23_dict, attack_24_dict]
            name = 'UNB'

        elif case[5] == '3':  # training and testing on mawi
            ### all norm data
            input_file = os.path.join(input_dir, "mawi/Normal_mawi_day1.txt")
            norm_dict = load_data_from_txt(input_file=input_file, data_range=(0, 62000), label=label_dict['norm'])

            attack_dict_lst = []
            name = 'MAWI'

        else:
            print('**** other case.')
            return -1

        train_set_dict, val_set_dict, test_set_dict = get_each_dataset_for_experiment_1(norm_dict=norm_dict,
                                                                                        attack_dict_lst=attack_dict_lst,
                                                                                        name=name,
                                                                                        insert_attack_percent=0.01,
                                                                                        test_set_percent=test_set_percent,
                                                                                        shuffle_flg=shuffle_flg,
                                                                                        random_state=random_state)
        print_dict(train_set_dict)
        print_dict(val_set_dict)
        print_dict(test_set_dict)

        return {'train_set_dict': train_set_dict, 'val_set_dict': val_set_dict, 'test_set_dict': test_set_dict}

    elif case[3] == '2':  # Experiment 2
        if case[5] == '1':  # train and validate on SYNT, test on SYNT, UNB and MAWI
            ### all attack data: SNYT
            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_normal_0.txt")
            synt_norm_dict = load_data_from_txt(input_file=input_file, data_range=(0, 77989), label=label_dict['norm'])

            ### all attack data_set 1 : SYNT
            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt")
            attack_11_dict = load_data_from_txt(input_file=input_file, data_range=(0, 36000),
                                                label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Recursive_GET.dms")
            attack_12_dict = load_data_from_txt(input_file=input_file, data_range=(0, 37000),
                                                label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms")
            attack_13_dict = load_data_from_txt(input_file=input_file, data_range=(0, 243), label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms")
            attack_14_dict = load_data_from_txt(input_file=input_file, data_range=(0, 1000), label=label_dict['attack'])

            synt_attack_lst = [attack_11_dict, attack_12_dict, attack_13_dict, attack_14_dict]
            synt_train_set_dict, synt_val_set_dict, synt_test_set_dict = get_each_dataset(norm_dict=synt_norm_dict,
                                                                                          attack_dict_lst=synt_attack_lst,
                                                                                          name='synt',
                                                                                          test_set_percent=test_set_percent,
                                                                                          shuffle_flg=shuffle_flg,
                                                                                          random_state=random_state)
            train_set_dict = synt_train_set_dict
            val_set_dict = synt_val_set_dict
            # test_set_dict.update(unb_test_set_dict)
            test_set_dict = synt_test_set_dict
            ###########################################################################################################
            ### add test_set 2 (unb test set)
            ### all norm data: UNB
            input_file = os.path.join(input_dir, "unb/Normal_UNB.txt")
            unb_norm_dict = load_data_from_txt(input_file=input_file, data_range=(0, 59832), label=label_dict['norm'])

            ### all attack data: UNB
            input_file = os.path.join(input_dir, "unb/DoSHulk_UNB.txt")
            attack_21_dict = load_data_from_txt(input_file=input_file, data_range=(0, 11530),
                                                label=label_dict['attack'])

            input_file = input_file = os.path.join(input_dir, "unb/DOSSlowHttpTest_UNB.txt")
            attack_22_dict = load_data_from_txt(input_file=input_file, data_range=(0, 6414), label=label_dict['attack'])

            input_file = os.path.join(input_dir, "unb/UNB_DosGoldenEye_UNB_IDS2017.txt")
            attack_23_dict = load_data_from_txt(input_file=input_file, data_range=(0, 1268), label=label_dict['attack'])

            input_file = input_file = os.path.join(input_dir, "unb/UNB_DoSSlowloris_UNB_IDS2017.txt")
            attack_24_dict = load_data_from_txt(input_file=input_file, data_range=(0, 16741),
                                                label=label_dict['attack'])

            unb_attack_lst = [attack_21_dict, attack_22_dict, attack_23_dict, attack_24_dict]
            unb_train_set_dict, unb_val_set_dict, unb_test_set_dict = get_each_dataset(norm_dict=unb_norm_dict,
                                                                                       attack_dict_lst=unb_attack_lst,
                                                                                       name='unb',
                                                                                       test_set_percent=test_set_percent,
                                                                                       shuffle_flg=shuffle_flg,
                                                                                       random_state=random_state)

            test_set_dict.update(unb_test_set_dict)

            ###########################################################################################################
            ### add test set 3 (mawi test set )
            ### all norm data: MAWI
            input_file = os.path.join(input_dir, "mawi/Normal_mawi_day1.txt")
            mawi_norm_dict = load_data_from_txt(input_file=input_file, data_range=(0, 62000), label=label_dict['norm'])

            mawi_attack_lst = []
            mawi_train_set_dict, mawi_val_set_dict, mawi_test_set_dict = get_each_dataset(norm_dict=mawi_norm_dict,
                                                                                          attack_dict_lst=mawi_attack_lst,
                                                                                          name='mawi',
                                                                                          test_set_percent=test_set_percent,
                                                                                          shuffle_flg=shuffle_flg,
                                                                                          random_state=random_state)

            test_set_dict.update(mawi_test_set_dict)

            print_dict(train_set_dict)
            print_dict(val_set_dict)
            print_dict(test_set_dict)

            return {'train_set_dict': train_set_dict, 'val_set_dict': val_set_dict, 'test_set_dict': test_set_dict}

        elif case[5] == '2':  # train and validate on UNB, test on SYNT, UNB and MAWI
            ### all norm data: UNB
            input_file = os.path.join(input_dir, "unb/Normal_UNB.txt")
            unb_norm_dict = load_data_from_txt(input_file=input_file, data_range=(0, 59832), label=label_dict['norm'])

            ### all attack data: UNB
            input_file = os.path.join(input_dir, "unb/DoSHulk_UNB.txt")
            attack_21_dict = load_data_from_txt(input_file=input_file, data_range=(0, 11530),
                                                label=label_dict['attack'])

            input_file = input_file = os.path.join(input_dir, "unb/DOSSlowHttpTest_UNB.txt")
            attack_22_dict = load_data_from_txt(input_file=input_file, data_range=(0, 6414), label=label_dict['attack'])

            input_file = os.path.join(input_dir, "unb/UNB_DosGoldenEye_UNB_IDS2017.txt")
            attack_23_dict = load_data_from_txt(input_file=input_file, data_range=(0, 1268), label=label_dict['attack'])

            input_file = input_file = os.path.join(input_dir, "unb/UNB_DoSSlowloris_UNB_IDS2017.txt")
            attack_24_dict = load_data_from_txt(input_file=input_file, data_range=(0, 16741),
                                                label=label_dict['attack'])

            unb_attack_lst = [attack_21_dict, attack_22_dict, attack_23_dict, attack_24_dict]
            unb_train_set_dict, unb_val_set_dict, unb_test_set_dict = get_each_dataset(norm_dict=unb_norm_dict,
                                                                                       attack_dict_lst=unb_attack_lst,
                                                                                       name='unb',
                                                                                       test_set_percent=test_set_percent,
                                                                                       shuffle_flg=shuffle_flg,
                                                                                       random_state=random_state)
            train_set_dict = unb_train_set_dict
            val_set_dict = unb_val_set_dict
            # test_set_dict.update(unb_test_set_dict)
            test_set_dict = unb_test_set_dict
            ###########################################################################################################
            ### test_set 2
            ### all attack data: SNYT
            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_normal_0.txt")
            synt_norm_dict = load_data_from_txt(input_file=input_file, data_range=(0, 77989), label=label_dict['norm'])

            ### all attack data_set 1 : SYNT
            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt")
            attack_11_dict = load_data_from_txt(input_file=input_file, data_range=(0, 36000),
                                                label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Recursive_GET.dms")
            attack_12_dict = load_data_from_txt(input_file=input_file, data_range=(0, 37000),
                                                label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms")
            attack_13_dict = load_data_from_txt(input_file=input_file, data_range=(0, 243), label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms")
            attack_14_dict = load_data_from_txt(input_file=input_file, data_range=(0, 1000), label=label_dict['attack'])

            ### test set: SYNT
            synt_attack_lst = [attack_11_dict, attack_12_dict, attack_13_dict, attack_14_dict]
            synt_train_set_dict, synt_val_set_dict, synt_test_set_dict = get_each_dataset(norm_dict=synt_norm_dict,
                                                                                          attack_dict_lst=synt_attack_lst,
                                                                                          name='synt',
                                                                                          test_set_percent=test_set_percent,
                                                                                          shuffle_flg=shuffle_flg,
                                                                                          random_state=random_state)

            test_set_dict.update(synt_test_set_dict)

            ###########################################################################################################
            ### test set 3
            ### all norm data: MAWI
            input_file = os.path.join(input_dir, "mawi/Normal_mawi_day1.txt")
            mawi_norm_dict = load_data_from_txt(input_file=input_file, data_range=(0, 62000), label=label_dict['norm'])

            mawi_attack_lst = []
            mawi_train_set_dict, mawi_val_set_dict, mawi_test_set_dict = get_each_dataset(norm_dict=mawi_norm_dict,
                                                                                          attack_dict_lst=mawi_attack_lst,
                                                                                          name='mawi',
                                                                                          test_set_percent=test_set_percent,
                                                                                          shuffle_flg=shuffle_flg,
                                                                                          random_state=random_state)

            test_set_dict.update(mawi_test_set_dict)

            print_dict(train_set_dict)
            print_dict(val_set_dict)
            print_dict(test_set_dict)

            return {'train_set_dict': train_set_dict, 'val_set_dict': val_set_dict, 'test_set_dict': test_set_dict}

        elif case[5] == '3':  # train and validate on MAWI, test on SYNT, UNB and MAWI
            # todo

            train_set_dict = OrderedDict()
            val_set_dict = OrderedDict()
            test_set_dict = OrderedDict()

            pass

    elif case[3] == '3':  # Experiment 3, compare AE and DT.
        if case[5] == '1':  # SYNT
            ### all attack data: SNYT
            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_normal_0.txt")
            synt_norm_dict = load_data_from_txt(input_file=input_file, data_range=(0, 77989), label=label_dict['norm'])

            ### all attack data_set 1 : SYNT
            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt")
            attack_11_dict = load_data_from_txt(input_file=input_file, data_range=(0, 36000),
                                                label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Recursive_GET.dms")
            attack_12_dict = load_data_from_txt(input_file=input_file, data_range=(0, 37000),
                                                label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms")
            attack_13_dict = load_data_from_txt(input_file=input_file, data_range=(0, 243), label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms")
            attack_14_dict = load_data_from_txt(input_file=input_file, data_range=(0, 1000), label=label_dict['attack'])

            synt_attack_lst = [attack_11_dict, attack_12_dict, attack_13_dict, attack_14_dict]
            synt_train_set_dict, synt_val_set_dict, synt_test_set_dict = get_each_dataset(norm_dict=synt_norm_dict,
                                                                                          attack_dict_lst=synt_attack_lst,
                                                                                          name='synt',
                                                                                          test_set_percent=test_set_percent,
                                                                                          shuffle_flg=shuffle_flg,
                                                                                          random_state=random_state)
            train_set_dict = synt_train_set_dict
            val_set_dict = synt_val_set_dict
            # test_set_dict.update(unb_test_set_dict)
            test_set_dict = synt_test_set_dict
            ###########################################################################################################
            ### add test_set 2 (unb test set)
            ### all norm data: UNB
            input_file = os.path.join(input_dir, "unb/Normal_UNB.txt")
            unb_norm_dict = load_data_from_txt(input_file=input_file, data_range=(0, 59832), label=label_dict['norm'])

            ### all attack data: UNB
            input_file = os.path.join(input_dir, "unb/DoSHulk_UNB.txt")
            attack_21_dict = load_data_from_txt(input_file=input_file, data_range=(0, 11530),
                                                label=label_dict['attack'])

            input_file = input_file = os.path.join(input_dir, "unb/DOSSlowHttpTest_UNB.txt")
            attack_22_dict = load_data_from_txt(input_file=input_file, data_range=(0, 6414), label=label_dict['attack'])

            input_file = os.path.join(input_dir, "unb/UNB_DosGoldenEye_UNB_IDS2017.txt")
            attack_23_dict = load_data_from_txt(input_file=input_file, data_range=(0, 1268), label=label_dict['attack'])

            input_file = input_file = os.path.join(input_dir, "unb/UNB_DoSSlowloris_UNB_IDS2017.txt")
            attack_24_dict = load_data_from_txt(input_file=input_file, data_range=(0, 16741),
                                                label=label_dict['attack'])

            unb_attack_lst = [attack_21_dict, attack_22_dict, attack_23_dict, attack_24_dict]
            unb_train_set_dict, unb_val_set_dict, unb_test_set_dict = get_each_dataset(norm_dict=unb_norm_dict,
                                                                                       attack_dict_lst=unb_attack_lst,
                                                                                       name='unb',
                                                                                       test_set_percent=test_set_percent,
                                                                                       shuffle_flg=shuffle_flg,
                                                                                       random_state=random_state)

            test_set_dict.update(unb_test_set_dict)

            ###########################################################################################################
            ### add test set 3 (mawi test set )
            ### all norm data: MAWI
            input_file = os.path.join(input_dir, "mawi/Normal_mawi_day1.txt")
            mawi_norm_dict = load_data_from_txt(input_file=input_file, data_range=(0, 62000), label=label_dict['norm'])

            mawi_attack_lst = []
            mawi_train_set_dict, mawi_val_set_dict, mawi_test_set_dict = get_each_dataset(norm_dict=mawi_norm_dict,
                                                                                          attack_dict_lst=mawi_attack_lst,
                                                                                          name='mawi',
                                                                                          test_set_percent=test_set_percent,
                                                                                          shuffle_flg=shuffle_flg,
                                                                                          random_state=random_state)

            test_set_dict.update(mawi_test_set_dict)

            print_dict(train_set_dict)
            print_dict(val_set_dict)
            print_dict(test_set_dict)

            return {'train_set_dict': train_set_dict, 'val_set_dict': val_set_dict, 'test_set_dict': test_set_dict}

            # x_norm = load_data_from_txt(input_file=os.path.join(input_dir, "synthetic_dataset/Sess_normal_0.txt"),
            #                             end=77989)
            # y_norm = np.ones(shape=[x_norm.shape[0], 1])
            # # split normal data
            # x_norm_train_all, x_norm_test, y_norm_train_all, y_norm_test = train_test_split(x_norm, y_norm,
            #                                                                                 test_size=test_size,
            #                                                                                 random_state=random_state)
            # x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all, y_norm_train_all,
            #                                                                       test_size=0.125,
            #                                                                       random_state=random_state)
            #
            # x_attack_11 = load_data_from_txt(
            #     input_file=os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"),
            #     end=36000)
            # x_attack_12 = load_data_from_txt(
            #     input_file=os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Recursive_GET.dms"),
            #     end=37000)
            # x_attack_13 = load_data_from_txt(
            #     input_file=os.path.join(input_dir,
            #                             "synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms"),
            #     end=243)
            # x_attack_14 = load_data_from_txt(
            #     input_file=os.path.join(input_dir,
            #                             "synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"),
            #     end=1000)
            # x_attack_1 = np.concatenate([x_attack_11, x_attack_12, x_attack_13, x_attack_14])
            # y_attack_1 = np.concatenate(
            #     [np.zeros([x_attack_11.shape[0]]), np.zeros([x_attack_12.shape[0]]), np.zeros([x_attack_13.shape[0]]),
            #      np.zeros([x_attack_14.shape[0]])])
            # y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))
            #
            # # x_attack_11 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt",
            # #                                  end=36000)
            # # x_attack_14 = load_data_from_txt(
            # #     input_file="dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms",
            # #     end=1000)
            # # x_attack_1 = np.concatenate([x_attack_11, x_attack_14])
            # # y_attack_1 = np.concatenate(
            # #     [np.zeros([x_attack_11.shape[0]]),np.zeros([x_attack_14.shape[0]])])
            # # y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))
            #
            # # split attackack data
            # x_attack_train_all, x_attack_test, y_attack_train_all, y_attack_test = train_test_split(x_attack_1,
            #                                                                                         y_attack_1,
            #                                                                                         test_size=test_size,
            #                                                                                         random_state=random_state)
            # x_attack_train, x_attack_val, y_attack_train, y_attack_val = train_test_split(x_attack_train_all,
            #                                                                               y_attack_train_all,
            #                                                                               test_size=0.125,
            #                                                                               random_state=random_state)
            #
            # SYNT_train_set = {'x_norm_train': x_norm_train, 'y_norm_train': y_norm_train,
            #                   'x_attack_train': x_attack_train, 'y_attack_train': y_attack_train}
            # SYNT_val_set = {'x_norm_val': x_norm_val, 'y_norm_val': y_norm_val,
            #                 'x_attack_val': x_attack_val, 'y_attack_val': y_attack_val}
            # SYNT_test_set = {'x_norm_test': x_norm_test, 'y_norm_test': y_norm_test,
            #                  'x_attack_test': x_attack_test, 'y_attack_test': y_attack_test}
            #
            # # x_attack_12 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms",
            # #                                   end=37000)
            # # x_attack_13 = load_data_from_txt(
            # #      input_file="dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms",
            # #      end=243)
            # # x_attack_test_2 = np.concatenate([x_attack_12, x_attack_13])
            # # y_attack_test_2 = np.concatenate([np.zeros([x_attack_12.shape[0]]), np.zeros([x_attack_13.shape[0]])])
            # # y_attack_test_2 = np.reshape(y_attack_test_2, (y_attack_test_2.shape[0], 1))
            # # SYNT_test_set_2 = (x_norm_test, y_norm_test, x_attack_test_2, y_attack_test_2)
            #
            # x_attack_UNB_11 = load_data_from_txt(input_file=os.path.join(input_dir, "unb/DoSHulk_UNB.txt"), end=11530)
            # x_attack_UNB_12 = load_data_from_txt(input_file=os.path.join(input_dir, "unb/DOSSlowHttpTest_UNB.txt"),
            #                                      end=6414)
            # x_attack_UNB_13 = load_data_from_txt(
            #     input_file=os.path.join(input_dir, "unb/UNB_DosGoldenEye_UNB_IDS2017.txt"), end=1268)
            # x_attack_UNB_14 = load_data_from_txt(
            #     input_file=os.path.join(input_dir, "unb/UNB_DoSSlowloris_UNB_IDS2017.txt"), end=16741)
            #
            # x_attack_UNB_1 = np.concatenate([x_attack_UNB_11, x_attack_UNB_12, x_attack_UNB_13, x_attack_UNB_14])
            # y_attack_UNB_1 = np.concatenate(
            #     [np.zeros([x_attack_UNB_11.shape[0]]), np.zeros([x_attack_UNB_12.shape[0]]),
            #      np.zeros([x_attack_UNB_13.shape[0]]),
            #      np.zeros([x_attack_UNB_14.shape[0]])])
            # y_attack_UNB_1 = np.reshape(y_attack_UNB_1, (y_attack_UNB_1.shape[0], 1))
            #
            # SYNT_test_set_2 = {'x_norm_test': x_norm_test, 'y_norm_test': y_norm_test,
            #                    'x_attack_test': x_attack_UNB_1, 'y_attack_test': y_attack_UNB_1}
            # test_sets_dict = OrderedDict(
            #     {"SYNT_test": SYNT_test_set, 'SYNT_test_2': SYNT_test_set_2})
            #
            # return SYNT_train_set, SYNT_val_set, test_sets_dict


        elif case[5] == '2':
            # unb  because of unb attack and normal exist huge difference, so DT can easily distingusih them on test set 1 and test set 2.
            ### all norm data: UNB
            input_file = os.path.join(input_dir, "unb/Normal_UNB.txt")
            unb_norm_dict = load_data_from_txt(input_file=input_file, data_range=(0, 59832), label=label_dict['norm'])

            ### all attack data: UNB
            input_file = os.path.join(input_dir, "unb/DoSHulk_UNB.txt")
            attack_21_dict = load_data_from_txt(input_file=input_file, data_range=(0, 11530),
                                                label=label_dict['attack'])

            input_file = input_file = os.path.join(input_dir, "unb/DOSSlowHttpTest_UNB.txt")
            attack_22_dict = load_data_from_txt(input_file=input_file, data_range=(0, 6414), label=label_dict['attack'])

            input_file = os.path.join(input_dir, "unb/UNB_DosGoldenEye_UNB_IDS2017.txt")
            attack_23_dict = load_data_from_txt(input_file=input_file, data_range=(0, 1268), label=label_dict['attack'])

            input_file = input_file = os.path.join(input_dir, "unb/UNB_DoSSlowloris_UNB_IDS2017.txt")
            attack_24_dict = load_data_from_txt(input_file=input_file, data_range=(0, 16741),
                                                label=label_dict['attack'])

            unb_attack_lst = [attack_21_dict, attack_22_dict, attack_23_dict, attack_24_dict]
            unb_train_set_dict, unb_val_set_dict, unb_test_set_dict = get_each_dataset(norm_dict=unb_norm_dict,
                                                                                       attack_dict_lst=unb_attack_lst,
                                                                                       name='unb',
                                                                                       test_set_percent=test_set_percent,
                                                                                       shuffle_flg=shuffle_flg,
                                                                                       random_state=random_state)
            train_set_dict = unb_train_set_dict
            val_set_dict = unb_val_set_dict
            # test_set_dict.update(unb_test_set_dict)
            test_set_dict = unb_test_set_dict

            ###########################################################################################################
            ### test_set 2
            ### all attack data: SNYT
            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_normal_0.txt")
            synt_norm_dict = load_data_from_txt(input_file=input_file, data_range=(0, 77989), label=label_dict['norm'])

            ### all attack data_set 1 : SYNT
            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt")
            attack_11_dict = load_data_from_txt(input_file=input_file, data_range=(0, 36000),
                                                label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Recursive_GET.dms")
            attack_12_dict = load_data_from_txt(input_file=input_file, data_range=(0, 37000),
                                                label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms")
            attack_13_dict = load_data_from_txt(input_file=input_file, data_range=(0, 243), label=label_dict['attack'])

            input_file = os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms")
            attack_14_dict = load_data_from_txt(input_file=input_file, data_range=(0, 1000), label=label_dict['attack'])

            ### test set: SYNT
            synt_attack_lst = [attack_11_dict, attack_12_dict, attack_13_dict, attack_14_dict]
            synt_train_set_dict, synt_val_set_dict, synt_test_set_dict = get_each_dataset(norm_dict=synt_norm_dict,
                                                                                          attack_dict_lst=synt_attack_lst,
                                                                                          name='synt',
                                                                                          test_set_percent=test_set_percent,
                                                                                          shuffle_flg=shuffle_flg,
                                                                                          random_state=random_state)

            test_set_dict.update(synt_test_set_dict)

            print_dict(train_set_dict)
            print_dict(val_set_dict)
            print_dict(test_set_dict)

            return {'train_set_dict': train_set_dict, 'val_set_dict': val_set_dict, 'test_set_dict': test_set_dict}

            # # x_norm = dataset(filename=["dataset/unb/Normal_UNB.txt"], label=1, end=59832)
            # x_norm = load_data_from_txt(input_file=os.path.join(input_dir, "dataset/unb/Normal_UNB.txt", end=59832))
            # y_norm = np.ones(shape=[x_norm.shape[0], 1])
            # # split normal data
            # x_norm_train_all, x_norm_test, y_norm_train_all, y_norm_test = train_test_split(x_norm, y_norm,
            #                                                                                 test_size=test_size,
            #                                                                                 random_state=random_state)
            # x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all, y_norm_train_all,
            #                                                                       test_size=0.125,
            #                                                                       random_state=random_state)
            #
            # # x_attack_11 = dataset(filename=["dataset/unb/DoSHulk_UNB.txt"], label=0, end=11530,
            # #                       mean=x_norm.mean, std=x_norm.std).data
            # # x_attack_12 = dataset(filename=["dataset/unb/DOSSlowHttpTest_UNB.txt"], label=0, end=6414,
            # #                       mean=x_norm.mean, std=x_norm.std).data
            # # x_attack_1 = np.concatenate([x_attack_11, x_attack_12])
            # # y_attack_1 = np.concatenate(
            # #     [np.zeros([x_attack_11.shape[0]]), np.zeros([x_attack_12.shape[0]])])
            # # y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))
            #
            # x_attack_11 = load_data_from_txt(
            #     input_file=os.path.join(input_dir, "dataset/unb/DoSHulk_UNB.txt", end=11530))
            # x_attack_12 = load_data_from_txt(
            #     input_file=os.path.join(input_dir, "dataset/unb/DOSSlowHttpTest_UNB.txt", end=6414))
            # x_attack_13 = load_data_from_txt(
            #     input_file=os.path.join(input_dir, "dataset/unb/UNB_DosGoldenEye_UNB_IDS2017.txt", end=1268))
            # x_attack_14 = load_data_from_txt(
            #     input_file=os.path.join(input_dir, "dataset/unb/UNB_DoSSlowloris_UNB_IDS2017.txt", end=16741))
            #
            # x_attack_1 = np.concatenate([x_attack_11, x_attack_12, x_attack_13, x_attack_14])
            # y_attack_1 = np.concatenate(
            #     [np.zeros([x_attack_11.shape[0]]), np.zeros([x_attack_12.shape[0]]), np.zeros([x_attack_13.shape[0]]),
            #      np.zeros([x_attack_14.shape[0]])])
            # y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))
            #
            # # split attackack data
            # x_attack_train_all, x_attack_test, y_attack_train_all, y_attack_test = train_test_split(x_attack_1,
            #                                                                                         y_attack_1,
            #                                                                                         test_size=test_size,
            #                                                                                         random_state=random_state)
            # x_attack_train, x_attack_val, y_attack_train, y_attack_val = train_test_split(x_attack_train_all,
            #                                                                               y_attack_train_all,
            #                                                                               test_size=0.125,
            #                                                                               random_state=random_state)
            #
            # UNB_train_set = {'x_norm_train': x_norm_train, 'y_norm_train': y_norm_train,
            #                  'x_attack_train': x_attack_train, 'y_attack_train': y_attack_train}
            # UNB_val_set = {'x_norm_val': x_norm_val, 'y_norm_val': y_norm_val,
            #                'x_attack_val': x_attack_val, 'y_attack_val': y_attack_val}
            # UNB_test_set = {'x_norm_test': x_norm_test, 'y_norm_test': y_norm_test,
            #                 'x_attack_test': x_attack_test, 'y_attack_test': y_attack_test}
            #
            # x_attack_SYNT_11 = load_data_from_txt(
            #     input_file="dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt",
            #     end=36000)
            # x_attack_SYNT_12 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms",
            #                                       end=37000)
            # x_attack_SYNT_13 = load_data_from_txt(
            #     input_file="dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms",
            #     end=243)
            # x_attack_SYNT_14 = load_data_from_txt(
            #     input_file="dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms",
            #     end=1000)
            # x_attack_test_SYNT = np.concatenate(
            #     [x_attack_SYNT_11, x_attack_SYNT_12, x_attack_SYNT_13, x_attack_SYNT_14])
            # y_attack_test_SYNT = np.concatenate(
            #     [np.zeros([x_attack_SYNT_11.shape[0]]), np.zeros([x_attack_SYNT_12.shape[0]]),
            #      np.zeros([x_attack_SYNT_13.shape[0]]),
            #      np.zeros([x_attack_SYNT_14.shape[0]])])
            # y_attack_test_SYNT = np.reshape(y_attack_test_SYNT, (y_attack_test_SYNT.shape[0], 1))
            #
            # # x_attack_SYNT_12 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms",
            # #                                       end=37000)
            # # x_attack_SYNT_13 = load_data_from_txt(
            # #     input_file="dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms",
            # #     end=243)
            # # x_attack_test_2 = np.concatenate([x_attack_13, x_attack_14])
            # # y_attack_test_2 = np.concatenate([np.zeros([x_attack_13.shape[0]]), np.zeros([x_attack_14.shape[0]])])
            # # y_attack_test_2 = np.reshape(y_attack_test_2, (y_attack_test_2.shape[0], 1))
            #
            # UNB_test_set_2 = {'x_norm_test': x_norm_test, 'y_norm_test': y_norm_test,
            #                   'x_attack_test': x_attack_test_SYNT, 'y_attack_test': y_attack_test_SYNT}
            # test_sets_dict = OrderedDict({"UNB_test": UNB_test_set, 'UNB_test_2': UNB_test_set_2})
            #
            # return UNB_train_set, UNB_val_set, test_sets_dict

        else:  # if case[5] == '3': # for SYNT test
            # TODO
            pass


def load_data(case, random_state=42, norm_flg=True, test_size=0.2):
    '''
    case[0] = u/s for Unsupervised or Supervised
    case[3] = Scenario
    case[5] = Source
    '''

    '''
    This load data function is a  bit complicated.

    However, when you call dataset(), if you pass the filename only, it will calculate new min and max.
    dataset() returns an object.
    hence, dataset.data contains 27 features for many datapoints

    dataset.mean is the min
    dataset.std is the max
    This can be  changed in the utils folder for standard scaling if you need me to do it.

    start = index of first data_point
    end = index of last data_point
    example : start = 10, end = 100 returns data points from [10,100]


    Attack: 0, Normal: 1
    '''

    input_dir = "/Users/kunyang/PycharmProjects/anomaly_detection_20190611/input_data/dataset"
    if case[3] == '1':  # Experiment 1
        if case[5] == '1':  # training and testing on SYNT (simulated data)
            ### shuffle data
            ### training on SYNT
            # x_norm = dataset(filename=["dataset/synthetic_dataset/Sess_normal_0.txt"], label=1, end=77989)
            x_norm = load_data_from_txt(input_file=os.path.join(input_dir, "synthetic_dataset/Sess_normal_0.txt"),
                                        end=77989)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])
            # split normal data
            x_norm_train_all, x_norm_test, y_norm_train_all, y_norm_test = train_test_split(x_norm, y_norm,
                                                                                            test_size=test_size,
                                                                                            random_state=random_state)
            x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all, y_norm_train_all,
                                                                                  test_size=0.125,
                                                                                  random_state=random_state)
            # x_attack_11 = dataset(filename=["dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"], label=0,
            #                       end=36000, mean=x_norm.mean, std=x_norm.std).data
            # x_attack_12 = dataset(filename=["dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms"], label=0,
            #                       end=37000, mean=x_norm.mean, std=x_norm.std).data
            # x_attack_13 = dataset(
            #     filename=["dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms"], label=0,
            #     end=243, mean=x_norm.mean, std=x_norm.std).data
            # x_attack_14 = dataset(
            #     filename=["dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"], label=0,
            #     end=1000, mean=x_norm.mean, std=x_norm.std).data
            x_attack_11 = load_data_from_txt(
                input_file=os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"),
                end=36000)
            x_attack_12 = load_data_from_txt(
                input_file=os.path.join(input_dir, "synthetic_dataset/Sess_DDoS_Recursive_GET.dms"),
                end=37000)
            x_attack_13 = load_data_from_txt(
                input_file=os.path.join(input_dir,
                                        "synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms"),
                end=243)
            x_attack_14 = load_data_from_txt(
                input_file=os.path.join(input_dir,
                                        "synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"),
                end=1000)
            x_attack_1 = np.concatenate([x_attack_11, x_attack_12, x_attack_13, x_attack_14])
            y_attack_1 = np.concatenate(
                [np.zeros([x_attack_11.shape[0]]), np.zeros([x_attack_12.shape[0]]), np.zeros([x_attack_13.shape[0]]),
                 np.zeros([x_attack_14.shape[0]])])
            y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))

            # if norm_flg:
            #     x_norm_train, x_norm_train_mu, x_norm_train_std = z_score_np(x_norm_train, mu='', d_std='')
            #     x_norm_val = z_score_np(x_norm_val, mu=x_norm_train_mu, d_std=x_norm_train_std)
            #     x_norm_test = z_score_np(x_norm_test, mu=x_norm_train_mu, d_std=x_norm_train_std)
            #     x_attack_1 = z_score_np(x_attack_1, mu=x_norm_train_mu, d_std=x_norm_train_std)
            #
        elif case[5] == '2':  # training and testing on unb
            ### with shuffle
            ### training on unb
            # x_norm = dataset(filename=["dataset/unb/Normal_UNB.txt"], label=1, end=59832)
            x_norm = load_data_from_txt(input_file=os.path.join(input_dir, "unb/Normal_UNB.txt"), end=59832)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])
            # split normal data
            x_norm_train_all, x_norm_test, y_norm_train_all, y_norm_test = train_test_split(x_norm, y_norm,
                                                                                            test_size=test_size,
                                                                                            random_state=random_state)
            x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all, y_norm_train_all,
                                                                                  test_size=0.125,
                                                                                  random_state=random_state)
            # x_attack_11 = dataset(filename=["dataset/unb/DoSHulk_UNB.txt"], label=0, end=11530,
            #                       mean=x_norm.mean, std=x_norm.std).data
            # x_attack_12 = dataset(filename=["dataset/unb/DOSSlowHttpTest_UNB.txt"], label=0, end=6414,
            #                       mean=x_norm.mean, std=x_norm.std).data
            # x_attack_13 = dataset(filename=["dataset/unb/UNB_DosGoldenEye_UNB_IDS2017.txt"], label=0, end=1268,
            #                       mean=x_norm.mean, std=x_norm.std).data
            # x_attack_14 = dataset(filename=["dataset/unb/UNB_DoSSlowloris_UNB_IDS2017.txt"], label=0, end=16741,
            #                       mean=x_norm.mean, std=x_norm.std).data

            x_attack_11 = load_data_from_txt(input_file=os.path.join(input_dir, "unb/DoSHulk_UNB.txt"), end=11530)
            x_attack_12 = load_data_from_txt(input_file=os.path.join(input_dir, "unb/DOSSlowHttpTest_UNB.txt"),
                                             end=6414)
            x_attack_13 = load_data_from_txt(input_file=os.path.join(input_dir, "unb/UNB_DosGoldenEye_UNB_IDS2017.txt"),
                                             end=1268)
            x_attack_14 = load_data_from_txt(input_file=os.path.join(input_dir, "unb/UNB_DoSSlowloris_UNB_IDS2017.txt"),
                                             end=16741)
            x_attack_1 = np.concatenate([x_attack_11, x_attack_12, x_attack_13, x_attack_14])
            y_attack_1 = np.concatenate(
                [np.zeros([x_attack_11.shape[0]]), np.zeros([x_attack_12.shape[0]]), np.zeros([x_attack_13.shape[0]]),
                 np.zeros([x_attack_14.shape[0]])])
            y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))

            # if norm_flg:
            #     x_norm_train, x_norm_train_mu, x_norm_train_std = z_score_np(x_norm_train, mu='', d_std='')
            #     x_norm_val = z_score_np(x_norm_val, mu=x_norm_train_mu, d_std=x_norm_train_std)
            #     x_norm_test = z_score_np(x_norm_test, mu=x_norm_train_mu, d_std=x_norm_train_std)
            #     x_attack_1 = z_score_np(x_attack_1, mu=x_norm_train_mu, d_std=x_norm_train_std)

        elif case[5] == '3':  # training and testing on mawi
            # x_norm = dataset(filename=["dataset/mawi/Normal_mawi_day1.txt"], label=1, end=62000)
            x_norm = load_data_from_txt(input_file="dataset/mawi/Normal_mawi_day1.txt", end=62000)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])
            # split normal data
            x_norm_train_all, x_norm_test, y_norm_train_all, y_norm_test = train_test_split(x_norm, y_norm,
                                                                                            test_size=test_size,
                                                                                            random_state=random_state)
            x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all, y_norm_train_all,
                                                                                  test_size=0.125,
                                                                                  random_state=random_state)
            x_attack_1 = None
            y_attack_1 = None

        return (x_norm_train, y_norm_train), (x_norm_val, y_norm_val), (x_norm_test, y_norm_test), (
            x_attack_1, y_attack_1)


    elif case[3] == '2':  # Experiment 2
        if case[5] == '1':
            pass
            # data_train = dataset(filename=["dataset/synthetic_dataset/Sess_normal_0.txt"], label=1, end=7000)
            # data_val = dataset(filename=["dataset/synthetic_dataset/Sess_normal_0.txt"], label=1, start=7000, end=8000,
            #                    mean=data_train.mean, std=data_train.std).data
            #
            # data_test_1_norm = dataset(filename=["dataset/synthetic_dataset/Sess_normal_0.txt"], label=1, start=8000,
            #                            end=10000, mean=data_train.mean, std=data_train.std).data
            # data_test_1_attack = dataset(
            #     filename=["dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms",
            #               "dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"], label=0, end=1000,
            #     mean=data_train.mean, std=data_train.std).data
            #
            # data_test_2_norm = dataset(filename=["dataset/synthetic_dataset/Sess_normal_0.txt"], label=1, start=10000,
            #                            end=12000, mean=data_train.mean, std=data_train.std).data
            # data_test_2_attack = dataset(filename=["dataset/unb/DoSHulk_UNB.txt", "dataset/unb/DOSSlowHttpTest_UNB.txt"],
            #                           label=0, end=1000, mean=data_train.mean, std=data_train.std).data


        elif case[5] == '2':  # training and testing on unb
            ### without shuffle.
            # data_train = dataset(filename=["dataset/unb/Normal_UNB.txt"], label=1, end=7000)
            # data_val = dataset(filename=["dataset/unb/Normal_UNB.txt"], label=1, start=7000, end=8000,
            #                    mean=data_train.mean, std=data_train.std).data
            #
            # data_test_1_norm = dataset(filename=["dataset/unb/Normal_UNB.txt"], label=1, start=8000, end=10000,
            #                            mean=data_train.mean, std=data_train.std).data
            # data_test_1_attack = dataset(filename=["dataset/unb/DoSHulk_UNB.txt", "dataset/unb/DOSSlowHttpTest_UNB.txt"],
            #                           label=0, end=1000, mean=data_train.mean, std=data_train.std).data
            #
            # data_test_2_norm = dataset(filename=["dataset/unb/Normal_UNB.txt"], label=1, start=10000, end=12000,
            #                            mean=data_train.mean, std=data_train.std).data
            # data_test_2_attack = dataset(
            #     filename=["dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms",
            #               "dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"], label=0, end=1000,
            #     mean=data_train.mean, std=data_train.std).data
            #
            # # data_train = dataset(filename = ["dataset/unb/DoSHulk_UNB.txt"], label = 1, end = 7000).data

            ### with shuffle
            ### training on unb
            # x_norm = dataset(filename=["dataset/unb/Normal_UNB.txt"], label=1, end=59832)
            # y_norm = np.ones(shape=[x_norm.data.shape[0], 1])
            x_norm = load_data_from_txt(input_file="dataset/unb/Normal_UNB.txt", end=59832)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])
            # split normal data
            x_norm_train_all, x_norm_test, y_norm_train_all, y_norm_test = train_test_split(x_norm, y_norm,
                                                                                            test_size=test_size,
                                                                                            random_state=random_state)
            x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all, y_norm_train_all,
                                                                                  test_size=0.125,
                                                                                  random_state=random_state)

            # # x_attack_11 = dataset(filename=["dataset/unb/DoSHulk_UNB.txt"], label=0, end=11530,
            # #                       mean=x_norm.mean, std=x_norm.std).data
            # # x_attack_12 = dataset(filename=["dataset/unb/DOSSlowHttpTest_UNB.txt"], label=0, end=6414,
            # #                       mean=x_norm.mean, std=x_norm.std).data
            # x_attack_11 = load_data_from_txt(input_file="dataset/unb/DoSHulk_UNB.txt", end=11530)
            # x_attack_12 = load_data_from_txt(input_file="dataset/unb/DOSSlowHttpTest_UNB.txt", end=6414)
            # x_attack_1 = np.concatenate([x_attack_11, x_attack_12])
            # y_attack_1 = np.concatenate([np.zeros([x_attack_11.shape[0]]), np.zeros([x_attack_12.shape[0]])])
            # y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))

            x_attack_11 = load_data_from_txt(input_file="dataset/unb/DoSHulk_UNB.txt", end=11530)
            x_attack_12 = load_data_from_txt(input_file="dataset/unb/DOSSlowHttpTest_UNB.txt", end=6414)
            x_attack_13 = load_data_from_txt(input_file="dataset/unb/UNB_DosGoldenEye_UNB_IDS2017.txt", end=1268)
            x_attack_14 = load_data_from_txt(input_file="dataset/unb/UNB_DoSSlowloris_UNB_IDS2017.txt", end=16741)
            x_attack_1 = np.concatenate([x_attack_11, x_attack_12, x_attack_13, x_attack_14])
            y_attack_1 = np.concatenate(
                [np.zeros([x_attack_11.shape[0]]), np.zeros([x_attack_12.shape[0]]), np.zeros([x_attack_13.shape[0]]),
                 np.zeros([x_attack_14.shape[0]])])
            y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))

            # split attackack data
            x_attack_train_all, x_attack_test, y_attack_train_all, y_attack_test = train_test_split(x_attack_1,
                                                                                                    y_attack_1,
                                                                                                    test_size=test_size,
                                                                                                    random_state=random_state)
            x_attack_train, x_attack_val, y_attack_train, y_attack_val = train_test_split(x_attack_train_all,
                                                                                          y_attack_train_all,
                                                                                          test_size=0.125,
                                                                                          random_state=random_state)

            ### testing on unb
            UNB_train_set = (x_norm_train, y_norm_train, x_attack_train, y_attack_train)
            UNB_val_set = (x_norm_val, y_norm_val, x_attack_val, y_attack_val)
            UNB_test_set = (x_norm_test, y_norm_test, x_attack_test, y_attack_test)

            ### testing on SYNT
            # x_norm_test_SYNT = dataset(filename=["dataset/synthetic_dataset/Sess_normal_0.txt"], label=1, end=77989,
            #                            mean=x_norm.mean, std=x_norm.std)
            x_norm_SYNT = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_normal_0.txt", end=77989)
            y_norm_SYNT = np.ones(shape=[x_norm_SYNT.shape[0], 1])

            x_norm_train_SYNT, x_norm_test_SYNT, y_norm_train_SYNT, y_norm_test_SYNT = train_test_split(x_norm_SYNT,
                                                                                                        y_norm_SYNT,
                                                                                                        test_size=test_size,
                                                                                                        random_state=random_state)
            # x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all, y_norm_train_all,
            #                                                                       test_size=0.125,
            #                                                                       random_state=random_state)

            # # x_attack_test_SYNT = dataset(
            # #     filename=["dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms",
            # #               "dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"], label=0, end=1000,
            # #     mean=x_norm_test_SYNT.mean, std=x_norm_test_SYNT.std).data
            #
            # # y_attack_test_SYNT = np.ones(shape=[x_attack_test_SYNT.data.shape[0], 1])
            # # x_attack_SYNT_11 = dataset(filename=["dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"], label=0,
            # #                            end=36000, mean=x_norm.mean, std=x_norm.std).data
            # # x_attack_SYNT_14 = dataset(
            # #     filename=["dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"], label=0,
            # #     end=1000, mean=x_norm.mean, std=x_norm.std).data
            # x_attack_SYNT_11 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt", end=36000)
            # x_attack_SYNT_14 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms", end=1000)
            # x_attack_test_SYNT = np.concatenate([x_attack_SYNT_11, x_attack_SYNT_14])
            # y_attack_test_SYNT = np.concatenate(
            #     [np.zeros([x_attack_SYNT_11.shape[0]]), np.zeros([x_attack_SYNT_14.shape[0]])])
            # y_attack_test_SYNT = np.reshape(y_attack_test_SYNT, (y_attack_test_SYNT.shape[0], 1))

            x_attack_SYNT_11 = load_data_from_txt(
                input_file="dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt",
                end=36000)
            x_attack_SYNT_12 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms",
                                                  end=37000)
            x_attack_SYNT_13 = load_data_from_txt(
                input_file="dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms",
                end=243)
            x_attack_SYNT_14 = load_data_from_txt(
                input_file="dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms",
                end=1000)
            x_attack_test_SYNT = np.concatenate(
                [x_attack_SYNT_11, x_attack_SYNT_12, x_attack_SYNT_13, x_attack_SYNT_14])
            y_attack_test_SYNT = np.concatenate(
                [np.zeros([x_attack_SYNT_11.shape[0]]), np.zeros([x_attack_SYNT_12.shape[0]]),
                 np.zeros([x_attack_SYNT_13.shape[0]]),
                 np.zeros([x_attack_SYNT_14.shape[0]])])
            y_attack_test_SYNT = np.reshape(y_attack_test_SYNT, (y_attack_test_SYNT.shape[0], 1))

            SYNT_test_set = (x_norm_test_SYNT, y_norm_test_SYNT, x_attack_test_SYNT, y_attack_test_SYNT)

            ### testing on mawi
            # x_norm_test_MAWI = dataset(filename=["dataset/mawi/Normal_mawi_day1.txt"], label=1, end=62000,
            #                            mean=x_norm.mean, std=x_norm.std)
            x_norm_MAWI = load_data_from_txt(input_file="dataset/mawi/Normal_mawi_day1.txt", end=62000)
            y_norm_MAWI = np.ones(shape=[x_norm_MAWI.shape[0], 1])

            x_norm_train_MAWI, x_norm_test_MAWI, y_norm_train_MAWI, y_norm_test_MAWI = train_test_split(x_norm_MAWI,
                                                                                                        y_norm_MAWI,
                                                                                                        test_size=test_size,
                                                                                                        random_state=random_state)
            MAWI_test_set = (x_norm_test_MAWI, y_norm_test_MAWI, None, None)

            return UNB_train_set, UNB_val_set, UNB_test_set, SYNT_test_set, MAWI_test_set


        elif case[5] == '3':
            pass
        #     data_train = dataset(filename=["dataset/mawi/Normal_mawi_day1.txt"], label=1, end=7000)
        #     data_val = dataset(filename=["dataset/mawi/Normal_mawi_day1.txt"], label=1, start=7000, end=8000,
        #                        mean=data_train.mean, std=data_train.std).data
        #
        #     data_test_1_norm = dataset(filename=["dataset/mawi/Normal_mawi_day2.txt"], label=1, end=2000,
        #                                mean=data_train.mean, std=data_train.std).data
        #     data_test_1_attack = dataset(filename=["dataset/unb/DoSHulk_UNB.txt", "dataset/unb/DOSSlowHttpTest_UNB.txt"],
        #                               label=0, end=1000, mean=data_train.mean, std=data_train.std).data
        #
        #     data_test_2_norm = dataset(filename=["dataset/mawi/Normal_mawi_day2.txt"], label=1, start=2000, end=4000,
        #                                mean=data_train.mean, std=data_train.std).data
        #     data_test_2_attack = dataset(
        #         filename=["dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms",
        #                   "dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"], label=0, end=1000,
        #         mean=data_train.mean, std=data_train.std).data
        # data_test_1 = np.concatenate([data_test_1_norm, data_test_1_attack])
        # data_test_1_labels = np.concatenate([np.ones([2000]), np.zeros([2000])])
        #
        # data_test_2 = np.concatenate([data_test_2_norm, data_test_2_attack])
        # data_test_2_labels = np.concatenate([np.ones([2000]), np.zeros([2000])])

    elif case[3] == '3':  # Experiment 3, compare AE and DT.
        if case[5] == '1':  # SYNT
            # pass
            # data_train = dataset(filename=["dataset/synthetic_dataset/dt_train00.txt"], label=1)
            # data_train_labels = np.concatenate([np.ones([6000]), np.zeros([5902])])
            #
            # data_test_1 = dataset(filename=["dataset/synthetic_dataset/Sess_normal_0.txt"], label=1, start=49600, end=62000,
            #                       mean=data_train.mean, std=data_train.std).data
            # data_test_2 = dataset(filename=["dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms"], label=1, start=26600,
            #                       end=34200, mean=data_train.mean, std=data_train.std).data
            # data_test_3 = dataset(filename=["dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"],
            #                       label=0, start=800, end=1000, mean=data_train.mean, std=data_train.std).data
            # data_test_4 = dataset(filename = ["dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms"], label = 0, end = 243, mean = data_train.mean, std = data_train.std).data
            # data_test_5 = dataset(filename = ["dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"], label = 0, end = 1000, mean = data_train.mean, std = data_train.std).data

            x_norm = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_normal_0.txt", end=77989)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])
            # split normal data
            x_norm_train_all, x_norm_test, y_norm_train_all, y_norm_test = train_test_split(x_norm, y_norm,
                                                                                            test_size=test_size,
                                                                                            random_state=random_state)
            x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all, y_norm_train_all,
                                                                                  test_size=0.125,
                                                                                  random_state=random_state)

            x_attack_11 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt",
                                             end=36000)
            x_attack_12 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms",
                                             end=37000)
            x_attack_13 = load_data_from_txt(
                input_file="dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms",
                end=243)
            x_attack_14 = load_data_from_txt(
                input_file="dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms",
                end=1000)
            x_attack_1 = np.concatenate([x_attack_11, x_attack_12, x_attack_13, x_attack_14])
            y_attack_1 = np.concatenate(
                [np.zeros([x_attack_11.shape[0]]), np.zeros([x_attack_12.shape[0]]), np.zeros([x_attack_13.shape[0]]),
                 np.zeros([x_attack_14.shape[0]])])
            y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))

            # x_attack_11 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt",
            #                                  end=36000)
            # x_attack_14 = load_data_from_txt(
            #     input_file="dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms",
            #     end=1000)
            # x_attack_1 = np.concatenate([x_attack_11, x_attack_14])
            # y_attack_1 = np.concatenate(
            #     [np.zeros([x_attack_11.shape[0]]),np.zeros([x_attack_14.shape[0]])])
            # y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))

            # split attackack data
            x_attack_train_all, x_attack_test, y_attack_train_all, y_attack_test = train_test_split(x_attack_1,
                                                                                                    y_attack_1,
                                                                                                    test_size=test_size,
                                                                                                    random_state=random_state)
            x_attack_train, x_attack_val, y_attack_train, y_attack_val = train_test_split(x_attack_train_all,
                                                                                          y_attack_train_all,
                                                                                          test_size=0.125,
                                                                                          random_state=random_state)

            SYNT_train_set = (x_norm_train, y_norm_train, x_attack_train, y_attack_train)
            SYNT_val_set = (x_norm_val, y_norm_val, x_attack_val, y_attack_val)
            SYNT_test_set = (x_norm_test, y_norm_test, x_attack_test, y_attack_test)

            # x_attack_12 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms",
            #                                   end=37000)
            # x_attack_13 = load_data_from_txt(
            #      input_file="dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms",
            #      end=243)
            # x_attack_test_2 = np.concatenate([x_attack_12, x_attack_13])
            # y_attack_test_2 = np.concatenate([np.zeros([x_attack_12.shape[0]]), np.zeros([x_attack_13.shape[0]])])
            # y_attack_test_2 = np.reshape(y_attack_test_2, (y_attack_test_2.shape[0], 1))
            # SYNT_test_set_2 = (x_norm_test, y_norm_test, x_attack_test_2, y_attack_test_2)

            x_attack_UNB_11 = load_data_from_txt(input_file="dataset/unb/DoSHulk_UNB.txt", end=11530)
            x_attack_UNB_12 = load_data_from_txt(input_file="dataset/unb/DOSSlowHttpTest_UNB.txt", end=6414)
            x_attack_UNB_13 = load_data_from_txt(input_file="dataset/unb/UNB_DosGoldenEye_UNB_IDS2017.txt", end=1268)
            x_attack_UNB_14 = load_data_from_txt(input_file="dataset/unb/UNB_DoSSlowloris_UNB_IDS2017.txt", end=16741)

            x_attack_UNB_1 = np.concatenate([x_attack_UNB_11, x_attack_UNB_12, x_attack_UNB_13, x_attack_UNB_14])
            y_attack_UNB_1 = np.concatenate(
                [np.zeros([x_attack_UNB_11.shape[0]]), np.zeros([x_attack_UNB_12.shape[0]]),
                 np.zeros([x_attack_UNB_13.shape[0]]),
                 np.zeros([x_attack_UNB_14.shape[0]])])
            y_attack_UNB_1 = np.reshape(y_attack_UNB_1, (y_attack_UNB_1.shape[0], 1))

            SYNT_test_set_2 = (x_norm_test, y_norm_test, x_attack_UNB_1, y_attack_UNB_1)

            return SYNT_train_set, SYNT_val_set, SYNT_test_set, SYNT_test_set_2


        elif case[5] == '2':
            # unb  because of unb attack and normal exist huge difference, so DT can easily distingusih them on test set 1 and test set 2.
            # x_norm = dataset(filename=["dataset/unb/Normal_UNB.txt"], label=1, end=59832)
            x_norm = load_data_from_txt(input_file="dataset/unb/Normal_UNB.txt", end=59832)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])
            # split normal data
            x_norm_train_all, x_norm_test, y_norm_train_all, y_norm_test = train_test_split(x_norm, y_norm,
                                                                                            test_size=test_size,
                                                                                            random_state=random_state)
            x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all, y_norm_train_all,
                                                                                  test_size=0.125,
                                                                                  random_state=random_state)

            # x_attack_11 = dataset(filename=["dataset/unb/DoSHulk_UNB.txt"], label=0, end=11530,
            #                       mean=x_norm.mean, std=x_norm.std).data
            # x_attack_12 = dataset(filename=["dataset/unb/DOSSlowHttpTest_UNB.txt"], label=0, end=6414,
            #                       mean=x_norm.mean, std=x_norm.std).data
            # x_attack_1 = np.concatenate([x_attack_11, x_attack_12])
            # y_attack_1 = np.concatenate(
            #     [np.zeros([x_attack_11.shape[0]]), np.zeros([x_attack_12.shape[0]])])
            # y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))

            x_attack_11 = load_data_from_txt(input_file="dataset/unb/DoSHulk_UNB.txt", end=11530)
            x_attack_12 = load_data_from_txt(input_file="dataset/unb/DOSSlowHttpTest_UNB.txt", end=6414)
            x_attack_13 = load_data_from_txt(input_file="dataset/unb/UNB_DosGoldenEye_UNB_IDS2017.txt", end=1268)
            x_attack_14 = load_data_from_txt(input_file="dataset/unb/UNB_DoSSlowloris_UNB_IDS2017.txt", end=16741)

            x_attack_1 = np.concatenate([x_attack_11, x_attack_12, x_attack_13, x_attack_14])
            y_attack_1 = np.concatenate(
                [np.zeros([x_attack_11.shape[0]]), np.zeros([x_attack_12.shape[0]]), np.zeros([x_attack_13.shape[0]]),
                 np.zeros([x_attack_14.shape[0]])])
            y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))

            # split attackack data
            x_attack_train_all, x_attack_test, y_attack_train_all, y_attack_test = train_test_split(x_attack_1,
                                                                                                    y_attack_1,
                                                                                                    test_size=test_size,
                                                                                                    random_state=random_state)
            x_attack_train, x_attack_val, y_attack_train, y_attack_val = train_test_split(x_attack_train_all,
                                                                                          y_attack_train_all,
                                                                                          test_size=0.125,
                                                                                          random_state=random_state)

            UNB_train_set = (x_norm_train, y_norm_train, x_attack_train, y_attack_train)
            UNB_val_set = (x_norm_val, y_norm_val, x_attack_val, y_attack_val)
            UNB_test_set = (x_norm_test, y_norm_test, x_attack_test, y_attack_test)

            x_attack_SYNT_11 = load_data_from_txt(
                input_file="dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt",
                end=36000)
            x_attack_SYNT_12 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms",
                                                  end=37000)
            x_attack_SYNT_13 = load_data_from_txt(
                input_file="dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms",
                end=243)
            x_attack_SYNT_14 = load_data_from_txt(
                input_file="dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms",
                end=1000)
            x_attack_test_SYNT = np.concatenate(
                [x_attack_SYNT_11, x_attack_SYNT_12, x_attack_SYNT_13, x_attack_SYNT_14])
            y_attack_test_SYNT = np.concatenate(
                [np.zeros([x_attack_SYNT_11.shape[0]]), np.zeros([x_attack_SYNT_12.shape[0]]),
                 np.zeros([x_attack_SYNT_13.shape[0]]),
                 np.zeros([x_attack_SYNT_14.shape[0]])])
            y_attack_test_SYNT = np.reshape(y_attack_test_SYNT, (y_attack_test_SYNT.shape[0], 1))

            # x_attack_SYNT_12 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms",
            #                                       end=37000)
            # x_attack_SYNT_13 = load_data_from_txt(
            #     input_file="dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms",
            #     end=243)
            # x_attack_test_2 = np.concatenate([x_attack_13, x_attack_14])
            # y_attack_test_2 = np.concatenate([np.zeros([x_attack_13.shape[0]]), np.zeros([x_attack_14.shape[0]])])
            # y_attack_test_2 = np.reshape(y_attack_test_2, (y_attack_test_2.shape[0], 1))

            UNB_test_set_2 = (x_norm_test, y_norm_test, x_attack_test_SYNT, y_attack_test_SYNT)

            return UNB_train_set, UNB_val_set, UNB_test_set, UNB_test_set_2

        else:  # if case[5] == '3': # for SYNT test

            #### for SYNT Test
            # x_norm = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_normal_0.txt", end=77989)
            # y_norm = np.ones(shape=[x_norm.shape[0], 1])
            # # split normal data
            # x_norm_train_all, x_norm_test, y_norm_train_all, y_norm_test = train_test_split(x_norm, y_norm,
            #                                                                                 test_size=test_size,
            #                                                                                 random_state=random_state)
            # x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all, y_norm_train_all,
            #                                                                       test_size=0.125,
            #                                                                       random_state=random_state)
            #
            # # x_attack_11 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt",
            # #                                  end=36000)
            # x_attack_12 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms",
            #                                  end=37000)
            # # x_attack_13 = load_data_from_txt(
            # #     input_file="dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms",
            # #     end=243)
            # x_attack_14 = load_data_from_txt(
            #     input_file="dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms",
            #     end=1000)
            # x_attack_1 = np.concatenate([x_attack_12, x_attack_14])
            # y_attack_1 = np.concatenate([np.zeros([x_attack_12.shape[0]]), np.zeros([x_attack_14.shape[0]])])
            # y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))
            #
            # # split attackack data
            # x_attack_train_all, x_attack_test, y_attack_train_all, y_attack_test = train_test_split(x_attack_1,
            #                                                                                         y_attack_1,
            #                                                                                         test_size=test_size,
            #                                                                                         random_state=random_state)
            # x_attack_train, x_attack_val, y_attack_train, y_attack_val = train_test_split(x_attack_train_all,
            #                                                                               y_attack_train_all,
            #                                                                               test_size=0.125,
            #                                                                               random_state=random_state)
            #
            # SYNT_train_set = (x_norm_train, y_norm_train, x_attack_train, y_attack_train)
            # SYNT_val_set = (x_norm_val, y_norm_val, x_attack_val, y_attack_val)
            # SYNT_test_set = (x_norm_test, y_norm_test, x_attack_test, y_attack_test)
            #
            # x_attack_11 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt",
            #                                  end=36000)
            # # x_attack_12 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms",
            # #                                  end=37000)
            # x_attack_13 = load_data_from_txt(
            #     input_file="dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms",
            #     end=243)
            # # x_attack_14 = load_data_from_txt(
            # #     input_file="dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms",
            # #     end=1000)
            # x_attack_SYNT_1 = np.concatenate([x_attack_11, x_attack_13])
            # y_attack_1 = np.concatenate([np.zeros([x_attack_11.shape[0]]), np.zeros([x_attack_13.shape[0]])])
            # y_attack_SYNT_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))
            #
            # SYNT_test_set_2 = (x_norm_test, y_norm_test, x_attack_SYNT_1, y_attack_SYNT_1)
            #
            # return SYNT_train_set, SYNT_val_set, SYNT_test_set, SYNT_test_set_2

            #### for unb Test set
            x_norm = load_data_from_txt(input_file="dataset/unb/Normal_UNB.txt", end=59832)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])
            # split normal data
            x_norm_train_all, x_norm_test, y_norm_train_all, y_norm_test = train_test_split(x_norm, y_norm,
                                                                                            test_size=test_size,
                                                                                            random_state=random_state)
            x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all, y_norm_train_all,
                                                                                  test_size=0.125,
                                                                                  random_state=random_state)

            x_attack_11 = load_data_from_txt(input_file="dataset/unb/DoSHulk_UNB.txt", end=11530)
            x_attack_12 = load_data_from_txt(input_file="dataset/unb/DOSSlowHttpTest_UNB.txt", end=6414)
            # x_attack_13 = load_data_from_txt(input_file="dataset/unb/UNB_DosGoldenEye_UNB_IDS2017.txt", end=1268)
            # x_attack_14 = load_data_from_txt(input_file="dataset/unb/UNB_DoSSlowloris_UNB_IDS2017.txt", end=16741)

            x_attack_1 = np.concatenate([x_attack_11, x_attack_12])
            y_attack_1 = np.concatenate(
                [np.zeros([x_attack_11.shape[0]]), np.zeros([x_attack_12.shape[0]])])
            y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))

            # split attackack data
            x_attack_train_all, x_attack_test, y_attack_train_all, y_attack_test = train_test_split(x_attack_1,
                                                                                                    y_attack_1,
                                                                                                    test_size=test_size,
                                                                                                    random_state=random_state)
            x_attack_train, x_attack_val, y_attack_train, y_attack_val = train_test_split(x_attack_train_all,
                                                                                          y_attack_train_all,
                                                                                          test_size=0.125,
                                                                                          random_state=random_state)

            UNB_train_set = (x_norm_train, y_norm_train, x_attack_train, y_attack_train)
            UNB_val_set = (x_norm_val, y_norm_val, x_attack_val, y_attack_val)
            UNB_test_set = (x_norm_test, y_norm_test, x_attack_test, y_attack_test)

            # x_attack_11 = load_data_from_txt(input_file="dataset/unb/DoSHulk_UNB.txt", end=11530)
            # x_attack_12 = load_data_from_txt(input_file="dataset/unb/DOSSlowHttpTest_UNB.txt", end=6414)
            x_attack_13 = load_data_from_txt(input_file="dataset/unb/UNB_DosGoldenEye_UNB_IDS2017.txt", end=1268)
            x_attack_14 = load_data_from_txt(input_file="dataset/unb/UNB_DoSSlowloris_UNB_IDS2017.txt", end=16741)

            x_attack_test_UNB = np.concatenate([x_attack_13, x_attack_14])
            y_attack_1 = np.concatenate(
                [np.zeros([x_attack_13.shape[0]]), np.zeros([x_attack_14.shape[0]])])
            y_attack_test_UNB = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))

            # x_attack_SYNT_12 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms",
            #                                       end=37000)
            # x_attack_SYNT_13 = load_data_from_txt(
            #     input_file="dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms",
            #     end=243)
            # x_attack_test_2 = np.concatenate([x_attack_13, x_attack_14])
            # y_attack_test_2 = np.concatenate([np.zeros([x_attack_13.shape[0]]), np.zeros([x_attack_14.shape[0]])])
            # y_attack_test_2 = np.reshape(y_attack_test_2, (y_attack_test_2.shape[0], 1))

            UNB_test_set_2 = (x_norm_test, y_norm_test, x_attack_test_UNB, y_attack_test_UNB)

            return UNB_train_set, UNB_val_set, UNB_test_set, UNB_test_set_2

#
# def dump_model(model, out_file):
#     """
#         save model to disk
#     :param model:
#     :param out_file:
#     :return:
#     """
#     out_dir = os.path.split(out_file)[0]
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#
#     with open(out_file, 'wb') as f:
#         pickle.dump(model, f)
#
#     print("Model saved in %s" % out_file)
#
#     return out_file
#
#
# def load_model(input_file):
#     """
#
#     :param input_file:
#     :return:
#     """
#     print("Loading model...")
#     with open(input_file, 'rb') as f:
#         model = pickle.load(f)
#     print("Model loaded.")
#
#     return model
#
#
# def show_data(data, x_label='epochs', y_label='y', fig_label='', title=''):
#     plt.figure()
#     plt.plot(data, 'r', alpha=0.5, label=fig_label)
#     plt.legend(loc='upper right')
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(title)
#     plt.show()
#
#
# def show_data_2(data1, data2, x_label='epochs', y_label='mean loss', title=''):
#     plt.figure()
#     plt.plot(data1, 'r', alpha=0.5, label='train_loss in each epoch')
#     plt.plot(data2, 'b', alpha=0.5, label='val loss in each epoch')
#     # plt.plot(new_decision_data[:, 2], 'g', alpha=0.5, label='D_G_fake')
#     plt.legend(loc='upper right')
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(title)
#     plt.show()
#
#
# def get_variable_name(data_var):
#     """
#         get variable name as string
#     :param data_var:
#     :return:
#     """
#     name = ''
#     keys = locals().keys()
#     for key, val in locals().items():
#         # if id(key) == id(data_var):
#         print(key, id(key), id(data_var), key is data_var)
#         # if id(key) == id(data_var):
#         if val == data_var:
#             name = key
#             break
#
#     return name
