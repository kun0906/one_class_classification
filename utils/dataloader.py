"""
    load features data from txt
"""
import os
from collections import OrderedDict, Counter

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


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

    :param train_set_dict:  train_set_dict = {'train_set': {'x': x_train, 'y': y_train}}
    :return:
    """
    for key, value in train_set_dict.items():
        print(f'{key}:', end='')
        x = value['x']
        y = value['y']
        print(f'x.shape: {x.shape}, y.shape: {Counter(y.reshape(-1,))}')


def load_data_from_txt(input_file, start=0, end=77989, discretize_flg=True):
    """
        features: ts, sip, dip, sport, dport, proto, dura, orig_pks, reply_pks, orig_bytes, reply_bytes, orig_min_pkt_size, orig_max_pkt_size, reply_min_pkt_size, reply_max_pkt_size, orig_min_interval, orig_max_interval, reply_min_interval, reply_max_interval, orig_min_ttl, orig_max_ttl, reply_min_ttl, reply_max_ttl, urg, ack, psh, rst, syn, fin, is_new, state, prev_state
            idx : 0    1    2    3      4      5      6      7         8          9            10            11                 12                  13                  14                       15            16                      17               18                19          20             21           22            23   24   25   26   27   28   29      30      31
    :param input_file:
    :param start:
    :param end:
    :param discretize_flg:
    :return:
    """
    data = []
    cnt = 0
    with open(input_file, 'r') as hdl:
        line = hdl.readline()
        while line != '' and cnt < end:
            if line.startswith('ts'):
                line = hdl.readline()
                continue
            if cnt >= start:
                if discretize_flg:
                    arr = line.split(',')[5:]
                    features = discretize_features(arr)
                    data.append(features)  # without : "ts, sip, dip, sport, dport"
                else:
                    data.append(line.split(',')[5:])  # without : "ts, sip, dip, sport, dport"
            line = hdl.readline()
            cnt += 1

    return np.asarray(data, dtype=float)


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


def load_data_and_discretize_features(case='uSc1C2_z-score_20_14', input_dir="input_data/dataset", shuffle_flg=False,
                                      random_state=42, test_size=0.2):
    '''
        case[0] = u/s for Unsupervised or Supervised
        case[3] = Scenario
        case[5] = Source
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
    root_dir = input_dir
    if case[3] == '1':  # Experiment 1
        if case[5] == '1':  # training and testing on SYNT (simulated data)
            x_norm = load_data_from_txt(input_file=os.path.join(root_dir, "synthetic_dataset/Sess_normal_0.txt"),
                                        end=77989)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])

            x_attack_11 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"),
                end=36000)
            x_attack_12 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Recursive_GET.dms"),
                end=37000)
            x_attack_13 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms"),
                end=243)
            x_attack_14 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"),
                end=1000)
        elif case[5] == '2':  # training and testing on unb
            x_norm = load_data_from_txt(input_file=os.path.join(root_dir, "unb/Normal_UNB.txt"), end=59832)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])

            x_attack_11 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/DoSHulk_UNB.txt"), end=11530)
            x_attack_12 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/DOSSlowHttpTest_UNB.txt"), end=6414)
            x_attack_13 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/UNB_DosGoldenEye_UNB_IDS2017.txt"),
                                             end=1268)
            x_attack_14 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/UNB_DoSSlowloris_UNB_IDS2017.txt"),
                                             end=16741)
        elif case[5] == '3':  # training and testing on mawi
            x_norm = load_data_from_txt(input_file=os.path.join(root_dir, "mawi/Normal_mawi_day1.txt"), end=62000)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])

            '''
               Train set : Validation set : Test set = 7 : 1 : 2,  only for normal samples
            '''
            # normal should be in oder to split.
            x_norm_train_size = int(x_norm.shape[0] * 0.7)  #
            x_norm_val_size = int(x_norm.shape[0] * 0.1)
            x_norm_test_size = int(x_norm.shape[0] * 0.2)
            x_norm_train = x_norm[:x_norm_train_size, :]  # 77989* 0.7
            y_norm_train = y_norm[:x_norm_train_size, :]  #
            x_norm_val = x_norm[x_norm_train_size:(x_norm_train_size + x_norm_val_size), :]  # 77989* 0.1
            y_norm_val = y_norm[x_norm_train_size:(x_norm_train_size + x_norm_val_size), :]
            x_norm_test = x_norm[(x_norm_train_size + x_norm_val_size):, :]  # 77989 * 02.
            y_norm_test = y_norm[(x_norm_train_size + x_norm_val_size):, :]

            x_train = x_norm_train
            y_train = y_norm_train
            x_val = x_norm_val
            y_val = y_norm_val
            x_test = x_norm_test
            y_test = y_norm_test

            train_set_dict = {'train_set': {'x': x_train, 'y': y_train}}
            val_set_dict = {'val_set': {'x': x_val, 'y': y_val}}
            test_sets_dict = OrderedDict({'test_set':{'x':x_test, 'y':y_test}} )  # test sets might more than 1.

            return train_set_dict, val_set_dict, test_sets_dict
        else:
            print('**** other case.')
            pass

        if shuffle_flg:  ### shuffle data
            # # split normal data
            # x_norm_train_all, x_norm_test, y_norm_train_all, y_norm_test = train_test_split(x_norm, y_norm,
            #                                                                                 test_size=test_size,
            #                                                                                 random_state=random_state)
            # x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all,
            #                                                                       y_norm_train_all,
            #                                                                       test_size=0.125,
            #                                                                       random_state=random_state)
            # TODO
            pass
        else:  ### not shuffle data
            '''
               Train set : Validation set : Test set = 7 : 1 : 2,  only for normal samples
            '''
            # normal should be in oder to split.
            x_norm_train_size = int(x_norm.shape[0] * 0.7)  #
            x_norm_val_size = int(x_norm.shape[0] * 0.1)
            x_norm_test_size = int(x_norm.shape[0] * 0.2)
            x_norm_train = x_norm[:x_norm_train_size, :]  # 77989* 0.7
            y_norm_train = y_norm[:x_norm_train_size, :]  #
            x_norm_val = x_norm[x_norm_train_size:(x_norm_train_size + x_norm_val_size), :]  # 77989* 0.1
            y_norm_val = y_norm[x_norm_train_size:(x_norm_train_size + x_norm_val_size), :]
            x_norm_test = x_norm[(x_norm_train_size + x_norm_val_size):, :]  # 77989 * 02.
            y_norm_test = y_norm[(x_norm_train_size + x_norm_val_size):, :]

            x_remained_attack, x_test, x_train, x_val, y_remained_attack, y_test, y_train, y_val = sample_from_attacks_with_equal_ratio(
                x_attack_11, x_attack_12, x_attack_13, x_attack_14, x_norm_test, x_norm_test_size, x_norm_train,
                x_norm_train_size, x_norm_val, x_norm_val_size, y_norm_test, y_norm_train, y_norm_val,
                attack_percent=0.01)

            x_test_2 = np.concatenate([x_norm_test, x_remained_attack], axis=0)  # concatenate by rows.
            y_test_2 = np.concatenate([y_norm_test, y_remained_attack], axis=0)

            train_set_dict = {'train_set':{'x': x_train, 'y': y_train}}
            val_set_dict = {'val_set':{'x': x_val, 'y': y_val}}
            test_sets_dict = OrderedDict({'test_set':{'x': x_test, 'y': y_test},
                              'test_set_2': {'x': x_test_2 , 'y': y_test_2}}) # test sets might more than 1.

            print_dict(train_set_dict)
            print_dict(val_set_dict)
            print_dict(test_sets_dict)

            return train_set_dict, val_set_dict, test_sets_dict


    elif case[3] == '2':  # Experiment 2
        if case[5] == '1':
            x_norm = load_data_from_txt(input_file=os.path.join(root_dir, "synthetic_dataset/Sess_normal_0.txt"),
                                        end=77989)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])

            x_attack_11 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"),
                end=36000)
            x_attack_12 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Recursive_GET.dms"),
                end=37000)
            x_attack_13 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms"),
                end=243)
            x_attack_14 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"),
                end=1000)

            # normal should be in oder to split.
            x_norm_train_size = int(x_norm.shape[0] * 0.7)  #
            x_norm_val_size = int(x_norm.shape[0] * 0.1)
            x_norm_test_size = int(x_norm.shape[0] * 0.2)
            x_norm_train = x_norm[:x_norm_train_size, :]  # 77989* 0.7
            y_norm_train = y_norm[:x_norm_train_size, :]  #
            x_norm_val = x_norm[x_norm_train_size:(x_norm_train_size + x_norm_val_size), :]  # 77989* 0.1
            y_norm_val = y_norm[x_norm_train_size:(x_norm_train_size + x_norm_val_size), :]
            x_norm_test = x_norm[(x_norm_train_size + x_norm_val_size):, :]  # 77989 * 02.
            y_norm_test = y_norm[(x_norm_train_size + x_norm_val_size):, :]

            x_remained_attack, x_test, x_train, x_val, y_remained_attack, y_test, y_train, y_val = sample_from_attacks_with_equal_ratio(
                x_attack_11, x_attack_12, x_attack_13, x_attack_14, x_norm_test, x_norm_test_size, x_norm_train,
                x_norm_train_size, x_norm_val, x_norm_val_size, y_norm_test, y_norm_train, y_norm_val,
                attack_percent=0.01)

            x_attack_1 = np.concatenate([x_attack_11, x_attack_12, x_attack_13, x_attack_14])
            y_attack_1 = np.concatenate(
                [np.zeros([x_attack_11.shape[0]]), np.zeros([x_attack_12.shape[0]]), np.zeros([x_attack_13.shape[0]]),
                 np.zeros([x_attack_14.shape[0]])])
            y_attack_1 = np.reshape(y_attack_1, (y_attack_1.shape[0], 1))

            SYNT_train_set = (x_train, y_train)
            SYNT_val_set = (x_val, y_val)
            SYNT_test_set = (x_test, y_test)

            # test on UNB
            x_norm_UNB = load_data_from_txt(input_file=os.path.join(root_dir, "unb/Normal_UNB.txt"), end=59832)
            y_norm_UNB = np.ones(shape=[x_norm_UNB.shape[0], 1])

            x_attack_UNB_11 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/DoSHulk_UNB.txt"), end=11530)
            x_attack_UNB_12 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/DOSSlowHttpTest_UNB.txt"),
                                                 end=6414)
            x_attack_UNB_13 = load_data_from_txt(
                input_file=os.path.join(root_dir, "unb/UNB_DosGoldenEye_UNB_IDS2017.txt"),
                end=1268)
            x_attack_UNB_14 = load_data_from_txt(
                input_file=os.path.join(root_dir, "unb/UNB_DoSSlowloris_UNB_IDS2017.txt"),
                end=16741)

            # normal should be in oder to split.
            x_norm_train_size = int(x_norm_UNB.shape[0] * 0.7)  #
            x_norm_val_size = int(x_norm_UNB.shape[0] * 0.1)
            x_norm_test_size = int(x_norm_UNB.shape[0] * 0.2)
            x_norm_train = x_norm_UNB[:x_norm_train_size, :]  # 77989* 0.7
            y_norm_train = y_norm_UNB[:x_norm_train_size, :]  #
            x_norm_val = x_norm_UNB[x_norm_train_size:(x_norm_train_size + x_norm_val_size), :]  # 77989* 0.1
            y_norm_val = y_norm_UNB[x_norm_train_size:(x_norm_train_size + x_norm_val_size), :]
            x_norm_test = x_norm_UNB[(x_norm_train_size + x_norm_val_size):, :]  # 77989 * 02.
            y_norm_test = y_norm_UNB[(x_norm_train_size + x_norm_val_size):, :]

            x_remained_attack, x_test, x_train, x_val, y_remained_attack, y_test, y_train, y_val = sample_from_attacks_with_equal_ratio(
                x_attack_UNB_11, x_attack_UNB_12, x_attack_UNB_13, x_attack_UNB_14, x_norm_test, x_norm_test_size,
                x_norm_train,
                x_norm_train_size, x_norm_val, x_norm_val_size, y_norm_test, y_norm_train, y_norm_val,
                attack_percent=0.01)

            UNB_train_set = (x_train, y_train)
            UNB_val_set = (x_val, y_val)
            UNB_test_set = (x_test, y_test)

            ### testing on mawi
            x_norm_MAWI = load_data_from_txt(input_file=os.path.join(root_dir, "mawi/Normal_mawi_day1.txt"), end=62000)
            y_norm_MAWI = np.ones(shape=[x_norm_MAWI.shape[0], 1])

            x_norm_train_MAWI, x_norm_test_MAWI, y_norm_train_MAWI, y_norm_test_MAWI = train_test_split(x_norm_MAWI,
                                                                                                        y_norm_MAWI,
                                                                                                        test_size=test_size,
                                                                                                        random_state=random_state)
            MAWI_test_set = (x_norm_test_MAWI, y_norm_test_MAWI)

            test_sets_dict = OrderedDict(
                {"SYNT_test": SYNT_test_set, 'UNB_test': UNB_test_set, 'MAWI_test': MAWI_test_set})

            return SYNT_train_set, SYNT_val_set, test_sets_dict

        elif case[5] == '2':  # training and testing on unb
            x_norm = load_data_from_txt(input_file=os.path.join(root_dir, "unb/Normal_UNB.txt"), end=59832)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])

            x_attack_11 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/DoSHulk_UNB.txt"), end=11530)
            x_attack_12 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/DOSSlowHttpTest_UNB.txt"), end=6414)
            x_attack_13 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/UNB_DosGoldenEye_UNB_IDS2017.txt"),
                                             end=1268)
            x_attack_14 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/UNB_DoSSlowloris_UNB_IDS2017.txt"),
                                             end=16741)

            # normal should be in oder to split.
            x_norm_train_size = int(x_norm.shape[0] * 0.7)  #
            x_norm_val_size = int(x_norm.shape[0] * 0.1)
            x_norm_test_size = int(x_norm.shape[0] * 0.2)
            x_norm_train = x_norm[:x_norm_train_size, :]  # 77989* 0.7
            y_norm_train = y_norm[:x_norm_train_size, :]  #
            x_norm_val = x_norm[x_norm_train_size:(x_norm_train_size + x_norm_val_size), :]  # 77989* 0.1
            y_norm_val = y_norm[x_norm_train_size:(x_norm_train_size + x_norm_val_size), :]
            x_norm_test = x_norm[(x_norm_train_size + x_norm_val_size):, :]  # 77989 * 02.
            y_norm_test = y_norm[(x_norm_train_size + x_norm_val_size):, :]

            x_remained_attack, x_test, x_train, x_val, y_remained_attack, y_test, y_train, y_val = sample_from_attacks_with_equal_ratio(
                x_attack_11, x_attack_12, x_attack_13, x_attack_14, x_norm_test, x_norm_test_size, x_norm_train,
                x_norm_train_size, x_norm_val, x_norm_val_size, y_norm_test, y_norm_train, y_norm_val,
                attack_percent=0.01)

            UNB_train_set = (x_train, y_train)
            UNB_val_set = (x_val, y_val)
            UNB_test_set = (x_test, y_test)

            x_norm_SYNT = load_data_from_txt(input_file=os.path.join(root_dir, "synthetic_dataset/Sess_normal_0.txt"),
                                             end=77989)
            y_norm_SYNT = np.ones(shape=[x_norm_SYNT.shape[0], 1])

            x_attack_SYNT_11 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"),
                end=36000)
            x_attack_SYNT_12 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Recursive_GET.dms"),
                end=37000)
            x_attack_SYNT_13 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms"),
                end=243)
            x_attack_SYNT_14 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"),
                end=1000)

            # normal should be in oder to split.
            x_norm_train_size = int(x_norm_SYNT.shape[0] * 0.7)  #
            x_norm_val_size = int(x_norm_SYNT.shape[0] * 0.1)
            x_norm_test_size = int(x_norm_SYNT.shape[0] * 0.2)
            x_norm_train = x_norm_SYNT[:x_norm_train_size, :]  # 77989* 0.7
            y_norm_train = y_norm_SYNT[:x_norm_train_size, :]  #
            x_norm_val = x_norm_SYNT[x_norm_train_size:(x_norm_train_size + x_norm_val_size), :]  # 77989* 0.1
            y_norm_val = y_norm_SYNT[x_norm_train_size:(x_norm_train_size + x_norm_val_size), :]
            x_norm_test = x_norm_SYNT[(x_norm_train_size + x_norm_val_size):, :]  # 77989 * 02.
            y_norm_test = y_norm_SYNT[(x_norm_train_size + x_norm_val_size):, :]

            x_remained_attack, x_test, x_train, x_val, y_remained_attack, y_test, y_train, y_val = sample_from_attacks_with_equal_ratio(
                x_attack_SYNT_11, x_attack_SYNT_12, x_attack_SYNT_13, x_attack_SYNT_14, x_norm_test, x_norm_test_size,
                x_norm_train,
                x_norm_train_size, x_norm_val, x_norm_val_size, y_norm_test, y_norm_train, y_norm_val,
                attack_percent=0.01)

            SYNT_train_set = (x_train, y_train)
            SYNT_val_set = (x_val, y_val)
            SYNT_test_set = (x_test, y_test)

            ### testing on mawi
            x_norm_MAWI = load_data_from_txt(
                input_file=os.path.join(root_dir, "mawi/Normal_mawi_day1.txt"), end=62000)
            y_norm_MAWI = np.ones(shape=[x_norm_MAWI.shape[0], 1])

            x_norm_train_MAWI, x_norm_test_MAWI, y_norm_train_MAWI, y_norm_test_MAWI = train_test_split(x_norm_MAWI,
                                                                                                        y_norm_MAWI,
                                                                                                        test_size=test_size,
                                                                                                        random_state=random_state)
            MAWI_test_set = (x_norm_test_MAWI, y_norm_test_MAWI)

            test_sets_dict = OrderedDict(
                {"SYNT_test": SYNT_test_set, 'UNB_test': UNB_test_set, 'MAWI_test': MAWI_test_set})

            return UNB_train_set, UNB_val_set, test_sets_dict


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
            x_norm = load_data_from_txt(input_file=os.path.join(root_dir, "synthetic_dataset/Sess_normal_0.txt"),
                                        end=77989)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])
            # split normal data
            x_norm_train_all, x_norm_test, y_norm_train_all, y_norm_test = train_test_split(x_norm, y_norm,
                                                                                            test_size=test_size,
                                                                                            random_state=random_state)
            x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all, y_norm_train_all,
                                                                                  test_size=0.125,
                                                                                  random_state=random_state)

            x_attack_11 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"),
                end=36000)
            x_attack_12 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Recursive_GET.dms"),
                end=37000)
            x_attack_13 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms"),
                end=243)
            x_attack_14 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"),
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

            SYNT_train_set = {'x_norm_train': x_norm_train, 'y_norm_train': y_norm_train,
                              'x_attack_train': x_attack_train, 'y_attack_train': y_attack_train}
            SYNT_val_set = {'x_norm_val': x_norm_val, 'y_norm_val': y_norm_val,
                            'x_attack_val': x_attack_val, 'y_attack_val': y_attack_val}
            SYNT_test_set = {'x_norm_test': x_norm_test, 'y_norm_test': y_norm_test,
                             'x_attack_test': x_attack_test, 'y_attack_test': y_attack_test}

            # x_attack_12 = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms",
            #                                   end=37000)
            # x_attack_13 = load_data_from_txt(
            #      input_file="dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms",
            #      end=243)
            # x_attack_test_2 = np.concatenate([x_attack_12, x_attack_13])
            # y_attack_test_2 = np.concatenate([np.zeros([x_attack_12.shape[0]]), np.zeros([x_attack_13.shape[0]])])
            # y_attack_test_2 = np.reshape(y_attack_test_2, (y_attack_test_2.shape[0], 1))
            # SYNT_test_set_2 = (x_norm_test, y_norm_test, x_attack_test_2, y_attack_test_2)

            x_attack_UNB_11 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/DoSHulk_UNB.txt"), end=11530)
            x_attack_UNB_12 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/DOSSlowHttpTest_UNB.txt"),
                                                 end=6414)
            x_attack_UNB_13 = load_data_from_txt(
                input_file=os.path.join(root_dir, "unb/UNB_DosGoldenEye_UNB_IDS2017.txt"), end=1268)
            x_attack_UNB_14 = load_data_from_txt(
                input_file=os.path.join(root_dir, "unb/UNB_DoSSlowloris_UNB_IDS2017.txt"), end=16741)

            x_attack_UNB_1 = np.concatenate([x_attack_UNB_11, x_attack_UNB_12, x_attack_UNB_13, x_attack_UNB_14])
            y_attack_UNB_1 = np.concatenate(
                [np.zeros([x_attack_UNB_11.shape[0]]), np.zeros([x_attack_UNB_12.shape[0]]),
                 np.zeros([x_attack_UNB_13.shape[0]]),
                 np.zeros([x_attack_UNB_14.shape[0]])])
            y_attack_UNB_1 = np.reshape(y_attack_UNB_1, (y_attack_UNB_1.shape[0], 1))

            SYNT_test_set_2 = {'x_norm_test': x_norm_test, 'y_norm_test': y_norm_test,
                               'x_attack_test': x_attack_UNB_1, 'y_attack_test': y_attack_UNB_1}
            test_sets_dict = OrderedDict(
                {"SYNT_test": SYNT_test_set, 'SYNT_test_2': SYNT_test_set_2})

            return SYNT_train_set, SYNT_val_set, test_sets_dict


        elif case[5] == '2':
            # unb  because of unb attack and normal exist huge difference, so DT can easily distingusih them on test set 1 and test set 2.
            # x_norm = dataset(filename=["dataset/unb/Normal_UNB.txt"], label=1, end=59832)
            x_norm = load_data_from_txt(input_file=os.path.join(root_dir, "dataset/unb/Normal_UNB.txt", end=59832))
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

            x_attack_11 = load_data_from_txt(
                input_file=os.path.join(root_dir, "dataset/unb/DoSHulk_UNB.txt", end=11530))
            x_attack_12 = load_data_from_txt(
                input_file=os.path.join(root_dir, "dataset/unb/DOSSlowHttpTest_UNB.txt", end=6414))
            x_attack_13 = load_data_from_txt(
                input_file=os.path.join(root_dir, "dataset/unb/UNB_DosGoldenEye_UNB_IDS2017.txt", end=1268))
            x_attack_14 = load_data_from_txt(
                input_file=os.path.join(root_dir, "dataset/unb/UNB_DoSSlowloris_UNB_IDS2017.txt", end=16741))

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

            UNB_train_set = {'x_norm_train': x_norm_train, 'y_norm_train': y_norm_train,
                             'x_attack_train': x_attack_train, 'y_attack_train': y_attack_train}
            UNB_val_set = {'x_norm_val': x_norm_val, 'y_norm_val': y_norm_val,
                           'x_attack_val': x_attack_val, 'y_attack_val': y_attack_val}
            UNB_test_set = {'x_norm_test': x_norm_test, 'y_norm_test': y_norm_test,
                            'x_attack_test': x_attack_test, 'y_attack_test': y_attack_test}

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

            UNB_test_set_2 = {'x_norm_test': x_norm_test, 'y_norm_test': y_norm_test,
                              'x_attack_test': x_attack_test_SYNT, 'y_attack_test': y_attack_test_SYNT}
            test_sets_dict = OrderedDict({"UNB_test": UNB_test_set, 'UNB_test_2': UNB_test_set_2})

            return UNB_train_set, UNB_val_set, test_sets_dict

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

    root_dir = "/Users/kunyang/PycharmProjects/anomaly_detection_20190611/input_data/dataset"
    if case[3] == '1':  # Experiment 1
        if case[5] == '1':  # training and testing on SYNT (simulated data)
            ### shuffle data
            ### training on SYNT
            # x_norm = dataset(filename=["dataset/synthetic_dataset/Sess_normal_0.txt"], label=1, end=77989)
            x_norm = load_data_from_txt(input_file=os.path.join(root_dir, "synthetic_dataset/Sess_normal_0.txt"),
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
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"),
                end=36000)
            x_attack_12 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Recursive_GET.dms"),
                end=37000)
            x_attack_13 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms"),
                end=243)
            x_attack_14 = load_data_from_txt(
                input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"),
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
            x_norm = load_data_from_txt(input_file=os.path.join(root_dir, "unb/Normal_UNB.txt"), end=59832)
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

            x_attack_11 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/DoSHulk_UNB.txt"), end=11530)
            x_attack_12 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/DOSSlowHttpTest_UNB.txt"), end=6414)
            x_attack_13 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/UNB_DosGoldenEye_UNB_IDS2017.txt"),
                                             end=1268)
            x_attack_14 = load_data_from_txt(input_file=os.path.join(root_dir, "unb/UNB_DoSSlowloris_UNB_IDS2017.txt"),
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


        elif case[
            5] == '2':  # unb  because of unb attack and normal exist huge difference, so DT can easily distingusih them on test set 1 and test set 2.
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
