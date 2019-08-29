import os
from collections import OrderedDict, Counter

import numpy as np

from preprocess.feature_selection import select_sub_features_data
from preprocess.normalization import normalize_data_with_z_score, normalize_data_with_min_max
from utils.dataloader import get_each_dataset_for_experiment_1


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
    print(
        f'input_file: {input_file}, data_range: {data_range}, label: {label}, discretize_flg: {discretize_flg}')
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


def load_norm_attack_data(case='', input_dir='input_data/datasets', label_dict={'norm': 1, 'attack': 0}):
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
    elif case[3] == '2':  # experiment 2
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


        else:  # if case[5] == '3': # for SYNT test
            # TODO
            pass
    else:
        print(f'case:{case} is not correct, please check it again.')
        return -1

    return norm_dict, attack_dict_lst


def get_experiment_datasets(case='', norm_dict={}, attack_dict_lst={}):
    train_set_dict = OrderedDict()
    val_set_dict = OrderedDict()
    test_set_dict = OrderedDict()

    if case[3] == '1':  # experiment I
        if case[5] == '1':
            name = 'SYNT'
        elif case[5] == '2':
            name = 'UNB'
        elif case[5] == '3':
            name = 'MAWI'
        else:
            return -1
        train_set_dict, val_set_dict, test_set_dict = get_each_dataset_for_experiment_1(norm_dict=norm_dict,
                                                                                        attack_dict_lst=attack_dict_lst,
                                                                                        name=name,
                                                                                        insert_attack_percent=0.01,
                                                                                        test_set_percent=0.2,
                                                                                        shuffle_flg=False,
                                                                                        random_state=42)
    elif case[3] == '2':
        pass

    elif case[3] == '3':
        pass
    else:
        print(f'case:{case} is not correct, please check it again.')
        return -1

    datasets_dict = {'train_set_dict': train_set_dict, 'val_set_dict': val_set_dict, 'test_set_dict': test_set_dict}

    print_dict(train_set_dict)
    print_dict(val_set_dict)
    print_dict(test_set_dict)

    return datasets_dict


def data_print(X_train_mu):
    return [float(v) for v in X_train_mu]


class Dataset():

    def __init__(self, case='experiment_1', input_dir='',
                 random_state=42):
        self.case = case
        self.input_dir = input_dir
        self.random_state = random_state

        self.norm_dict, self.attack_dict_lst = load_norm_attack_data(case=self.case, input_dir=self.input_dir,
                                                                     label_dict={'norm': 1, 'attack': 0})
        self.num_features = self.norm_dict['X'].shape[1]

        self.datasets_dict = get_experiment_datasets(case=self.case, norm_dict=self.norm_dict,
                                                     attack_dict_lst=self.attack_dict_lst)

    def normalize_data(self, norm_method='z-score', train_set_key='SYNT', not_normalized_features_lst=[]):

        train_set_key = train_set_key + '_train_set'
        print(f'\n-normalize {train_set_key} with {norm_method}')

        train_set_dict = self.datasets_dict['train_set_dict']
        val_set_dict = self.datasets_dict['val_set_dict']
        test_set_dict = self.datasets_dict['test_set_dict']

        new_train_set_dict = OrderedDict()
        new_val_set_dict = OrderedDict()
        new_test_set_dict = OrderedDict()

        if norm_method == 'z-score':
            X_train = train_set_dict[train_set_key]['X']
            new_X_train, X_train_mu, X_train_std = normalize_data_with_z_score(X_train, mu='', d_std='',
                                                                               not_normalized_features_lst=not_normalized_features_lst)
            print(f'--mu and std obtained from \'{train_set_key}\', in which,\nX_train_mu:{data_print(X_train_mu)},\n'
                  f'X_train_std:{data_print(X_train_std)}')
            new_train_set_dict.update({train_set_key: {}})
            new_train_set_dict[train_set_key].update({'X': new_X_train})
            new_train_set_dict[train_set_key]['y'] = train_set_dict[train_set_key]['y']

            for key, value_dict in val_set_dict.items():
                print(
                    f'\n--normalize {key} with {norm_method} and parameters (mu and std) obtained from {train_set_key},'
                    f'in which,\nX_train_mu:{data_print(X_train_mu)},\nX_train_std:{data_print(X_train_std)}')
                x_val = value_dict['X']
                new_x_val, _, _ = normalize_data_with_z_score(x_val, mu=X_train_mu, d_std=X_train_std,
                                                              not_normalized_features_lst=not_normalized_features_lst)
                new_val_set_dict[key] = {}
                new_val_set_dict[key]['X'] = new_x_val
                new_val_set_dict[key]['y'] = value_dict['y']

            for key, value_dict in test_set_dict.items():
                print(
                    f'\n--normalize {key} with {norm_method} and parameters (mu and std) obtained from {train_set_key}, '
                    f'in which,\nX_train_mu:{data_print(X_train_mu)},\nX_train_std:{data_print(X_train_std)}')
                x_test = value_dict['X']
                new_x_test, _, _ = normalize_data_with_z_score(x_test, mu=X_train_mu, d_std=X_train_std,
                                                               not_normalized_features_lst=not_normalized_features_lst)
                new_test_set_dict[key] = {}
                new_test_set_dict[key]['X'] = new_x_test
                new_test_set_dict[key]['y'] = value_dict['y']

        elif norm_method == 'min-max':
            X_train = train_set_dict[train_set_key]['X']
            new_X_train, X_train_min, X_train_max = normalize_data_with_min_max(X_train, min_val='',
                                                                                max_val='',
                                                                                not_normalized_features_lst=not_normalized_features_lst)
            print(f'--min and max obtained from \'{train_set_key}\', '
                  f'in which,\nX_train_min:{data_print(X_train_min)},\nX_train_max:{data_print(X_train_max)}')
            new_train_set_dict[train_set_key] = {}
            new_train_set_dict[train_set_key]['X'] = new_X_train
            new_train_set_dict[train_set_key]['y'] = train_set_dict[train_set_key]['y']

            for key, value_dict in val_set_dict.items():
                print(
                    f'\n--normalize {key} with {norm_method} and parameters (min and max) obtained from {train_set_key}, '
                    f'in which,\nX_train_min:{data_print(X_train_min)},\nX_train_max:{data_print(X_train_max)}')
                x_val = value_dict['X']
                new_x_val, _, _ = normalize_data_with_min_max(x_val, min_val=X_train_min,
                                                              max_val=X_train_max,
                                                              not_normalized_features_lst=not_normalized_features_lst)
                new_val_set_dict[key] = {}
                new_val_set_dict[key]['X'] = new_x_val
                new_val_set_dict[key]['y'] = value_dict['y']

            for key, value_dict in test_set_dict.items():
                print(
                    f'\n--normalize {key} with {norm_method} and parameters (min and max) obtained from {train_set_key}, '
                    f'in which,\nX_train_min:{data_print(X_train_min)},\nX_train_max:{data_print(X_train_max)}')
                x_test = value_dict['X']
                new_x_test, _, _ = normalize_data_with_min_max(x_test, min_val=X_train_min,
                                                               max_val=X_train_max,
                                                               not_normalized_features_lst=not_normalized_features_lst)
                new_test_set_dict[key] = {}
                new_test_set_dict[key]['X'] = new_x_test
                new_test_set_dict[key]['y'] = value_dict['y']
        else:
            print(f'norm_method {norm_method} is not correct.')
            return -1

        return {'train_set_dict': new_train_set_dict, 'val_set_dict': new_val_set_dict,
                'test_set_dict': new_test_set_dict}

    def selected_sub_features(self, features_lst=[]):

        if len(features_lst) > 0:

            for key_set, value_dict in self.datasets_dict.items():
                for sub_key, sub_value in value_dict.items():
                    new_x = select_sub_features_data(sub_value['X'], features_lst)
                    value_dict[sub_key]['X'] = new_x
                    value_dict[sub_key]['y'] = sub_value['y']
            self.num_features = len(features_lst)
            print(f'using selected_features_lst (size={self.num_features}): {features_lst}')
        else:
            num_features = self.num_features
            print(f'using all features: {num_features}')