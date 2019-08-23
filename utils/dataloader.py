"""
    load features data from txt
"""
from utils.csv_dataloader import csv_dataloader
import os

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#
# from utils.read_data_un import Dataset
# from utils.utils import normalise_data, zscore


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
    else:       # features_arr[0] == '17':  # 17: udp
        features.extend([0, 1])

    features.extend(features_arr[1:])

    return features


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
                    features=discretize_features(arr)
                    data.append(features)  # without : "ts, sip, dip, sport, dport"
                else:
                    data.append(line.split(',')[5:])  # without : "ts, sip, dip, sport, dport"
            line = hdl.readline()
            cnt += 1

    return np.asarray(data, dtype=float)


def load_data_and_discretize_features(case, input_dir = "input_data/dataset", shuffle_flg=False, random_state=42,test_size=0.2):
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
    root_dir =input_dir
    if case[3] == '1':  # Experiment 1
        if case[5] == '1':  # training and testing on SYNT (simulated data)
            ### training on SYNT
            x_norm = load_data_from_txt(input_file=os.path.join(root_dir, "synthetic_dataset/Sess_normal_0.txt"),
                                        end=77989)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])

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


        elif case[5] == '2':  # training and testing on unb
            ### training on unb
            # x_norm = dataset(filename=["dataset/unb/Normal_UNB.txt"], label=1, end=59832)
            x_norm = load_data_from_txt(input_file=os.path.join(root_dir, "unb/Normal_UNB.txt"), end=59832)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])

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

        elif case[5] == '3':  # training and testing on mawi
            # x_norm = dataset(filename=["dataset/mawi/Normal_mawi_day1.txt"], label=1, end=62000)
            x_norm = load_data_from_txt(input_file="dataset/mawi/Normal_mawi_day1.txt", end=62000)
            y_norm = np.ones(shape=[x_norm.shape[0], 1])

            x_attack_1 = None
            y_attack_1 = None

        else:
            print('**** other case.')
            pass

        if shuffle_flg:  ### shuffle data
            # split normal data
            x_norm_train_all, x_norm_test, y_norm_train_all, y_norm_test = train_test_split(x_norm, y_norm,
                                                                                            test_size=test_size,
                                                                                            random_state=random_state)
            x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all,
                                                                                  y_norm_train_all,
                                                                                  test_size=0.125,
                                                                                  random_state=random_state)
        else:  ### not shuffle data



            train_size = int(x_norm.shape[0] * 0.7 * 0.875)
            val_size = int(x_norm.shape[0] * 0.7 * 0.125)
            x_norm_train = x_norm[:train_size, :]  # 77989* 0.7* 0.875
            y_norm_train = y_norm[:train_size]

            attack_percent=0.1
            # insert 5% attack to train.
            x_attack_train_size = int(train_size*attack_percent)
            attack_test_size = 1- x_attack_train_size/ x_attack_1.shape[0]
            print(f'{attack_percent*100}% attack samples are used to train, the size is {x_attack_train_size} of all attack_samples:{x_attack_1.shape[0]}')
            x_attack_train, x_attack_test_raw, y_attack_train, y_attack_test_raw = train_test_split(x_attack_1,
                                                                    y_attack_1,
                                                                    test_size=attack_test_size,
                                                                    random_state=random_state)
            x_norm_train = np.concatenate([x_norm_train, x_attack_train])
            y_norm_train = np.concatenate([y_norm_train, y_attack_train])


            x_norm_val = x_norm[train_size:train_size + val_size, :]
            y_norm_val = y_norm[train_size:train_size + val_size]
            # insert 5% attack to val.
            x_attack_val_size = int(val_size*attack_percent)
            attack_test_size = 1 - x_attack_val_size / x_attack_test_raw.shape[0]   #
            print(f'({attack_percent*100}% * 0.95) attack samples are used to val, the size is {x_attack_val_size} of all attack_samples:{x_attack_1.shape[0]}')
            x_attack_val, x_attack_test, y_attack_val, y_attack_test = train_test_split(x_attack_test_raw,
                                                                                          y_attack_test_raw,
                                                                                          test_size=attack_test_size,
                                                                                          random_state=random_state)

            x_norm_val = np.concatenate([x_norm_val, x_attack_val])
            y_norm_val = np.concatenate([y_norm_val, y_attack_val])


            x_norm_test = x_norm[train_size + val_size:, :]  # 77989 * 0.2
            y_norm_test = y_norm[train_size + val_size:]

            # x_norm_test = np.concatenate([x_norm_test, x_attack_test])
            # y_norm_test = np.concatenate([y_norm_test, y_attack_test])

            x_attack_1 = x_attack_test
            y_attack_1 = y_attack_test

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
            if shuffle_flg:  ### shuffle data
                # split normal data
                x_norm_train_all, x_norm_test, y_norm_train_all, y_norm_test = train_test_split(x_norm, y_norm,
                                                                                                test_size=test_size,
                                                                                                random_state=random_state)
                x_norm_train, x_norm_val, y_norm_train, y_norm_val = train_test_split(x_norm_train_all,
                                                                                      y_norm_train_all,
                                                                                      test_size=0.125,
                                                                                      random_state=random_state)
            else:  ### not shuffle data
                train_size = int(x_norm.shape[0] * 0.7 * 0.875)
                val_size = int(x_norm.shape[0] * 0.7 * 0.125)
                x_norm_train = x_norm[:train_size, :]  # 77989* 0.7* 0.875
                y_norm_train = y_norm[:train_size]
                x_norm_val = x_norm[train_size:train_size + val_size, :]
                y_norm_val = y_norm[train_size:train_size + val_size]
                x_norm_test = x_norm[train_size + val_size:, :]  # 77989 * 0.2
                y_norm_test = y_norm[train_size + val_size:]


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
            x_norm = load_data_from_txt(input_file=os.path.join(root_dir, "synthetic_dataset/Sess_normal_0.txt"), end=77989)
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
            x_attack_11 = load_data_from_txt(input_file=os.path.join(root_dir, "synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"),
                                             end=36000)
            x_attack_12 = load_data_from_txt(input_file=os.path.join(root_dir,"synthetic_dataset/Sess_DDoS_Recursive_GET.dms"),
                                             end=37000)
            x_attack_13 = load_data_from_txt(
                input_file=os.path.join(root_dir,"synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms"),
                end=243)
            x_attack_14 = load_data_from_txt(
                input_file=os.path.join(root_dir,"synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"),
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
            x_norm = load_data_from_txt(input_file=os.path.join(root_dir,"unb/Normal_UNB.txt"), end=59832)
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

            x_attack_11 = load_data_from_txt(input_file=os.path.join(root_dir,"unb/DoSHulk_UNB.txt"), end=11530)
            x_attack_12 = load_data_from_txt(input_file=os.path.join(root_dir,"unb/DOSSlowHttpTest_UNB.txt"), end=6414)
            x_attack_13 = load_data_from_txt(input_file=os.path.join(root_dir,"unb/UNB_DosGoldenEye_UNB_IDS2017.txt"), end=1268)
            x_attack_14 = load_data_from_txt(input_file=os.path.join(root_dir,"unb/UNB_DoSSlowloris_UNB_IDS2017.txt"), end=16741)
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

    #         data_train = dataset(filename=["dataset/synthetic_dataset/dt_train00.txt"], label=1)
    #         data_train_labels = np.concatenate([np.ones([6000]), np.zeros([5902])])
    #
    #         data_test_1 = dataset(filename=["dataset/synthetic_dataset/Sess_normal_0.txt"], label=1, start=49600,
    #                               end=62000,
    #                               mean=data_train.mean, std=data_train.std).data
    #         data_test_2 = dataset(filename=["dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms"], label=1,
    #                               start=26600,
    #                               end=34200, mean=data_train.mean, std=data_train.std).data
    #         data_test_3 = dataset(
    #             filename=["dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"],
    #             label=0, start=800, end=1000, mean=data_train.mean, std=data_train.std).data
    #         data_test_4 = dataset(
    #             filename=["dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms"], label=0,
    #             end=243, mean=data_train.mean, std=data_train.std).data
    #         data_test_5 = dataset(
    #             filename=["dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"], label=0,
    #             end=1000, mean=data_train.mean, std=data_train.std).data
    #
    #     # SYNT
    #     # data_test_1 = dataset(filename = ["dataset/synthetic_dataset/Sess_normal_0.txt"], label = 1,start = 49600, end = 62000, mean = data_train.mean, std = data_train.std).data
    #     # data_test_2 = dataset(filename = ["dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"], label = 1, end = 35000, mean = data_train.mean, std = data_train.std).data
    #     # data_test_3 = dataset(filename = ["dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms"], label = 0, end = 243, mean = data_train.mean, std = data_train.std).data
    #
    #     # unb
    #     # data_test_1 = dataset(filename = ["dataset/unb/Normal_UNB.txt"], label = 1, start = 47865, end = 59832, mean = data_train.mean, std = data_train.std).data
    #     # data_test_2 = dataset(filename = ["dataset/unb/DoSHulk_UNB.txt"], label = 0, start = 9224,end = 11530, mean = data_train.mean, std = data_train.std).data
    #     # data_test_3 = dataset(filename = ["dataset/unb/DOSSlowHttpTest_UNB.txt"], label = 0,start = 5131, end = 6414, mean = data_train.mean, std = data_train.std).data
    #     # data_test_4 = dataset(filename = ["dataset/unb/UNB_DosGoldenEye_UNB_IDS2017.txt"], label = 0,start = 1014, end = 1268, mean = data_train.mean, std = data_train.std).data
    #     # data_test_5 = dataset(filename = ["dataset/unb/UNB_DoSSlowloris_UNB_IDS2017.txt"], label = 0,start = 13392, end = 16741, mean = data_train.mean, std = data_train.std).data
    #
    #     # SYNT
    #     # data_test_1 = dataset(filename = ["dataset/synthetic_dataset/Sess_normal_0.txt"], label = 1, start = 63292, end = 78889, mean = data_train.mean, std = data_train.std).data
    #     ##data_test_2 = dataset(filename = ["dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt"], label = 0, end = 36000, mean = data_train.mean, std = data_train.std).data
    #     # data_test_3 = dataset(filename = ["dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms"], label = 0, end = 37000, mean = data_train.mean, std = data_train.std).data
    #     # data_test_4 = dataset(filename = ["dataset/synthetic_dataset/Sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait.dms"], label = 0, end = 243, mean = data_train.mean, std = data_train.std).data
    #     # data_test_5 = dataset(filename = ["dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"], label = 0, end = 1000, mean = data_train.mean, std = data_train.std).data
    #
    #     # data_test_1 = dataset(filename = ["dataset/mawi/Normal_mawi_day1.txt"], label = 1,start = 49600, end = 62000, mean = data_train.mean, std = data_train.std).data
    #     # return None, None, data_test_1, np.ones(12400)
    #
    #     s2 = data_test_2.shape[0] + data_test_3.shape[0]  # + data_test_4.shape[0] + data_test_5.shape[0]
    #     s1 = data_test_1.shape[0]
    #     data_test_1 = np.concatenate([data_test_1, data_test_2, data_test_3])
    #     data_test_1_labels = np.concatenate([np.ones(s1), np.zeros(s2)])
    #
    #     s3 = data_test_4.shape[0] + data_test_5.shape[0]  # + data_test_4.shape[0] + data_test_5.shape[0]
    #     s1 = data_test_1.shape[0]
    #     data_test_2 = np.concatenate([data_test_1, data_test_4, data_test_5])
    #     data_test_2_labels = np.concatenate([np.ones(s1), np.zeros(s3)])
    #
    #     return data_train, data_train_labels, data_test_1, data_test_1_labels, data_test_2, data_test_2_labels
    #     # data_train_2 = dataset(filename = ["dataset/synthetic_dataset/Sess_DDoS_Recursive_GET.dms"], label = 0, end = 37000)
    #     # data_train_3 = dataset(filename = ["dataset/synthetic_dataset/Sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait.dms"], label = 0, end = 1000)
    #
    #     # data_train = np.concatenate([data_train_1, data_train_2, data_train_3])
    #     # ata_label = np.concatenate([])
    #
    # try:
    #     return data_train, data_val, data_test_1, data_test_2, data_test_1_labels, None
    # except:
    #     print("Nope")
    #     return None, None, None, None, None, None

# # These are comments from old code
#        '''
#        #vals = []
#        #for t in thres:
#        #    data_train, data_val, data_test_1, data_test_2, data_test_1_labels, data_test_2_labels = load_data(case)
#        #    data_train, data_val, data_test_1 = corr_apply(data_train, data_val, data_test_1, t)
#        #    train_AE(data_train, data_val, case, data_train.shape[1])
#            #fp, fn = test_AE(data_test_1, data_test_1_labels, case, data_test_1.shape[1])
#        #    vals.append(data_test_1.shape[1])
#            #fpr.append(fp*100)
#            #fnr.append(fn*100)
#
#        #print(fpr)
#        #print(fnr)
#
#        #print(thres)
#        #print(vals)
#
#        #axx = [0,0,27]
#        #ayy = [100,0,0]
#
#        #with plt.style.context(('ggplot')):
#        #    fig, ax = plt.subplots()
#        #    ax.plot(vals, fpr, "#A40000", label='FPR')
#        #    ax.plot(vals, fnr, "#B3B72C", label='FNR')
#        #    ax.plot(axx,ayy,"#000000")
#        #    ax.set_facecolor((1, 1, 1))
#        #    plt.legend(loc='best')
#        #    plt.savefig('/Users/sarjakthakkar/Documents/MacBook/Work/hsnl/fpfncorr_up.eps', format='eps', dpi=1500)
#            #pplt.show()
#
#
#        #print(vals)
#        #data_train, data_val, data_test_1 = corr_apply(data_train, data_val, data_test_1,0.1)
#
#
#        #data_train = shuffle(data_train.data, random_state=42)
#        #_,_, data, labels = load_data(case)
#        #data, labels,_,_ = load_data(case)
#        #train_DT(data.data, labels, case)
#        #test_DT(data, labels, case)
#        #data_train, data_val, data_test_1, data_test_2 = corr_apply(data_train, data_val, data_test_1, data_test_2)
#
#        #print(data_train.shape)
#        #plotHist(data_train, np.ones((data_train.shape[0])), case, 27)
#
#
#        #train_AE(data_train[:41882], data_train[41882:], case, 27)
#        #train_AE(data_train.data, data_val, case, data_train.data.shape[1])
#        #print(data_train)
#
#        #test_AE(data_test_1, data_test_1_labels, case, 27)
#        #test_AE(data, labels, case, data.shape[1])
#        #test_OCSVM(data_test_1, data_test_1_labels, case)
#        #test_IF(data_test_1, data_test_1_labels, case)
#        #test_PCA(data_test_1, data_test_1_labels, case)
#        #print("Above is for test 1")
#
#        #test_AE(data_test_2, data_test_2_labels, case, data_train.data.shape[1])
#        #print("Above is for test 2")
#
#        '''
#        '''
#        print("\nCASE : ", case, "\n")
#        '''
#        '''
#        # Test OCSVM
#        test_OCSVM(data_test_1, data_test_1_labels, case)
#        print("Above is for test 1")
#
#        test_OCSVM(data_test_2, data_test_2_labels, case)
#        print("Above is for test 2")
#
#        # Test OCSVM
#        test_IF(data_test_1, data_test_1_labels, case)
#        print("Above is for test 1")
#
#        test_IF(data_test_2, data_test_2_labels, case)
#        print("Above is for test 2")
#
#        # Test OCSVM
#        test_PCA(data_test_1, data_test_1_labels, case)
#        print("Above is for test 1")
#
#        test_PCA(data_test_2, data_test_2_labels, case)
#        print("Above is for test 2")
#        '''
#
#        print("\n")

#
# def load_data(input_data='', norm_flg=True, random_state= 42, train_val_test_percent=[0.7 * 0.9, 0.7 * 0.1, 0.3]):
#     """
#
#     :param input_data:
#     :param norm_flg: default True
#     :param train_val_test_percent: train_set = 0.7*0.9, val_set = 0.7*0.1, test_set = 0.3
#     :return:
#     """
#     if 'mnist' in input_data:
#         from utils.mnist_dataloader import MNIST_DataLoader
#         # load input_data with input_data loader
#         dataset = MNIST_DataLoader(ad_experiment=1)
#         train_set, val_set, test_set = (dataset._X_train, dataset._y_train), (dataset._X_val, dataset._y_val), (
#             dataset._X_test, dataset._y_test)
#     elif 'csv' in input_data:
#         # train_set, val_set, test_set = csv_dataloader(input_data,norm_flg)
#         (X, y) = csv_dataloader(input_data)
#
#         # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_val_test_percent[-1], random_state=1)
#         # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=train_val_test_percent[1],
#         #                                                   random_state=1)
#         # train_set, val_set, test_set = (X_train, y_train), (X_val, y_val), (X_test, y_test)
#
#     else:
#         print('error dataset.')
#         return -1
#
#     # if norm_flg:
#     #     train_set = (normalizate_data(train_set[0]),train_set[1]) # X, y
#     #     val_set=(normalizate_data(val_set[0]),val_set[1])
#     #     test_set=(normalizate_data(test_set[0]),test_set[1])
#
#     return train_set, val_set, test_set
#
#
# def load_data_with_new_principle(input_data='', norm_flg=True, train_val_test_percent=[0.7 * 0.9, 0.7 * 0.1, 0.3]):
#     """
#     Case1:
#         sess_normal_0 + sess_TDL4_HTTP_Requests_0
#     Case2:
#         sess_normal_0  + sess_Rcv_Wnd_Size_0_0
#
#     Case1 and Case 2:
#         Train set : (0.7 * all_normal_data)*0.9
#         Val_set: (0.7*all_normal_data)*0.1 + 0.1*all_abnormal_data
#         Test_set: 0.3*all_normal_data+ 0.9*all_abnormal_data
#
#     :param input_data:
#     :param norm_flg: default True
#     :param train_val_test_percent: train_set = 0.7*0.9, val_set = 0.7*0.1, test_set = 0.3
#     :return:
#     """
#     if 'mnist' in input_data:
#         from utils.mnist_dataloader import MNIST_DataLoader
#         # load input_data with input_data loader
#         dataset = MNIST_DataLoader(ad_experiment=1)
#         train_set, val_set, test_set = (dataset._X_train, dataset._y_train), (dataset._X_val, dataset._y_val), (
#             dataset._X_test, dataset._y_test)
#     elif 'csv' in input_data:
#         # train_set, val_set, test_set = csv_dataloader(input_data,norm_flg)
#         (X, y) = csv_dataloader(input_data)
#         if norm_flg:
#             X = normalizate_data(X)
#
#         lab = Counter(y)
#         len_normal, len_abnormal = lab[0], lab[1]
#         # X_normal=[]
#         # y_normal=[]
#         # X_abnormal=[]
#         # y_abnormal=[]
#         X_train = []
#         y_train = []
#         X_val = []
#         y_val = []
#         X_test = []
#         y_test = []
#         train_set_size = 0
#         val_set_size = 0
#         test_set_size = 0
#         for i in range(len(y)):
#             if y[i] == 1:
#                 # X_abnormal.append(X[i])
#                 # y_abnormal.append(y[i])
#                 if test_set_size < int(len_abnormal * 0.9):
#                     X_test.append(X[i])
#                     y_test.append(y[i])
#                     test_set_size += 1
#                 else:
#                     X_val.append(X[i])
#                     y_val.append(y[i])
#             elif y[i] == 0:
#                 # X_normal.append(X[i])
#                 # y_normal.append(y[i])
#                 if train_set_size < int(len_normal * 0.7 * 0.9):
#                     X_train.append(X[i])
#                     y_train.append(y[i])
#                     train_set_size += 1
#                 elif val_set_size < int(len_normal * 0.7 * 0.1):
#                     X_val.append(X[i])
#                     y_val.append(y[i])
#                     val_set_size += 1
#                 else:
#                     X_test.append(X[i])
#                     y_test.append(y[i])
#             else:
#                 pass
#         X_train = np.asarray(X_train, dtype=float)
#         y_train = np.asarray(y_train, dtype=int)
#         X_val = np.asarray(X_val, dtype=float)
#         y_val = np.asarray(y_val, dtype=int)
#         X_test = np.asarray(X_test, dtype=float)
#         y_test = np.asarray(y_test, dtype=int)
#         #
#         # len_train_set = int(len(y_normal)*0.7)
#         # len_val_set = int(len_train_set * 0.1)
#         # X_train = np.asarray(X_normal[:len_train_set-len_val_set],dtype=float)
#         # y_train = np.asarray(y_normal[:len_train_set-len_val_set], dtype = int)
#         # len_test_set = int(len(y_abnormal)*0.9)
#         # X_test = np.asarray(X_abnormal[:len_test_set].extend(X_normal[len_train_set:]), dtype=float)
#         # y_test = np.asarray(y_abnormal[:len_test_set].extend(y_normal[len_train_set:]),dtype=int)
#         # X_val = np.asarray(X_normal[len_train_set-len_val_set:len_train_set].extend(X_abnormal[len_test_set:]),dtype=float)
#         # y_val = np.asarray(y_normal[len_train_set-len_val_set:len_train_set].extend(y_abnormal[len_test_set:]), dtype=int)
#
#         # X_train, X_test, y_train, y_test = train_test_split(X_normal, y_normal, test_size=train_val_test_percent[-1], random_state=1)
#         # train_set = (X_train,y_train)
#         # X_val, X_test, y_val, y_test = train_test_split(X_abnormal, y_abnormal, test_size=0.9,random_state=1)
#         # test_set=(test_set+)
#
#         train_set, val_set, test_set = (X_train, y_train), (X_val, y_val), (X_test, y_test)
#
#     else:
#         print('error dataset.')
#         return -1
#
#     # if norm_flg:
#     #     train_set = (normalizate_data(train_set[0]),train_set[1]) # X, y
#     #     val_set=(normalizate_data(val_set[0]),val_set[1])
#     #     test_set=(normalizate_data(test_set[0]),test_set[1])
#
#     return train_set, val_set, test_set
#
#
# def split_normal2train_val_test_from_files(file_list, norm_flg=True,
#                                            train_val_test_percent=[0.7, 0.1, 0.2], shuffle_flg=False):
#     """
#         only split normal_data to train, val, test (only includes 0.2*normal and not normlized in this function, )
#     :param files_dict:  # 0 is normal, 1 is abnormal
#     :param norm_flg:
#     :param train_val_test_percent: train_set=0.7*normal, val_set = 0.1*normal, test_set = 0.2*normal,
#     :return:
#     """
#     np.set_printoptions(suppress=True)  # Suppresses the use of scientific notation for small numbers in numpy array
#     print('file_list:', file_list)
#     X_normal = []
#     y_normal = []
#     for normal_file in file_list:
#         X_tmp, y_tmp = open_file(normal_file, label='0')
#         X_tmp_new = []
#         y_tmp_new = []
#         cnt = 0
#         i = 0
#         for x in X_tmp:
#             if x[-3] == '0' or x[0] == '17':  # is_new: 1 or prtl = 17
#                 X_tmp_new.append(x)
#                 # y_tmp_new.append(y)
#                 cnt += 1
#                 print('i = %d, x=%s' % (i, ','.join(x)))
#             i += 1
#         print('is_new ==0,the cnt data is %d' % (cnt))
#         X_normal.extend(X_tmp)
#         y_normal.extend(y_tmp)
#     X_normal = np.asarray(X_normal, dtype=np.float64)
#     y_normal = np.asarray(y_normal, dtype=int)
#     print('normal_data:', X_normal.shape)
#
#     if shuffle_flg:
#         print('not implement yet.')
#     else:
#         if norm_flg:
#             # train set only includes 0.7*normal_data
#             train_set_len = int(len(y_normal) * train_val_test_percent[0])
#             X_train_normal = X_normal[:train_set_len, :]
#             u_normal = np.mean(X_train_normal, axis=0)
#             std_normal = np.std(X_train_normal, axis=0)
#             print('u_normal:', u_normal)
#             print('std_normal:', std_normal)
#             for i in range(std_normal.shape[0]):
#                 if std_normal[i] == 0:
#                     std_normal[i] += 10e-4
#             print('std_normal_modified:', std_normal)
#             X_train_normal = (X_train_normal - u_normal) / std_normal
#             y_train_normal = y_normal[:train_set_len]
#             train_set = (X_train_normal, y_train_normal)
#
#             # val set only includes 0.1* normal_data
#             val_set_len = int(len(y_normal) * train_val_test_percent[1])
#             X_val_normal = (X_normal[train_set_len:train_set_len + val_set_len, :] - u_normal) / std_normal
#             val_set = (X_val_normal, y_normal[train_set_len:train_set_len + val_set_len])
#
#             X_normal_test = X_normal[train_set_len + val_set_len:, :]
#             y_normal_test = y_normal[train_set_len + val_set_len:]
#             test_normal_set = (X_normal_test, y_normal_test)
#
#             cnt = 0
#             i = 0
#             for x in X_normal_test:
#                 if x[-3] == 0.0 or x[0] == 17.0:  # is_new: 1 or prtl = 17
#                     # X_tmp_new.append(x)
#                     # y_tmp_new.append(y)
#                     cnt += 1
#                     # print('i = %d, x=%s' % (i, ','.join(x)))
#                 i += 1
#             print('normal_test is_new ==0 and prtl == 17,the cnt data is %d' % (cnt))
#
#     return train_set, val_set, test_normal_set, u_normal, std_normal
#
# def achieve_train_val_test_from_files(files_dict={'normal_files': [], 'attack_files': []}, norm_flg=True,
#                                       train_val_test_percent=[0.7, 0.1, 0.2], shuffle_flg=False):
#     """
#
#     :param files_dict:  # 0 is normal, 1 is abnormal
#     :param norm_flg:
#     :param train_val_test_percent: train_set=0.7*normal, val_set = 0.1*normal test_set = (0.2*normal +1*abnormal),
#     :return:
#     """
#     X_normal = []
#     y_normal = []
#     for normal_file in files_dict['normal_files']:
#         X_tmp, y_tmp = open_file(normal_file, label='0')
#         X_normal.extend(X_tmp)
#         y_normal.extend(y_tmp)
#     X_attack = []
#     y_attack = []
#     for attack_file in files_dict['attack_files']:
#         X_tmp, y_tmp = open_file(attack_file, label='1')
#         X_attack.extend(X_tmp)
#         y_attack.extend(y_tmp)
#
#     print('normal_data:', len(X_normal), ', attack_data:', len(X_attack))
#     X_normal = np.asarray(X_normal, dtype=float)
#     y_normal = np.asarray(y_normal, dtype=int)
#     if shuffle_flg:
#         print('not implement yet.')
#     else:
#         if norm_flg:
#             # train set only includes 0.7*normal_data
#             train_set_len = int(len(y_normal) * train_val_test_percent[0])
#             X_train_normal = X_normal[:train_set_len, :]
#             u_normal = np.mean(X_train_normal, axis=0)
#             std_normal = np.std(X_train_normal, axis=0)
#             print('u_normal:', u_normal)
#             print('std_normal:', std_normal)
#             for i in range(std_normal.shape[0]):
#                 if std_normal[i] == 0:
#                     std_normal[i] += 10e-4
#             print('std_normal_modified:', std_normal)
#             X_train_normal = (X_train_normal - u_normal) / std_normal
#             y_train_normal = y_normal[:train_set_len]
#             train_set = (X_train_normal, y_train_normal)
#
#             # val set only includes 0.1* normal_data
#             val_set_len = int(len(y_normal) * train_val_test_percent[1])
#             X_val_normal = (X_normal[train_set_len:train_set_len + val_set_len, :] - u_normal) / std_normal
#             val_set = (X_val_normal, y_normal[train_set_len:train_set_len + val_set_len])
#
#             # test set includes (0.2*normal_data + 1*abnormal_data)
#             # test_set_len = len(y_normal)-train_set_len-val_set_len
#             X_test_normal = X_normal[train_set_len + val_set_len:, :]
#             X_test = np.concatenate((X_test_normal, np.asarray(X_attack, dtype=float)), axis=0)
#             y_test = np.concatenate((np.reshape(y_normal[train_set_len + val_set_len:], (-1, 1)),
#                                      np.reshape(np.asarray(y_attack, dtype=int), (len(y_attack), 1))))
#             test_set_original = (X_test, y_test.flatten())
#             X_test = (X_test - u_normal) / std_normal
#             test_set = (X_test, y_test.flatten())
#
#     return train_set, val_set, test_set, u_normal, std_normal, test_set_original
#

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

    return out_file


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


def show_data_2(data1, data2, x_label='epochs', y_label='mean loss', title=''):
    plt.figure()
    plt.plot(data1, 'r', alpha=0.5, label='train_loss in each epoch')
    plt.plot(data2, 'b', alpha=0.5, label='val loss in each epoch')
    # plt.plot(new_decision_data[:, 2], 'g', alpha=0.5, label='D_G_fake')
    plt.legend(loc='upper right')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


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
