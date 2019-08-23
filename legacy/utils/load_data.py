import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from utils.read_data_un import Dataset
from utils.utils import normalise_data, zscore


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


def load_data_from_txt(input_file, start=0, end=77989):
    data = []
    cnt = 0
    with open(input_file, 'r') as hdl:
        line = hdl.readline()
        while line != '' and cnt < end:
            if line.startswith('ts'):
                line = hdl.readline()
                continue
            if cnt >= start:
                data.append(line.split(',')[5:])  # without : "ts, sip, dip, sport, dport"
            line = hdl.readline()
            cnt += 1

    return np.asarray(data, dtype=float)


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

    if case[3] == '1':  # Experiment 1
        if case[5] == '1':  # training and testing on SYNT (simulated data)
            ### shuffle data
            ### training on SYNT
            # x_norm = dataset(filename=["dataset/synthetic_dataset/Sess_normal_0.txt"], label=1, end=77989)
            x_norm = load_data_from_txt(input_file="dataset/synthetic_dataset/Sess_normal_0.txt", end=77989)
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
            # x_attack_13 = dataset(filename=["dataset/unb/UNB_DosGoldenEye_UNB_IDS2017.txt"], label=0, end=1268,
            #                       mean=x_norm.mean, std=x_norm.std).data
            # x_attack_14 = dataset(filename=["dataset/unb/UNB_DoSSlowloris_UNB_IDS2017.txt"], label=0, end=16741,
            #                       mean=x_norm.mean, std=x_norm.std).data

            x_attack_11 = load_data_from_txt(input_file="dataset/unb/DoSHulk_UNB.txt", end=11530)
            x_attack_12 = load_data_from_txt(input_file="dataset/unb/DOSSlowHttpTest_UNB.txt", end=6414)
            x_attack_13 = load_data_from_txt(input_file="dataset/unb/UNB_DosGoldenEye_UNB_IDS2017.txt", end=1268)
            x_attack_14 = load_data_from_txt(input_file="dataset/unb/UNB_DoSSlowloris_UNB_IDS2017.txt", end=16741)
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
