import os
from collections import OrderedDict

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from plot_all_results import plot_sub_features_metrics
from utils.load_data import load_data, load_data_from_txt
from utils.metrics_measure import z_score_np, save_sub_features_res_metrics
from utils.read_data_un import Dataset
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectFromModel
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# import statsmodels.api as sm


def select_sub_features_data(x_norm_train, sub_features_list):
    for idx, new_feature in enumerate(sub_features_list):
        if idx == 0:
            x_norm_train_sub = x_norm_train[:, new_feature]
            x_norm_train_sub = np.reshape(x_norm_train_sub, (x_norm_train_sub.shape[0], 1))
            continue
        else:
            x_norm_train_sub = np.concatenate(
                (x_norm_train_sub, np.reshape(x_norm_train[:, new_feature], (x_norm_train.shape[0], 1))), axis=1)

    return x_norm_train_sub


def feature_selection_with_sklearn(X, var=0.8, num_features=2):
    # clf = Pipeline([
    #     ('feature_selection', SelectFromModel(DecisionTreeClassifier())),
    #     ('classification', DecisionTreeClassifier())
    # ])
    # clf.fit(X, y)
    # print(clf.named_steps['classification'].feature_importances_)
    # clf.feature_importances_.index()
    X_new = SelectKBest(chi2, k=2)
    y = np.ones((X.shape[0],))
    X_new.fit(X, y)

    sel = VarianceThreshold(threshold=var)
    new_X = sel.fit_transform(X)

    features_descended_lst = {}
    for idx, v in enumerate(sel.variances_):
        features_descended_lst[idx] = v

    features_descended_lst = dict(sorted(features_descended_lst.items(), key=lambda x: x[1], reverse=True))

    sub_features_list = select_top_k_features(features_descended_lst, num_features=num_features)

    return sub_features_list


def select_top_k_features(feature_descended_dict, num_features=2):
    sub_features_list = []
    for i, (key, value) in enumerate(feature_descended_dict.items()):
        if i < num_features:
            sub_features_list.append(key)

    return sorted(sub_features_list)


def feature_selection_1(x_norm_train, num_features=2, corr_thres=0.5, show_flg=True):
    data_df = pd.DataFrame(data=x_norm_train)
    corr = data_df.corr(method='pearson')
    if show_flg:
        # Using Pearson Correlation
        plt.figure(figsize=(12, 10))
        # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        sns.heatmap(corr, annot=True, cmap=plt.cm.Blues)
        plt.show()

    features_descended_dict = {}
    cnt = 0
    try_times = 100
    while corr_thres >= 0.1 and corr_thres < 1.0:
        print(f'current corr_thres:{corr_thres}')
        # print(f'{corr.values[0,:]}')
        corr_tmp = corr
        columns = np.full((corr.shape[0],), True, dtype=bool)  # False means that this feature won't be selected.
        for i in range(corr_tmp.shape[0]):
            # if columns[i] == False:
            #     continue
            for j in range(i + 1, corr_tmp.shape[0]):
                if abs(corr_tmp.iloc[i, j]) > corr_thres or abs(corr_tmp.iloc[i, j]) == np.nan:
                    if columns[j]:
                        columns[j] = False

        for i, keep in enumerate(columns):
            if keep == True:
                features_descended_dict[i] = 1

        # features_descended_dict = dict(sorted(features_descended_dict.items(), key=lambda x: x[1], reverse=True))
        # print(f'{len(features_descended_dict.items())}, {features_descended_dict.items()}')

        if len(features_descended_dict.keys()) > num_features:
            sub_feature_lst = []
            for key, value in features_descended_dict.items():
                sub_feature_lst.append(key)

            print(f'1-current corr_thres:{corr_thres}')

            return sorted(sub_feature_lst[:num_features], reverse=False)

        elif len(features_descended_dict.keys()) < num_features:
            corr_thres -= 0.02
            cnt += 1
        else:
            print(f'2-current corr_thres:{corr_thres}')

            return sorted(features_descended_dict.keys(), reverse=False)

        if cnt > try_times:
            print(f'try more than {try_times} times.')
            break

    print(f'3-current corr_thres:{corr_thres}')
    return sorted(features_descended_dict.keys(), reverse=False)


def get_min_max_in_matrix(data_n_n, max_val=0):
    rows, cols = data_n_n.shape[0], data_n_n.shape[1]

    min_val = 10
    for i in range(rows):
        for j in range(i + 1, cols):
            if np.isnan(data_n_n[i][j]):
                continue
            if abs(data_n_n[i][j]) >= max_val:
                max_val = abs(data_n_n[i][j])
                # data_n_n[i][j] = np.nan
            if abs(data_n_n[i][j]) <= min_val:
                min_val = abs(data_n_n[i][j])
                # data_n_n[i][j] ='-'

    return data_n_n, min_val, max_val


def feature_selection_new(x_norm_train, num_features=2, corr_thres=0.01, show_flg=True):
    data_df = pd.DataFrame(data=x_norm_train)
    corr = data_df.corr(method='pearson')
    if show_flg:
        # Using Pearson Correlation
        plt.figure(figsize=(12, 10))
        # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        sns.heatmap(corr, annot=True, cmap=plt.cm.Blues)
        plt.show()

    features_descended_dict = OrderedDict()

    all_features = [i for i in range(x_norm_train.shape[1])]
    pre_key = 1.01  # the start value
    if len(features_descended_dict) == 0:
        features_descended_dict[pre_key] = all_features

    cnt = 0
    data_n_n = corr.values
    columns = np.full((corr.shape[0],), True, dtype=bool)
    num_feat = data_n_n.shape[1]
    data_n_n, min_val, max_val = get_min_max_in_matrix(data_n_n, max_val=0)
    while min_val < max_val:  # corr_thres = abs()
        max_val = 0
        rows, cols = data_n_n.shape[0], data_n_n.shape[1]
        for i in range(rows):
            if np.isnan(data_n_n[i][i]):
                columns[i] = False
                continue
            data_n_n, min_val, max_val = get_min_max_in_matrix(data_n_n, max_val)
            for j in range(i + 1, cols):
                if np.isnan(data_n_n[i][j]):  # data_n_n[i][j] == np.nan: float, np.nan: float64.
                    continue
                if abs(data_n_n[i][j]) >= max_val:
                    data_n_n[i][j] = np.nan
                    columns[j] = False

        corr_thres = max_val
        selected_columns = data_df.columns[columns]
        num_feat = len(selected_columns)
        # print('thres= ', corr_thres, ' , num= ', len(selected_columns), ':', selected_columns, ' max_thres:',max_val)

        # if len(features_descended_dict) == 0:
        #     features_descended_dict[corr_thres] = selected_columns.values.tolist()
        #     pre_key = max_val
        # else:
        if features_descended_dict[pre_key] != selected_columns.values.tolist():
            print(corr_thres, selected_columns.values.tolist())
            features_descended_dict[corr_thres] = selected_columns.values.tolist()
            pre_key = corr_thres
        else:
            val_tmp = features_descended_dict[pre_key]
            features_descended_dict.pop(pre_key, None)  # remove pre_key
            print(f'remove key={pre_key}, its value={val_tmp} from the dict')
            features_descended_dict[corr_thres] = val_tmp
            pre_key = corr_thres

        cnt += 1

    features_descended_dict = dict(sorted(features_descended_dict.items(), key=lambda x: len(x[1]), reverse=False))

    print('features_descended_dict')
    for key, value in features_descended_dict.items():
        print(f'corr_thres={key}, num={len(value)}, features={value}')

    return features_descended_dict


def feature_selection(x_norm_train, num_features=2, corr_thres=0.01, show_flg=True):
    data_df = pd.DataFrame(data=x_norm_train)
    corr = data_df.corr(method='pearson')
    if show_flg:
        # Using Pearson Correlation
        plt.figure(figsize=(12, 10))
        # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        sns.heatmap(corr, annot=True, cmap=plt.cm.Blues)
        plt.show()

    features_descended_dict = OrderedDict()
    while corr_thres <= 1.0:
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            if np.allclose(corr[i], np.asarray([np.nan] * corr.shape[1]), equal_nan=True):
                columns[i] = False
                continue
            for j in range(i + 1, corr.shape[0]):
                if abs(corr.iloc[i, j]) >= corr_thres or type(corr.iloc[i, j]) == type(np.NaN):
                    if columns[j]:
                        columns[j] = False

        selected_columns = data_df.columns[columns]
        print('thres= ', corr_thres, ' , num= ', len(selected_columns), ':', selected_columns)
        if len(features_descended_dict) == 0:
            features_descended_dict[corr_thres] = selected_columns.values.tolist()
            pre_key = corr_thres
        else:
            if features_descended_dict[pre_key] != selected_columns.values.tolist():
                print(corr_thres, selected_columns.values.tolist())
                features_descended_dict[corr_thres] = selected_columns.values.tolist()
                pre_key = corr_thres

        corr_thres += 0.05

    all_features = [i for i in range(x_norm_train.shape[1])]
    if features_descended_dict[pre_key] != all_features:
        print(corr_thres, 'all features:', all_features)
        features_descended_dict[corr_thres] = all_features
        # pre_key = corr_thres

    return features_descended_dict


def feature_selection_correction(Epochs=2, optimal_thres_AE=5, find_optimal_thres_flg=False,
                                 cases=['uSc3C2_min_max_20_14'], factor=10, random_state=42, show_flg=True,
                                 balance_train_data_flg=True, title_flg=True, corr_thres=0.5):
    # # data_train = Dataset(
    # #     filename=["Dataset/Simulated_DataSet/Sess_normal_0.txt", "Dataset/Public_Dataset/Log_normal.txt"], label=1,
    # #     end=7000)
    # data_train = Dataset(filename=[dataset], label=1,end=7000)
    # data_test_1 = Dataset(filename=["Dataset/Public_Dataset/Log_normal.txt"], label=1, start=7000, end=12000,
    #                       mean=data_train.mean, std=data_train.std)
    # data_test_2 = Dataset(filename=["Dataset/Public_Dataset/Log_doshulk.txt"], label=0, mean=data_train.mean,
    #                       std=data_train.std, end=1500)
    # data_test_3 = Dataset(filename=["Dataset/Simulated_DataSet/Sess_DDoS_Excessive_GET_POST.txt"], label=0, end=4000,
    #                       mean=data_train.mean, std=data_train.std)
    # data_test_4 = Dataset(filename=['Dataset/Simulated_DataSet/Sess_normal_0.txt'], label=1, start=7000, end=12000,
    #                       mean=data_train.mean, std=data_train.std)

    norm_flg = True
    for case in cases:
        if case[3] == '1':
            if case[5] == '1':
                print('\n@@@train and evaluate AE on SYNT:', case)
            elif case[5] == '2':
                print('\n@@@train and evaluate AE on UNB:', case)
            elif case[5] == '3':
                print('\n@@@train and evaluate AE on MAWI:', case)
            else:
                print('not implement')
                return -1

            print("\nStep 1. loading data...")
            (x_norm_train_raw, y_norm_train), (x_norm_val_raw, y_norm_val), (x_norm_test_raw, y_norm_test), (
                x_attack_1_raw, y_attack_1) = load_data(case, random_state=random_state)
            print("\n+Step 1-1. preprocessing data...")
            if norm_flg:
                x_norm_train_raw, x_norm_train_mu, x_norm_train_std = z_score_np(x_norm_train_raw, mu='', d_std='')
                x_norm_val_raw, _, _ = z_score_np(x_norm_val_raw, mu=x_norm_train_mu, d_std=x_norm_train_std)
                x_norm_test_raw, _, _ = z_score_np(x_norm_test_raw, mu=x_norm_train_mu, d_std=x_norm_train_std)
                if type(x_attack_1_raw) != type(None):
                    x_attack_1_raw, _, _ = z_score_np(x_attack_1_raw, mu=x_norm_train_mu, d_std=x_norm_train_std)

            # data_train = pd.DataFrame(data=x_norm_train_raw)
            # corr = data_train.corr(method='pearson')
            # print(corr)
            #
            # if show_flg:
            #     # Using Pearson Correlation
            #     plt.figure(figsize=(12, 10))
            #     # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
            #     sns.heatmap(corr, annot=True, cmap=plt.cm.Blues)
            #     plt.show()
            #
            #
            # while corr_thres <= 1.0:
            #     columns = np.full((corr.shape[0],), True, dtype=bool)
            #     for i in range(corr.shape[0]):
            #         for j in range(i + 1, corr.shape[0]):
            #             if abs(corr.iloc[i, j]) >= corr_thres:
            #                 if columns[j]:
            #                     columns[j] = False
            #
            #     selected_columns = data_train.columns[columns]
            #     print('thres= ', corr_thres, ' , num= ', len(selected_columns), ':', selected_columns)
            #     corr_thres += 0.05

            feature_selection(x_norm_train_raw, num_features=2, corr_thres=0.01, show_flg=True)

            # if show_flg:
            #     # Using Pearson Correlation
            #     plt.figure(figsize=(12, 10))
            #     cor = data_train.df.corr()
            #     # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
            #     sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
            #     plt.show()

            '''
            0 - 2
            0.1 - 3
            0.15 - 4
            0.2 - 8
            0.3 - 9
            0.35 - 10
            0.45 - 11
            0.5 - 12
            0.6 - 13
            0.7 - 15
            0.75 - 17
            0.8 - 18
            0.85 - 19
            0.95 - 21
            1 - 27
            '''
            # print(np.corrcoef(data_train.data.T))

            # data_test = np.concatenate([data_test_1.data, data_test_4.data, data_test_2.data, data_test_3.data])
            # data_test_labels = np.concatenate(
            #     [data_test_1.data_label, data_test_4.data_label, data_test_2.data_label, data_test_3.data_label])
            #
            # data = np.concatenate([data_train.data, data_test])
            # label = np.concatenate([data_train.data_label, data_test_labels])
            #
            # print(np.corrcoef(data.T))

            # selector = VarianceThreshold(0.5)
            # selector.fit(data_train.data)
            #
            # s2 = SelectKBest(k=15)
            # s2.fit(data, label)

            # print(selector.get_support())
            # print(s2.get_support())

            # print(selector.get_support())

            '''
            for i in range(27):
                print(np.var(data_train.data.T[i]))
            '''


def feature_selection_2(dataset, corr_thres=0.2, show_flg=True):
    # data_train = Dataset(
    #     filename=["Dataset/Simulated_DataSet/Sess_normal_0.txt", "Dataset/Public_Dataset/Log_normal.txt"], label=1,
    #     end=7000)
    data_train = Dataset(filename=[dataset], label=1, end=7000)
    data_test_1 = Dataset(filename=["Dataset/Public_Dataset/Log_normal.txt"], label=1, start=7000, end=12000,
                          mean=data_train.mean, std=data_train.std)
    data_test_2 = Dataset(filename=["Dataset/Public_Dataset/Log_doshulk.txt"], label=0, mean=data_train.mean,
                          std=data_train.std, end=1500)
    data_test_3 = Dataset(filename=["Dataset/Simulated_DataSet/Sess_DDoS_Excessive_GET_POST.txt"], label=0, end=4000,
                          mean=data_train.mean, std=data_train.std)
    data_test_4 = Dataset(filename=['Dataset/Simulated_DataSet/Sess_normal_0.txt'], label=1, start=7000, end=12000,
                          mean=data_train.mean, std=data_train.std)

    corr = data_train.df.corr(method='pearson')
    print(corr)
    while corr_thres <= 0.82:
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if abs(corr.iloc[i, j]) >= corr_thres:
                    if columns[j]:
                        columns[j] = False

        selected_columns = data_train.df.columns[columns]
        print('thres= ', corr_thres, ' , num= ', len(selected_columns), ':', selected_columns)
        corr_thres += 0.05

    if show_flg:
        # Using Pearson Correlation
        plt.figure(figsize=(12, 10))
        cor = data_train.df.corr()
        # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
        plt.show()

        '''
        0 - 2
        0.1 - 3
        0.15 - 4
        0.2 - 8
        0.3 - 9
        0.35 - 10
        0.45 - 11
        0.5 - 12
        0.6 - 13
        0.7 - 15
        0.75 - 17
        0.8 - 18
        0.85 - 19
        0.95 - 21
        1 - 27
        '''
        # print(np.corrcoef(data_train.data.T))

    data_test = np.concatenate([data_test_1.data, data_test_4.data, data_test_2.data, data_test_3.data])
    data_test_labels = np.concatenate(
        [data_test_1.data_label, data_test_4.data_label, data_test_2.data_label, data_test_3.data_label])

    data = np.concatenate([data_train.data, data_test])
    label = np.concatenate([data_train.data_label, data_test_labels])

    # print(np.corrcoef(data.T))

    selector = VarianceThreshold(0.5)
    selector.fit(data_train.data)

    s2 = SelectKBest(k=15)
    s2.fit(data, label)

    # print(selector.get_support())
    # print(s2.get_support())

    # print(selector.get_support())

    '''
    for i in range(27):
        print(np.var(data_train.data.T[i]))
    '''


if __name__ == '__main__':
    dataset = "Dataset/Simulated_DataSet/Sess_normal_0.txt"
    x_norm_train_raw = load_data_from_txt(dataset, start=0, end=77989)
    norm_flg = True
    if norm_flg:
        x_norm_train_raw, x_norm_train_mu, x_norm_train_std = z_score_np(x_norm_train_raw, mu='', d_std='')
    feature_selection_with_sklearn(x_norm_train_raw)

    ### parameters
    Epochs = 1  # when Epochs = 300, the better optimal_thres_AE=0.8~1.2
    find_optimal_thres_flg = False  # false: use the given optimal_thres_AE, otherwise, find the optimal one by using training loss.
    optimal_thres_AE = 1.0  # the val will be changed from train, training_loss*factor: factor =10,
    factor = 15
    cases = ['uSc1C2_z-score']
    feature_selection_correction(Epochs=Epochs, cases=cases, corr_thres=0.01)
