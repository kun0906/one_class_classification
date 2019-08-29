# -*- coding: utf-8 -*-
"""
    useful tools

    several standard data normalization techniques such as min-max, softmax, z-score, decimal scaling, box-cox and etc
"""
import os


def normalize_all_data(datasets_dict, train_set_name='SYNT_train_set', params_dict={}):
    """
     :param selected_features_lst: if -1, use all features; otherwise, selected_features_lst = [5, 8, 10]
    :param norm_method: 'z-score' or 'min-max'
    :param not_normalized_features_lst: if [], normalized all features; otherwise, not_normalized_features_lst = [1, 3, 5]
    :return:
    """

    train_set_dict = datasets_dict['train_set_dict']
    val_set_dict = datasets_dict['val_set_dict']
    test_set_dict = datasets_dict['test_set_dict']

    new_train_set_dict = OrderedDict()
    new_val_set_dict = OrderedDict()
    new_test_set_dict = OrderedDict()

    norm_flg = params_dict['norm_flg']
    norm_method = params_dict['norm_method']
    not_normalized_features_lst = params_dict['not_normalized_features_lst']
    if norm_flg:
        print(f'normalized data: {norm_flg}, norm_method: {norm_method}')
        if norm_method == 'z-score':
            key = train_set_name
            x_train = train_set_dict[key]['X']
            print(f'-- obtain mu and std from \'{key}\'')
            new_x_train, x_train_mu, x_train_std = normalize_data_with_z_score(x_train, mu='', d_std='',
                                                                               not_normalized_features_lst=not_normalized_features_lst)
            new_train_set_dict.update({key: {}})
            new_train_set_dict[key].update({'X': new_x_train})
            new_train_set_dict[key]['y'] = train_set_dict[key]['y']

            min_val = np.min(new_x_train, axis=0)  # X
            max_val = np.max(new_x_train, axis=0)
            range_val = (max_val - min_val)
            print(f'after z-score, range_val:{range_val}')

            for key, value_dict in val_set_dict.items():
                print(f'--normalize {key} with {norm_method}')
                x_val = value_dict['X']
                new_x_val, _, _ = normalize_data_with_z_score(x_val, mu=x_train_mu, d_std=x_train_std,
                                                              not_normalized_features_lst=not_normalized_features_lst)
                new_val_set_dict[key] = {}
                new_val_set_dict[key]['X'] = new_x_val
                new_val_set_dict[key]['y'] = value_dict['y']

                min_val = np.min(new_x_val, axis=0)  # X
                max_val = np.max(new_x_val, axis=0)
                range_val = (max_val - min_val)
                print(f'after z-score, range_val:{range_val}')

            for key, value_dict in test_set_dict.items():
                print(f'--normalize {key} with {norm_method}')
                x_test = value_dict['X']
                new_x_test, _, _ = normalize_data_with_z_score(x_test, mu=x_train_mu, d_std=x_train_std,
                                                               not_normalized_features_lst=not_normalized_features_lst)
                new_test_set_dict[key] = {}
                new_test_set_dict[key]['X'] = new_x_test
                new_test_set_dict[key]['y'] = value_dict['y']

                min_val = np.min(new_x_test, axis=0)  # X
                max_val = np.max(new_x_test, axis=0)
                range_val = (max_val - min_val)
                print(f'after z-score, range_val:{range_val}')

        elif norm_method == 'min-max':
            key = train_set_name
            print(f'-- obtain min and max from \'{key}\'')
            x_train = train_set_dict[key]['X']
            new_x_train, x_train_min, x_train_max = normalize_data_with_min_max(x_train, min_val='',
                                                                                max_val='',
                                                                                not_normalized_features_lst=not_normalized_features_lst)
            new_train_set_dict[key] = {}
            new_train_set_dict[key]['X'] = new_x_train
            new_train_set_dict[key]['y'] = train_set_dict[key]['y']

            min_val = np.min(new_x_train, axis=0)  # X
            max_val = np.max(new_x_train, axis=0)
            range_val = (max_val - min_val)
            print(f'after z-score, range_val:{range_val}')

            for key, value_dict in val_set_dict.items():
                print(f'--normalize {key} with {norm_method}')
                x_val = value_dict['X']
                new_x_val, _, _ = normalize_data_with_min_max(x_val, min_val=x_train_min,
                                                              max_val=x_train_max,
                                                              not_normalized_features_lst=not_normalized_features_lst)
                new_val_set_dict[key] = {}
                new_val_set_dict[key]['X'] = new_x_val
                new_val_set_dict[key]['y'] = value_dict['y']

                min_val = np.min(new_x_val, axis=0)  # X
                max_val = np.max(new_x_val, axis=0)
                range_val = (max_val - min_val)
                print(f'after z-score, range_val:{range_val}')

            for key, value_dict in test_set_dict.items():
                print(f'--normalize {key} with {norm_method}')
                x_test = value_dict['X']
                new_x_test, _, _ = normalize_data_with_min_max(x_test, min_val=x_train_min,
                                                               max_val=x_train_max,
                                                               not_normalized_features_lst=not_normalized_features_lst)
                new_test_set_dict[key] = {}
                new_test_set_dict[key]['X'] = new_x_test
                new_test_set_dict[key]['y'] = value_dict['y']

                min_val = np.min(new_x_test, axis=0)  # X
                max_val = np.max(new_x_test, axis=0)
                range_val = (max_val - min_val)
                print(f'after z-score, range_val:{range_val}')
        else:
            print(f'norm_method {norm_method} is not correct.')
            return -1

        return {'train_set_dict': new_train_set_dict, 'val_set_dict': new_val_set_dict,
                'test_set_dict': new_test_set_dict}


def normalize_data_with_min_max(data_arr, eplison=10e-4, min_val=[], max_val=[], not_normalized_features_lst=[]):
    """

    :param np_arr:
    :param eplison: handle with 0.
    :return:
    """
    if len(min_val) == 0 or len(max_val) == 0:
        print(f'len(min_val) = {len(min_val)}, len(max_val) = {len(max_val)}')
        min_val = np.min(data_arr, axis=0)
        max_val = np.max(data_arr, axis=0)

    range_val = (max_val - min_val)

    norm_data = []
    for i in range(data_arr.shape[1]):
        val = data_arr[:, i]
        if i not in not_normalized_features_lst:
            if range_val[i] == 0.0:
                print(f'the range of the {i}-th feature is 0.')
                range_val[i] += eplison
            val = (val - min_val[i]) / range_val[i]
        norm_data.append(val)

    norm_data_arr = np.asarray(norm_data, dtype=float).transpose()

    before_range = (np.max(data_arr, axis=0) - np.min(data_arr, axis=0))
    value = list(map('{:.0f}'.format, before_range))  # limit the float to int print.
    value = [float(v) for v in value]
    print(f'before normalization, range_val: {value}')  # f'{value:{width}.{precision}}'

    if len(not_normalized_features_lst) == 0:
        print(f'normalize all features. (because not_normalized_features_lst is {not_normalized_features_lst})')
    else:
        print(f'normalize all features except for {not_normalized_features_lst}')

    after_range = (np.max(norm_data_arr, axis=0) - np.min(norm_data_arr, axis=0))
    value = list(map('{:.0f}'.format, after_range))  # limit the float to int print.
    value = [float(v) for v in value]
    print(f'after normalization,  range_val: {value}')  # f'{value:{width}.{precision}}'

    return norm_data_arr, min_val, max_val


def normalize_data_with_z_score(data_arr, mu=[], d_std=[], eplison=10e-4, not_normalized_features_lst=[]):
    if len(mu) == 0 or len(d_std) == 0:
        print(f'len(mu) = {len(mu)}, len(d_std) = {len(d_std)}')
        mu = data_arr.mean(axis=0)
        d_std = data_arr.std(axis=0)

    norm_data = []
    for i in range(data_arr.shape[1]):
        val = data_arr[:, i]
        if i not in not_normalized_features_lst:
            if d_std[i] == 0:
                print(f'the range of the {i}-th feature is 0.')
                d_std[i] = d_std[i] + eplison
            val = (val - mu[i]) / d_std[i]
        else:
            continue

        norm_data.append(val)

    norm_data_arr = np.asarray(norm_data, dtype=np.float64).transpose()

    before_range = (np.max(data_arr, axis=0) - np.min(data_arr, axis=0))
    value = list(map('{:.0f}'.format, before_range))  # limit the float to int print.
    value = [float(v) for v in value]
    print(f'before normalization, range_val: {value}')  # f'{value:{width}.{precision}}'

    after_range = (np.max(norm_data_arr, axis=0) - np.min(norm_data_arr, axis=0))
    value = list(map('{:.0f}'.format, after_range))  # limit the float to int print.
    value = [float(v) for v in value]
    print(f'after normalization,  range_val: {value}')  # f'{value:{width}.{precision}}'

    return norm_data_arr, mu, d_std


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


from collections import OrderedDict

from sklearn.metrics import confusion_matrix

# from utils.load_data import z_score_np
import numpy as np


def z_score_np_1(data_np, mu=[], d_std=[]):
    # def min_max_np(data_np, d_min=[], d_max=[]):
    d_min = mu
    d_max = d_std

    if len(d_min) == 0 or len(d_max) == 0:
        print(f'len(d_min) = {len(d_min)}, len(d_max) = {len(d_max)}')
        d_min = data_np.min(axis=0)
        d_max = data_np.max(axis=0)

    diff = d_max - d_min
    # avoid diff equals 0.
    for idx in range(diff.shape[0]):
        if diff[idx] == 0:
            print(f'the range of the {idx}-th feature is 0.')
            diff[idx] = diff[idx] + np.math.exp(-8)

    data_np = (data_np - d_min) / diff * (1 - (-1)) + (-1)  # normalize the result to [-1,1]

    return data_np, d_min, d_max


def calucalate_metrics(y_true='', y_pred=''):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    conf = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print('cm:', conf)
    if len(conf) == 1:
        print(f'tpr (recall): {-1}, fnr: {-1}, fpr: {-1}, tnr: {-1}, acc: {-1}')
        # return recall, fnr, fpr, tnr, acc
        return -1, -1, -1, -1, -1

    if (conf[0, 0] + conf[1, 0]) != 0:
        pr = conf[0, 0] / (conf[0, 0] + conf[1, 0])
    else:
        pr = 0

    if (conf[0, 0] + conf[0, 1]) != 0:
        recall = conf[0, 0] / (conf[0, 0] + conf[0, 1])
    else:
        recall = 0

    if (pr + recall) != 0:
        f1 = (2 * pr * recall) / (pr + recall)
    else:
        f1 = 0

    # print(classification_report(data_test_labels,pred_class))
    # try:
    #     print("F1 Score : ", f1)
    #     fpr = (conf[1, 0] / (conf[1, 0] + conf[1, 1]))
    #     print("FPR : ", (conf[1, 0] / (conf[1, 0] + conf[1, 1])))
    #     print("DR : ", recall)
    #     print(conf)
    # except:
    #     fpr = 0
    #     print("FPR = 100")

    if (conf[0, 0] + conf[0, 1]) != 0:
        fnr = conf[0, 1] / (conf[0, 0] + conf[0, 1])
    else:
        fnr = 0
    if (conf[1, 0] + conf[1, 1]) != 0:
        fpr = conf[1, 0] / (conf[1, 0] + conf[1, 1])
    else:
        fpr = 0

    if (conf[1, 0] + conf[1, 1]) != 0:
        tnr = (conf[1, 1]) / (conf[1, 0] + conf[1, 1])
    else:
        tnr = 0

    if (conf[0, 0] + conf[0, 1] + conf[1, 0] + conf[1, 1]) != 0:
        acc = (conf[0, 0] + conf[1, 1]) / (conf[0, 0] + conf[0, 1] + conf[1, 0] + conf[1, 1])
    else:
        acc = 0
    dr = recall

    print(f'tpr (recall): {recall}, fnr: {fnr}, fpr: {fpr}, tnr: {tnr}, acc: {acc}')

    return recall, fnr, fpr, tnr, acc


# Calculate f1_score
def f1_score(conf):
    if (conf[0, 0] + conf[1, 0]) != 0:
        pr = conf[0, 0] / (conf[0, 0] + conf[1, 0])
    else:
        pr = 0

    if (conf[0, 0] + conf[0, 1]) != 0:
        recall = conf[0, 0] / (conf[0, 0] + conf[0, 1])
    else:
        recall = 0

    if (pr + recall) != 0:
        f1 = (2 * pr * recall) / (pr + recall)
    else:
        f1 = 0

    return f1


def get_optimal_thres(training_losses, factor_AE_thres=2, key='loss'):
    """
    :param training_losses:
    :param factor_AE_thres:
    :param key: loss or val_loss
    :return:
    """

    if len(training_losses[key]) >= 4:
        s_thres = 0.0
        for v in training_losses[key][-4:]:
            s_thres += v
        optimal_thres_AE = s_thres / 4 * factor_AE_thres
    else:
        optimal_thres_AE = training_losses[key][-1] * factor_AE_thres

    return optimal_thres_AE, factor_AE_thres, key


def get_max_reconstruction_error(reconstr_errs_lst):
    max_val = -100000000
    min_val = 100000000
    for value in reconstr_errs_lst:
        # [dist, y]= value
        if value[0] >= max_val:
            max_val = value[0]
        if value[0] <= min_val:
            min_val = value[0]

    return max_val, min_val


def normalize_for_ml(UNB_train_set, mu='', d_std='', flg='norm_for_unsupervised_ml', test_flg=False):
    if flg == 'norm_for_unsupervised_ml':
        x_norm_train, y_norm_train, x_attack_train, y_attack_train = UNB_train_set
        x_norm_train, x_norm_train_mu, x_norm_train_std = z_score_np(x_norm_train, mu=mu, d_std=d_std)

        if len(mu) != 0 and test_flg == True:
            x_attack_train, _, _ = z_score_np(x_attack_train, mu=mu, d_std=d_std)

        norm_UNB_train_set = (x_norm_train, y_norm_train, x_attack_train, y_attack_train)

        return norm_UNB_train_set, x_norm_train_mu, x_norm_train_std

    if flg == 'norm_for_supervised_ml':
        x_norm_train, y_norm_train, x_attack_train, y_attack_train = UNB_train_set
        x_train = np.concatenate([x_norm_train, x_attack_train])
        y_train = np.concatenate([y_norm_train, y_attack_train])
        x_train, x_train_mu, x_train_std = z_score_np(x_train, mu=mu, d_std=d_std)
        x_norm_train = x_train[:x_norm_train.shape[0]]
        y_norm_train = y_train[:y_norm_train.shape[0]]
        x_attack_train = x_train[x_norm_train.shape[0]:]
        y_attack_train = y_train[y_norm_train.shape[0]:]

        return (x_norm_train, y_norm_train, x_attack_train, y_attack_train), x_train_mu, x_train_std


def lst_to_str(data):
    lst_str = '['
    for v in data:
        lst_str += str(v) + ','

    lst_str += ']'

    return lst_str


def save_features_selection_results(out_file, features_descended_dict):
    with open(out_file, 'w') as hdl:
        for key, value in features_descended_dict.items():
            line = str(key) + ':' + str(len(value)) + ':' + lst_to_str(value)
            hdl.write(line + '\n')

    return out_file


def save_reconstruction_errors(reconstruction_errors_lst, case, feat_size, out_file='figures/recon_err.txt'):
    output_dir = os.path.dirname(out_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(out_file, 'w') as out_hdl:
        out_hdl.write('reconstruction_error, label')
        for value in reconstruction_errors_lst:
            out_hdl.write(str(value[0]) + ',' + str(value[1]) + '\n')

    return out_file


def save_sub_features_res_metrics(out_file, all_res_metrics_lst, optimal_thres):
    with open(out_file, 'w') as out_hdl:
        for idx in range(len(all_res_metrics_lst)):
            sub_features_list, res_metrics_feat_dict, optimal_thres_AE = all_res_metrics_lst[idx]
            line = f'thres_AE, tpr, fnr, fpr, tnr, acc, num_features={len(sub_features_list)}, sub_features are {sub_features_list}'
            out_hdl.write(line + '\n')
            for i in range(len(res_metrics_feat_dict['acc'])):
                line = str(optimal_thres) + ',' + str(res_metrics_feat_dict['tpr'][i]) + ',' + str(
                    res_metrics_feat_dict['fnr'][i]) + ',' + str(
                    res_metrics_feat_dict['fpr'][i]) + ',' + str(res_metrics_feat_dict['tnr'][i]) + ',' + str(
                    res_metrics_feat_dict['acc'][i])
                out_hdl.write(line + '\n')


def save_thres_res_metrics(out_file, thres_lst, res_metrics_lst):
    with open(out_file, 'w') as out_hdl:
        line = 'thres_AE, tpr, fnr, fpr, tnr, acc'
        out_hdl.write(line + '\n')
        for i, thres in enumerate(thres_lst):
            line = str(thres) + ',' + str(res_metrics_lst['tpr'][i]) + ',' + str(res_metrics_lst['fnr'][i]) + ',' + str(
                res_metrics_lst['fpr'][i]) + ',' + str(res_metrics_lst['tnr'][i]) + ',' + str(res_metrics_lst['acc'][i])
            out_hdl.write(line + '\n')


def save_roc_to_txt(out_file, all_y_test_pred_labels_dict):
    """

    :param out_file:
    :param all_y_test_pred_labels_dict:
        all_y_test_pred_proba_dict = {'test_set':{'AE':(y_true, y_pred_proba)}, 'test_set_2': {'AE':(,)}, {..}}
    :return:
    """
    output_dir = os.path.dirname(out_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # print(f'y_test_pred_labels_dict:{y_test_pred_labels_dict}')
    with open(out_file, 'w') as out_hdl:
        line = 'model-y_true[0]:y_preds_probs[0], y_true[1]:y_preds_probs[1],....'
        out_hdl.write(line + '\n')
        for key_test_set, value_dict in all_y_test_pred_labels_dict.items():

            for idx, (key_algorithm, value) in enumerate(value_dict.items()):
                line = key_test_set + '=>' + str(key_algorithm) + '@'
                if len(value) == 0:
                    print(f'key:{key}, value:{value}')
                    continue
                y_true, y_preds_labels, y_preds_proba = value
                y_true = np.reshape(y_true, (y_true.shape[0],))
                for i in range(y_true.shape[0] - 1):
                    line += str(y_true[i]) + ':' + str(y_preds_labels[i]) + ','
                line += str(y_true[-1]) + ':' + str(y_preds_labels[i])
                out_hdl.write(line + '\n')
