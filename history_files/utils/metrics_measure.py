from collections import OrderedDict

from sklearn.metrics import confusion_matrix

# from utils.load_data import z_score_np
import numpy as np


def z_score_np(data_np, mu=[], d_std=[]):
    if len(mu) == 0 or len(d_std) == 0:
        print(f'len(mu) = {len(mu)}, len(d_std) = {len(d_std)}')
        mu = data_np.mean(axis=0)
        d_std = data_np.std(axis=0)
    # avoid d_std equals 0.
    for idx in range(d_std.shape[0]):
        if d_std[idx] == 0:
            print(f'the variance of the {idx}-th feature is 0.')
            d_std[idx] = d_std[idx] + np.math.exp(-8)

    data_np = (data_np - mu) / d_std

    return data_np, mu, d_std


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


def calucalate_metrics(y_ture, y_preds):
    conf = confusion_matrix(y_ture, y_preds)
    print(conf)
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


def get_optimal_thres(training_losses, factor=2, key='loss'):
    """
    :param training_losses:
    :param factor:
    :param key: loss or val_loss
    :return:
    """

    if len(training_losses[key]) >= 10:
        s_thres = 0.0
        for v in training_losses[key][-10:]:
            s_thres += v
        optimal_thres_AE = s_thres / 10 * factor
    else:
        optimal_thres_AE = training_losses[key][-1] * factor

    return optimal_thres_AE, factor, key


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


def save_reconstruction_errors(reconstruction_errors_lst, case, feat_size, out_file='Figures/recon_err.txt'):
    # model = create_autoencoder(feat_size)
    # model.compile(loss=euclidean_distance_loss, optimizer='adam')
    #
    # # Load weights
    # if feat_size < 25:
    #     model.load_weights("Models_dump/corr_AE_" + str(feat_size) + case + ".hdf5")
    # else:
    #     model.load_weights("Models_dump/new_AE_" + case + ".hdf5")
    #
    # a = time.time()
    # data_pred = model.predict(x_test)
    # # reconstruction_errors_dict = {'norm':[], 'attack':[]}
    # # for i in range(len(data_pred)):
    # #     if y_test[i] == 1:
    # #         reconstruction_errors_dict['norm'].append(distance(data_pred[i], x_test[i]))
    # #     else: # y_test[i] == 0:
    # #         reconstruction_errors_dict['attack'].append(distance(data_pred[i], x_test[i]))
    # reconstruction_errors_lst = []
    # for i in range(len(data_pred)):
    #     reconstruction_errors_lst.append([distance(data_pred[i], x_test[i]), y_test[i]])

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


def save_roc_to_txt(out_file, y_test_pred_labels_dict):
    # print(f'y_test_pred_labels_dict:{y_test_pred_labels_dict}')
    with open(out_file, 'w') as out_hdl:
        line = 'model-y_true[0]:y_preds_probs[0], y_true[1]:y_preds_probs[1],....'
        out_hdl.write(line + '\n')
        for idx, (key, value) in enumerate(y_test_pred_labels_dict.items()):
            line = str(key) + '@'
            if len(value) == 0:
                print(f'key:{key}, value:{value}')
                continue
            y_true, y_preds_labels = value
            y_true = np.reshape(y_true, (y_true.shape[0],))
            for i in range(y_true.shape[0] - 1):
                line += str(y_true[i]) + ':' + str(y_preds_labels[i]) + ','
            line += str(y_true[-1]) + ':' + str(y_preds_labels[i])
            out_hdl.write(line + '\n')
