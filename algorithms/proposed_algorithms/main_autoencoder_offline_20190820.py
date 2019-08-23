"""
    autoencoder evaluation on offline experiments

"""
import configuration

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.callbacks import ModelCheckpoint

from classical_ml_algorithms import test_DT, train_PCA, test_PCA, train_IF, test_IF, test_OCSVM, train_OCSVM, train_DT
from feature_selection_20190611 import feature_selection, select_sub_features_data, feature_selection_with_sklearn, \
    select_top_k_features, feature_selection_new
from utils.load_data import load_data, balance_data
from utils.metrics_measure import calucalate_metrics, save_sub_features_res_metrics, get_optimal_thres, \
    normalize_for_ml, save_reconstruction_errors, save_roc_to_txt, save_thres_res_metrics, get_max_reconstruction_error, \
    z_score_np, save_features_selection_results
from plot_all_results import plot_reconstruction_errors_from_txt, plot_loss, plot_roc, plot_AE_thresholds_metrics, \
    plot_sub_features_metrics, plot_features_selection_results
from utils.read_data_un import Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from sklearn.utils import shuffle

from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load
import os
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from sklearn.ensemble import IsolationForest
import time
# from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import regularizers
from sklearn.utils import shuffle
from sklearn.decomposition import PCA as PCAS
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D



# Loss function
def euclidean_distance_loss(y_true, y_pred, axis=1):
    """

    :param y_true:
    :param y_pred:
    :param axis: 1: columns (the results has number of rows ), 0: rows
    :return:
    """
    # return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))  # what is difference between -1 and 1.

    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=axis))


# Original Loss function
def distance(predictions, labels, axis=1):
    """

    :param predictions: y_preds
    :param labels:  y_true
    :return:
    """
    distance = np.sqrt(np.sum(np.square(predictions - labels), axis=axis))  # for each y, sum all value to one.
    # distance = np.sqrt(np.sum(np.square(predictions - labels)) / len(labels))

    return distance


# Create the autoencoder for training and testing
def create_autoencoder(feat_size):
    """

    :param feat_size: input_size
    :return:
    """
    # declaring the schematic of the autoencoder
    inp_dim = int(feat_size)
    h1 = 24
    h2 = 16

    model = Sequential()
    model.add(Dense(h1, input_dim=inp_dim))
    model.add(LeakyReLU())
    # model.add(Dropout(0.1))
    model.add(Dense(h2))  # latent layer
    model.add(LeakyReLU())
    # model.add(Dropout(0.1))
    model.add(Dense(h1))
    model.add(LeakyReLU())
    model.add(Dense(inp_dim))
    model.add(LeakyReLU())

    return model


# To train the autoencoder model
def train_AE(train_data, val_data, case, feat_size, Epochs=2, batch_size=32):
    """
        only use normal samples to train AE.
    :param train_data:
    :param val_data:
    :param case:
    :param feat_size:
    :param Epochs:
    :return:
    """
    model = create_autoencoder(feat_size)
    model.compile(loss=euclidean_distance_loss, optimizer='adam')

    st = time.time()
    training_results = model.fit(train_data, train_data, validation_data=(val_data, val_data), epochs=Epochs,
                                 batch_size=batch_size,
                                 verbose=0)
    print('AE training for Case : ', case, ' = ', time.time() - st, 's')

    score = model.evaluate(train_data, train_data, batch_size=batch_size, verbose=0)
    print("Train Loss = ", score, 'model.metrics_names', model.metrics_names)

    score = model.evaluate(val_data, val_data, batch_size=batch_size, verbose=0)
    print("Validation Loss = ", score, 'model.metrics_names', model.metrics_names)

    # # To save the weights of the model
    # if feat_size < 25:
    #     model.save_weights("models_dumping/corr_AE_" + str(feat_size) + case + ".hdf5")
    # else:
    #     model.save_weights("models_dumping/new_AE_" + case + ".hdf5")
    model.save_weights("models_dumping/AE_" + str(feat_size) + '_' + case + ".hdf5")

    return training_results.history


# TO test the model
def test_AE(data_test, data_test_labels, data_train, case, feat_size, thres_AE=1.5):
    """

    :param data_test:
    :param data_test_labels:
    :param case:
    :param feat_size:
    :param thres_AE:
    :return:
    """
    model = create_autoencoder(feat_size)
    model.compile(loss=euclidean_distance_loss, optimizer='adam')

    # # Load weights
    # if feat_size < 25:
    #     model.load_weights("models_dumping/corr_AE_" + str(feat_size) + case + ".hdf5")
    # else:
    #     model.load_weights("models_dumping/new_AE_" + case + ".hdf5")
    model.load_weights("models_dumping/AE_" + str(feat_size) + '_' + case + ".hdf5")

    st = time.time()
    data_preds = model.predict(data_test)  # data_preds equals to input data
    print("AE Test time for ", case, " = : ", time.time() - st)

    #### for test different distance.
    # # import tensorflow as tf
    # # a_true = tf.convert_to_tensor(data_test, dtype=tf.float32)
    # # a_pred = tf.convert_to_tensor(data_preds, dtype=tf.float32)
    # # p=euclidean_distance_loss(a_true, a_pred, axis=1)
    # # p_1 = K.eval(p) # tensor to numpy
    # #
    # # a1=distance(data_preds, data_test, axis=1)
    # # print('euclidean_distance_loss() = distance() ?', p_1 - a1)  # there is a little difference.

    # pred_arr = []
    # reconstr_errors_lst = []
    # for i in range(len(data_preds)):
    #     pred_arr.append(distance(data_preds[i], data_test[i]))
    #     reconstr_errors_lst.append([distance(data_preds[i], data_test[i]), data_test_labels[i]])
    # pred_arr = np.array(pred_arr)

    pred_arr = distance(data_preds, data_test, axis=1)
    y_pred_label_AE = np.zeros((pred_arr.shape))
    y_pred_label_AE[pred_arr < thres_AE] = 1

    # data_preds = model.predict(data_train)  # data_preds equals to input data
    # def predict_proba(X):           # come from PCA: predict_proba(self, X, method='linear'):
    #     train_scores = self.decision_scores_
    #
    #     test_scores = self.decision_function(X)
    #
    #     probs = np.zeros([X.shape[0], int(self._classes)])
    #     if method == 'linear':
    #         scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
    #         probs[:, 1] = scaler.transform(
    #             test_scores.reshape(-1, 1)).ravel().clip(0, 1)
    #         probs[:, 0] = 1 - probs[:, 1]
    #         return probs

    ### change the value to probability, use min-max method.
    probs = np.zeros([data_test.shape[0], 2])
    # pred_arr_train = distance(model.predict(data_train) , data_train, axis=1)
    scaler = MinMaxScaler().fit(pred_arr.reshape(-1, 1))
    probs[:, 1] = 1 - scaler.transform(pred_arr.reshape(-1, 1)).ravel().clip(0, 1)  # normlize to [0,1]
    probs[:, 0] = 1 - probs[:, 1]
    y_pred_probs_AE = probs[:, 1]

    pred_arr = np.reshape(pred_arr, (pred_arr.shape[0], 1))
    reconstr_errors_lst = np.concatenate([pred_arr, data_test_labels], axis=1)  # concatenate by columns

    # tp, tn, fp, fn = [], [], [], []
    # tpp, tnp, fpp, fnp = [], [], [], []
    #
    # for i in range(y_preds.shape[0]):
    #     if y_preds[i] == 1 and data_test_labels[i] == 1:
    #         tn.append(data_test[i])
    #         tnp.append(pred_arr[i])
    #     elif y_preds[i] == 1 and data_test_labels[i] == 0:
    #         fn.append(data_test[i])
    #         fnp.append(pred_arr[i])
    #     elif y_preds[i] == 0 and data_test_labels[i] == 1:
    #         fp.append(data_test[i])
    #         fpp.append(pred_arr[i])
    #     elif y_preds[i] == 0 and data_test_labels[i] == 0:
    #         tp.append(data_test[i])
    #         tpp.append(pred_arr[i])
    #
    # # plot_point(tp,tn,fp,fn)
    #
    # # for i in range(10):
    # #    print("\t\t", tpp[i], "\t",tnp[i], "\t",fpp[i], "\t", fnp[i])
    #
    # # print("\n\n", len(tp), "\n",len(tn), "\n",len(fp), "\n",len(fn), "\n\n")
    #
    # conf = confusion_matrix(data_test_labels, y_preds)
    #
    # try:
    #     pr = conf[0, 0] / (conf[0, 0] + conf[1, 0])
    # except:
    #     pr = 0
    #
    # try:
    #     recall = conf[0, 0] / (conf[0, 0] + conf[0, 1])
    # except:
    #     recall = 0
    #
    # try:
    #     f1 = (2 * pr * recall) / (pr + recall)
    # except:
    #     f1 = 0
    #
    # # print(classification_report(data_test_labels,pred_class))
    # try:
    #     print("F1 Score : ", f1)
    #     fpr = (conf[1, 0] / (conf[1, 0] + conf[1, 1]))
    #     print("FPR : ", (conf[1, 0] / (conf[1, 0] + conf[1, 1])))
    #     print("DR : ", recall)
    #     print(conf)
    # except:
    #     fpr = 0
    #     print("FPR = 100")
    #
    # try:
    #     fnr = (conf[0, 1] / (conf[0, 1] + conf[0, 0]))
    # except:
    #     fnr = 0

    # recall, fnr, fpr, tnr, acc = calucalate_metrics(data_test_labels, y_preds)

    return y_pred_label_AE, y_pred_probs_AE, reconstr_errors_lst


def experiment_1_without_feature_selection(Epochs, case, optimal_thres_AE, find_optimal_thres_flg, random_state,
                                           x_norm_train, y_norm_train, x_norm_val,
                                           y_norm_val, x_norm_test, y_norm_test,
                                           x_attack_1, y_attack_1, factor, show_flg, balance_train_data_flg,
                                           all_res_metrics_feat_list=[], sub_features_lst=[], title_flg=True,
                                           cnt_thres=2):
    if type(x_attack_1) != type(None):
        print(
            f'**all_norm:{x_norm_train.shape[0]+x_norm_val.shape[0]+x_norm_test.shape[0]}, all_attack:{x_attack_1.shape[0]}')
        print(f'x_norm_train:{x_norm_train.shape}, x_attack_train:{0}')
        print(f'x_norm_val:{x_norm_val.shape}, x_attack_val:{0}')
        print(f'x_norm_test:{x_norm_test.shape}, x_attack_test:{x_attack_1.shape}')
    else:
        print(
            f'all_norm:{x_norm_train.shape[0]+x_norm_val.shape[0]+x_norm_test.shape[0]}, all_attack:{0}')
        print(f'x_norm_train:{x_norm_train.shape}, x_attack_train:{0}')
        print(f'x_norm_val:{x_norm_val.shape}, x_attack_val:{0}')
        print(f'x_norm_test:{x_norm_test.shape}, x_attack_test:{0}')

    print('\nStep 3. train and evaluate AE on case:', case)
    num_features = x_norm_train.shape[1]
    print('\nStep 3-1. train AE on case:', case)
    print(f'x_norm_train.shape: {x_norm_train.shape}')
    training_losses = train_AE(x_norm_train, x_norm_val, case, num_features, Epochs)
    print(f'training_losses:{training_losses}')
    if find_optimal_thres_flg:
        print('find the optimal threshold for AE by using training losss')
        optimal_thres_AE, factor, key, = get_optimal_thres(training_losses, factor=factor, key='loss')
    else:
        print('using the presetted threshold')
        optimal_thres_AE = optimal_thres_AE
        key = None
        factor = None
    print(f'optimal_thres_AE_val:{optimal_thres_AE}, factor = {factor}, key = {key}')
    if show_flg:
        plot_loss(training_losses['loss'], training_losses['val_loss'], title_flg=title_flg, title=f'{case}, key:{key}')
    print('Test AE on train set on case:', case)
    print(f'x_norm_train.shape: {x_norm_train.shape}')
    y_pred_label_AE, y_pred_probs_AE, _ = test_AE(x_norm_train, y_norm_train, x_norm_train, case, num_features,
                                                  thres_AE=optimal_thres_AE)
    tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_norm_train, y_pred_label_AE)

    ### evaluation
    print('\nStep 3-2. evaluate AE on case:', case)
    y_test_pred_prob_dict = {'AE': [], 'DT': [], 'PCA': [], 'IF': [], 'OCSVM': []}
    if type(x_attack_1) != type(None):
        x_test = np.concatenate([x_norm_test, x_attack_1])
        y_test = np.concatenate([y_norm_test, y_attack_1])
    else:
        x_test = x_norm_test
        y_test = y_norm_test

    ### 1. evaluate AE
    print('\n1). evaluate AE on test set:')
    print(f'x_test.shape: {x_test.shape}, case: {case}')
    print(f'optimal_thres_AE {optimal_thres_AE} achieved from key={key}, factor = {factor}, key = {key}')
    y_pred_label_AE, y_pred_probs_AE, reconstr_errs_arr = test_AE(x_test, y_test, x_norm_train, case, num_features,
                                                                  thres_AE=optimal_thres_AE)
    y_test_pred_prob_dict['AE'] = (y_test, y_pred_probs_AE)  # used for ROC, which need probability.
    tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_test, y_pred_label_AE)  # confusion matrix just need y_pred_labels.
    ### 0) Relation between the number of feature and FPR and FNR.
    print(f'\nsave the FPR and FNR reausts of AE with num ={num_features} features on test set.')
    res_metrics_feat_dict = {'tpr': [], 'fnr': [], 'fpr': [], 'tnr': [], 'acc': []}
    res_metrics_feat_dict['tpr'].append(tpr)
    res_metrics_feat_dict['fnr'].append(fnr)
    res_metrics_feat_dict['fpr'].append(fpr)
    res_metrics_feat_dict['tnr'].append(tnr)
    res_metrics_feat_dict['acc'].append(acc)
    all_res_metrics_feat_list.append([sub_features_lst, res_metrics_feat_dict, optimal_thres_AE])

    ### 1) Reconstruction error of normal and attack data
    out_file = f'figures/{case}_recon_err_of_{str(num_features)}_features.txt'
    out_file = save_reconstruction_errors(reconstr_errs_arr, case, x_test.shape[1], out_file)
    if show_flg:
        if case[5] == '3':  # mawi, only has normal samples, so it won't draw attack samples.
            pass
        else:
            title = os.path.split(out_file)[-1].split('.')[0]
            plot_reconstruction_errors_from_txt(input_file=out_file, balance_data_flg=True, random_state=random_state,
                                                title_flg=title_flg, title=title)

    ### 2) Relation between Thresholds and FPR and FNR
    print(f'\n-Evaluate AE with different thresholds on test set with num = {num_features} features.')
    res_metrics_dict = {'tpr': [], 'fnr': [], 'fpr': [], 'tnr': [], 'acc': []}
    thres_lst = []
    max_reconstr_err, min_reconstr_err = get_max_reconstruction_error(
        reconstr_errs_arr)  # max and min val come from test set.
    for idx, thres in enumerate(np.linspace(min_reconstr_err, optimal_thres_AE * 2, num=cnt_thres, endpoint=True)):
        print(f'*** idx: {idx},  thres = {thres}')
        y_pred_label_AE, y_pred_probs_AE, _ = test_AE(x_test, y_test, x_norm_train, case, x_test.shape[1],
                                                      thres_AE=thres)
        tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_test, y_pred_label_AE)
        res_metrics_dict['tpr'].append(tpr)
        res_metrics_dict['fnr'].append(fnr)
        res_metrics_dict['fpr'].append(fpr)
        res_metrics_dict['tnr'].append(tnr)
        res_metrics_dict['acc'].append(acc)
        thres_lst.append(thres)
    out_file = f'figures/{case}_thres_res_metrics_of_{str(num_features)}_features.txt'
    save_thres_res_metrics(out_file, thres_lst, res_metrics_dict)
    title = os.path.split(out_file)[-1].split('.')[0]
    plot_AE_thresholds_metrics(out_file, title_flg=title_flg, title=title)

    ### 2. evaluate DT
    # without attack data in training, so no DT model in this case.
    print('\n2). no need to evaluate DT on test set')

    ### 3. evaluate PCA
    print('\n3). train and evaluate PCA...')
    train_PCA(x_norm_train, case)
    print(f'x_norm_train.shape: {x_norm_train.shape}')
    test_PCA(x_norm_train, y_norm_train, case)

    print(f'x_test.shape: {x_test.shape}')
    y_pred_label_PCA, y_pred_probs_PCA = test_PCA(x_test, y_test, case)
    y_test_pred_prob_dict['PCA'] = (y_test, y_pred_probs_PCA)

    # ### 4. evaluate IF
    print('\n4). train and evaluate IF...')
    train_IF(x_norm_train, case)
    print(f'x_norm_train.shape: {x_norm_train.shape}')
    test_IF(x_norm_train, y_norm_train, case)

    print(f'x_test.shape: {x_test.shape}')
    y_pred_label_IF, y_pred_probs_IF = test_IF(x_test, y_test, case)
    y_test_pred_prob_dict['IF'] = (y_test, y_pred_probs_IF)

    ### 5. evaluate OC-SVM
    print('\n3). train and evaluate OCSVM...')
    train_OCSVM(x_norm_train, case)
    print(f'x_norm_train.shape: {x_norm_train.shape}')
    test_OCSVM(x_norm_train[:100], y_norm_train[:100], case)

    print(f'x_test.shape: {x_test.shape}')
    y_pred_label_OCSVM, y_pred_probs_OCSVM = test_OCSVM(x_test, y_test, case)
    y_test_pred_prob_dict['OCSVM'] = (y_test, y_pred_probs_OCSVM)

    out_file = f'figures/{case}_roc_data_of_{str(num_features)}_features.txt'
    print(f'roc, out_file:{out_file}')
    save_roc_to_txt(out_file, y_test_pred_prob_dict)
    if show_flg:
        # y_test_PCA = np.reshape(y_test_PCA, (y_test_PCA.shape[0],))
        title = os.path.split(out_file)[-1].split('.')[0]
        # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob) # IMPORTANT: first argument is true values, second argument is predicted probabilities
        plot_roc(out_file, balance_data_flg=True, title_flg=title_flg, title=title)

    return all_res_metrics_feat_list, optimal_thres_AE


def main(Epochs=2, optimal_thres_AE=5, find_optimal_thres_flg=False, cases=['uSc3C2_min_max_20_14'], factor=10,
         random_state=42, show_flg=True,
         balance_train_data_flg=True, title_flg=True, sub_features_flg=False, sub_features_lst=[]):
    """

    :param Epochs:
    :param cases:  For scenario 1 case 1
    :param show_flg:
    :param balance_train_data_flg:
    :return:
    """
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # what is its purpose?
    print(f'\nEpochs:{Epochs}, optimal_thres_AE:{optimal_thres_AE}, find_optimal_thres_flg:{find_optimal_thres_flg}')

    norm_flg = True
    for case in cases:
        if case[3] == '1':
            if case[5] == '1':
                print('\n@@@train and evaluate AE on SYNT:', case)
            elif case[5] == '2':
                print('\n@@@train and evaluate AE on unb:', case)
            elif case[5] == '3':
                print('\n@@@train and evaluate AE on mawi:', case)
            else:
                print('not implement')
                return -1

            print("\nStep 1. loading data...")
            (x_norm_train_raw, y_norm_train), (x_norm_val_raw, y_norm_val), (x_norm_test_raw, y_norm_test), (
                x_attack_1_raw, y_attack_1) = load_data(case, random_state=random_state)
            print("\n+Step 1-1. preprocessing data...")

            analyize_features_flg = True  # True: do feature selection.
            if not analyize_features_flg:

                if sub_features_flg:
                    print(f'\n---num_sub_features:{len(sub_features_lst)}, they are {sub_features_lst}')
                    print('select the corresponding features data.')
                    ### training
                    x_norm_train_raw = select_sub_features_data(x_norm_train_raw, sub_features_lst)
                    x_norm_val_raw = select_sub_features_data(x_norm_val_raw, sub_features_lst)
                    x_norm_test_raw = select_sub_features_data(x_norm_test_raw, sub_features_lst)
                    if case[5] == '1' or case[5] == '2':  # case[5] != '3'
                        x_attack_1_raw = select_sub_features_data(x_attack_1_raw, sub_features_lst)
                else:
                    print(f'\n---all_features:{x_norm_train_raw.shape[1]}')

                if norm_flg:
                    x_norm_train_raw, x_norm_train_mu, x_norm_train_std = z_score_np(x_norm_train_raw, mu='', d_std='')
                    x_norm_val_raw, _, _ = z_score_np(x_norm_val_raw, mu=x_norm_train_mu, d_std=x_norm_train_std)
                    x_norm_test_raw, _, _ = z_score_np(x_norm_test_raw, mu=x_norm_train_mu, d_std=x_norm_train_std)
                    if type(x_attack_1_raw) != type(None):
                        x_attack_1_raw, _, _ = z_score_np(x_attack_1_raw, mu=x_norm_train_mu, d_std=x_norm_train_std)

                all_res_metrics_feat_list = []
                # sub_features_lst=[]
                experiment_1_without_feature_selection(Epochs, case, optimal_thres_AE, find_optimal_thres_flg,
                                                       random_state, x_norm_train_raw, y_norm_train,
                                                       x_norm_val_raw, y_norm_val, x_norm_test_raw, y_norm_test,
                                                       x_attack_1_raw, y_attack_1, factor, show_flg,
                                                       balance_train_data_flg,
                                                       all_res_metrics_feat_list, sub_features_lst, title_flg)
            else:
                print(f"\nStep 2. analyize different number of features on case:{case}.")
                ### Relation between the number of feature and FPR and FNR.
                all_res_metrics_feat_list = []  # save all the sub_features results. each item is a dictionary.
                feat_selection_method = 1  # # pearson correlation
                # for i in range(2, x_norm_train_raw.shape[1] + 1, 8):
                features_descended_dict = feature_selection_new(x_norm_train_raw)
                out_file = f"figures/{case}_sub_features_list.txt"
                out_file = save_features_selection_results(out_file, features_descended_dict)
                plot_features_selection_results(out_file, title_flg=True, title=f'feature_number')

                cnt = 0
                for key, value in features_descended_dict.items():
                    # if feat_selection_method == 1:  # pearson correlation
                    #     sub_features_list = feature_selection(x_norm_train_raw, num_features=i)
                    # else:  # feat_selection_method == 2:  # sklearn: VarianceThreshold
                    #     sub_features_list = feature_selection_with_sklearn(x_norm_train_raw, None)
                    sub_features_lst = value
                    if len(sub_features_lst) == x_norm_train_raw.shape[1]:
                        print(
                            f'len(sub_features_lst) == x_norm_train_raw.shape[1]: {len(sub_features_lst) == x_norm_train_raw.shape[1]}')

                    elif len(sub_features_lst) < 2 or len(sub_features_lst) % 2 != 0:
                        continue
                    else:
                        pass
                    print(f'\n---num_sub_features:{len(sub_features_lst)}, they are {sub_features_lst}')

                    print('select the corresponding features data.')
                    ### training
                    x_norm_train = select_sub_features_data(x_norm_train_raw, sub_features_lst)
                    x_norm_val = select_sub_features_data(x_norm_val_raw, sub_features_lst)
                    x_norm_test = select_sub_features_data(x_norm_test_raw, sub_features_lst)
                    x_attack_1 = select_sub_features_data(x_attack_1_raw, sub_features_lst)

                    if norm_flg:
                        x_norm_train, x_norm_train_mu, x_norm_train_std = z_score_np(x_norm_train, mu='',
                                                                                     d_std='')
                        x_norm_val, _, _ = z_score_np(x_norm_val, mu=x_norm_train_mu, d_std=x_norm_train_std)
                        x_norm_test, _, _ = z_score_np(x_norm_test, mu=x_norm_train_mu, d_std=x_norm_train_std)
                        if type(x_attack_1) != type(None):
                            x_attack_1, _, _ = z_score_np(x_attack_1, mu=x_norm_train_mu,
                                                          d_std=x_norm_train_std)

                    experiment_1_without_feature_selection(Epochs, case, optimal_thres_AE, find_optimal_thres_flg,
                                                           random_state, x_norm_train, y_norm_train,
                                                           x_norm_val, y_norm_val, x_norm_test, y_norm_test,
                                                           x_attack_1, y_attack_1, factor, show_flg,
                                                           balance_train_data_flg, all_res_metrics_feat_list,
                                                           sub_features_lst, title_flg)

                    # cnt +=1
                    # if cnt >= 2:
                    #     break

                if show_flg:
                    "Relation between the number of feature and FPR and FNR."
                    out_file = f'figures/{case}_all_num_features_res_metrics.txt'
                    save_sub_features_res_metrics(out_file, all_res_metrics_feat_list, optimal_thres_AE)
                    title = os.path.split(out_file)[-1].split('.')[0]
                    plot_sub_features_metrics(out_file, title_flg=title_flg, title=title)

        elif case[3] == '2':  # Experiment 2
            if case[5] == '1':  # train on SYNT
                print('\n@@@train on SYNT:', case)
                pass
            elif case[5] == '2':
                print('\n@@@train on unb:', case)

                print("\nStep 1. loading data...")
                UNB_train_set_raw, UNB_val_set_raw, UNB_test_set_raw, SYNT_test_set_raw, MAWI_test_set_raw = load_data(
                    case, random_state)

                print("\n+Step 1-1. preprocessing data...")
                if norm_flg:
                    ### normal_for unsupervised_ml
                    UNB_train_set_tmp, x_norm_train_mu, x_norm_train_std = normalize_for_ml(UNB_train_set_raw, mu='',
                                                                                            d_std='',
                                                                                            flg='norm_for_unsupervised_ml')  # should be noticed that attack samples do not be normalized
                    UNB_val_set_tmp, _, _ = normalize_for_ml(UNB_val_set_raw, mu=x_norm_train_mu,
                                                             d_std=x_norm_train_std,
                                                             flg='norm_for_unsupervised_ml')
                    UNB_test_set_tmp, _, _ = normalize_for_ml(UNB_test_set_raw, mu=x_norm_train_mu,
                                                              d_std=x_norm_train_std,
                                                              flg='norm_for_unsupervised_ml', test_flg=True)
                    SYNT_test_set_tmp, _, _ = normalize_for_ml(SYNT_test_set_raw, mu=x_norm_train_mu,
                                                               d_std=x_norm_train_std,
                                                               flg='norm_for_unsupervised_ml', test_flg=True)

                    x_norm_train, y_norm_train, x_attack_train, y_attack_train = UNB_train_set_tmp
                    x_attack_train, _, _ = z_score_np(x_attack_train, mu=x_norm_train_mu, d_std=x_norm_train_std)
                    x_norm_val, y_norm_val, x_attack_val, y_attack_val = UNB_val_set_tmp
                    x_attack_val, _, _ = z_score_np(x_attack_val, mu=x_norm_train_mu, d_std=x_norm_train_std)

                    x_norm_test_UNB, y_norm_test_UNB, x_attack_test_UNB, y_attack_test_UNB = UNB_test_set_tmp

                    x_norm_test_SYNT, y_norm_test_SYNT, x_attack_test_SYNT, y_attack_test_SYNT = SYNT_test_set_tmp

                    x_norm_test_MAWI, y_norm_test_MAWI, _, _ = MAWI_test_set_raw
                    x_norm_test_MAWI, _, _ = z_score_np(x_norm_test_MAWI, mu=x_norm_train_mu, d_std=x_norm_train_std)

                else:
                    x_norm_train, y_norm_train, x_attack_train, y_attack_train = UNB_train_set_raw
                    x_norm_val, y_norm_val, x_attack_val, y_attack_val = UNB_val_set_raw
                    x_norm_test_UNB, y_norm_test_UNB, x_attack_test_UNB, y_attack_test_UNB = UNB_test_set_raw
                    x_norm_test_SYNT, y_norm_test_SYNT, x_attack_test_SYNT, y_attack_test_SYNT = SYNT_test_set_raw
                    x_norm_test_MAWI, y_norm_test_MAWI, _, _ = MAWI_test_set_raw

                num_features = x_norm_train.shape[1]
                print(
                    f'**all_norm:{x_norm_train.shape[0]+x_norm_val.shape[0]+x_norm_test_UNB.shape[0]}, all_attack:{x_attack_train.shape[0]+x_attack_val.shape[0]+x_attack_test_UNB.shape[0]}')
                print(f'x_norm_train:{x_norm_train.shape}, x_attack_train:{x_attack_train.shape}')
                print(f'x_norm_val:{x_norm_val.shape}, x_attack_val:{x_attack_val.shape}')
                print(f'x_norm_test_UNB:{x_norm_test_UNB.shape}, x_attack_test_UNB:{x_attack_test_UNB.shape}')
                print(f'x_norm_test_SYNT:{x_norm_test_SYNT.shape}, x_attack_test_SYNT:{x_attack_test_SYNT.shape}')
                print(f'x_norm_test_MAWI:{x_norm_test_MAWI.shape}, x_attack_test_MAWI:{0}')

                ### training
                print('\nStep 2. train and evaluate AE on case:', case)
                print('\nStep 2-1. train AE on case:', case)
                print(f'x_norm_train.shape: {x_norm_train.shape}')
                training_losses = train_AE(x_norm_train, x_norm_val, case, x_norm_train.shape[1], Epochs)
                if find_optimal_thres_flg:
                    print('find the optimal threshold for AE by using training losss')
                    optimal_thres_AE, factor, key, = get_optimal_thres(training_losses, factor=factor, key='loss')
                else:
                    print('using the presetted threshold')
                    optimal_thres_AE = optimal_thres_AE
                    key = None
                    factor = None
                print(f'optimal_thres_AE_val:{optimal_thres_AE}, factor = {factor}, key = {key}')
                if show_flg:
                    title = f'{case}, key:{key}'
                    plot_loss(training_losses['loss'], training_losses['val_loss'], title_flg=title_flg, title=title)
                print('Test AE on train set on case:', case)
                print(f'x_norm_train.shape: {x_norm_train.shape}')
                y_pred_AE, reconstr_errs_arr = test_AE(x_norm_train, y_norm_train, x_norm_train, case,
                                                       x_norm_train.shape[1], thres_AE=optimal_thres_AE)
                tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_norm_train, y_pred_AE)

                ### evaluation
                print('\nStep 2-2. test AE on test set, case:', case)
                # unb test set

                x_test_UNB = np.concatenate([x_norm_test_UNB, x_attack_test_UNB])
                y_test_UNB = np.concatenate([y_norm_test_UNB, y_attack_test_UNB])

                ### 2) Relation between Thresholds and FPR and FNR
                print(f'\n-Evaluate AE with different thresholds on test set with num = {num_features} features.')
                res_metrics_dict = {'tpr': [], 'fnr': [], 'fpr': [], 'tnr': [], 'acc': []}
                thres_lst = []
                max_reconstr_err, min_reconstr_err = get_max_reconstruction_error(
                    reconstr_errs_arr)  # max and min val come from test set.
                for thres in np.linspace(min_reconstr_err, optimal_thres_AE * 2, num=50, endpoint=True):
                    print(f'*** thres = {thres}')
                    y_pred_AE, _ = test_AE(x_test_UNB, y_test_UNB, x_norm_train, case, x_test_UNB.shape[1],
                                           thres_AE=thres)
                    tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_test_UNB, y_pred_AE)
                    res_metrics_dict['tpr'].append(tpr)
                    res_metrics_dict['fnr'].append(fnr)
                    res_metrics_dict['fpr'].append(fpr)
                    res_metrics_dict['tnr'].append(tnr)
                    res_metrics_dict['acc'].append(acc)
                    thres_lst.append(thres)
                out_file = f'figures/{case}_thres_res_metrics_of_{str(num_features)}_features.txt'
                save_thres_res_metrics(out_file, thres_lst, res_metrics_dict)
                title = os.path.split(out_file)[-1].split('.')[0]
                plot_AE_thresholds_metrics(out_file, title_flg=title_flg, title=title)

                print('\n1)test AE on unb test set')
                print(f'x_test_UNB.shape: {x_test_UNB.shape}')
                y_pred_UNB, reconstr_errs_arr_UNB = test_AE(x_test_UNB, y_test_UNB, x_norm_train, case,
                                                            x_test_UNB.shape[1],
                                                            thres_AE=optimal_thres_AE)
                tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_test_UNB, y_pred_UNB)

                # SYNT test set
                print('\n2)test AE on SYNT test set')
                x_test_SYNT = np.concatenate([x_norm_test_SYNT, x_attack_test_SYNT])
                y_test_SYNT = np.concatenate([y_norm_test_SYNT, y_attack_test_SYNT])
                print(f'x_test_SYNT.shape: {x_test_SYNT.shape}')
                y_pred_SYNT, reconstr_errs_arr_SYNT = test_AE(x_test_SYNT, y_test_SYNT, x_norm_train, case,
                                                              x_test_SYNT.shape[1],
                                                              thres_AE=optimal_thres_AE)
                tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_test_SYNT, y_pred_SYNT)

                # mawi test set
                print('\n3)test AE on mawi test set')
                x_test_MAWI = x_norm_test_MAWI
                y_test_MAWI = y_norm_test_MAWI
                print(f'x_test_MAWI.shape: {x_test_MAWI.shape}')
                y_pred_MAWI, reconstr_errs_arr_MAWI = test_AE(x_test_MAWI, y_test_MAWI, x_norm_train, case,
                                                              x_test_MAWI.shape[1],
                                                              thres_AE=optimal_thres_AE)
                tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_test_MAWI, y_pred_MAWI)

                print('\nStep 3. train and evaluate PCA , case:', case)
                ### 3. evaluate PCA
                train_PCA(x_norm_train, case)
                print(f'x_norm_train.shape: {x_norm_train.shape}')
                test_PCA(x_norm_train, y_norm_train, case)

                print(f'x_test_UNB.shape: {x_test_UNB.shape}')
                test_PCA(x_test_UNB, y_test_UNB, case)
                print(f'x_test_SYNT.shape: {x_test_SYNT.shape}')
                test_PCA(x_test_SYNT, y_test_SYNT, case)
                print(f'x_test_MAWI.shape: {x_test_MAWI.shape}')
                test_PCA(x_test_MAWI, y_test_MAWI, case)

                ### 4. evaluate IF
                print('\nStep 4. train and evaluate IF, case:', case)
                train_IF(x_norm_train, case)
                print(f'x_norm_train.shape: {x_norm_train.shape}')
                test_IF(x_norm_train, y_norm_train, case)

                print(f'x_test_UNB.shape: {x_test_UNB.shape}')
                test_IF(x_test_UNB, y_test_UNB, case)
                print(f'x_test_SYNT.shape: {x_test_SYNT.shape}')
                test_IF(x_test_SYNT, y_test_SYNT, case)
                print(f'x_test_MAWI.shape: {x_test_MAWI.shape}')
                test_IF(x_test_MAWI, y_test_MAWI, case)
                #
                ### 5. evaluate OC-SVM
                print('\nStep 5. train and evaluate OCSVM,  case:', case)
                train_OCSVM(x_norm_train, case)
                print(f'x_norm_train.shape: {x_norm_train.shape}')
                test_OCSVM(x_norm_train, y_norm_train, case)

                print(f'x_test_UNB.shape: {x_test_UNB.shape}')
                test_OCSVM(x_test_UNB, y_test_UNB, case)
                print(f'x_test_SYNT.shape: {x_test_SYNT.shape}')
                test_OCSVM(x_test_SYNT, y_test_SYNT, case)
                print(f'x_test_MAWI.shape: {x_test_MAWI.shape}')
                test_OCSVM(x_test_MAWI, y_test_MAWI, case)

                print('\nStep 6. train and evaluate DT on case:', case)
                ### evaluate DT
                print("\n-Step 3-1. preprocessing data for DT ...")
                # **** train set for supervised machine learning (DT)
                # norm_flg = False
                if norm_flg:
                    ### normal_for supervised_ml
                    ### normal_for unsupervised_ml
                    UNB_train_set_tmp, x_norm_train_mu, x_norm_train_std = normalize_for_ml(UNB_train_set_raw, mu='',
                                                                                            d_std='',
                                                                                            flg='norm_for_unsupervised_ml')  # should be noticed that attack samples do not be normalized
                    UNB_val_set_tmp, _, _ = normalize_for_ml(UNB_val_set_raw, mu=x_norm_train_mu,
                                                             d_std=x_norm_train_std,
                                                             flg='norm_for_supervised_ml')
                    UNB_test_set_tmp, _, _ = normalize_for_ml(UNB_test_set_raw, mu=x_norm_train_mu,
                                                              d_std=x_norm_train_std,
                                                              flg='norm_for_supervised_ml', test_flg=True)
                    SYNT_test_set_tmp, _, _ = normalize_for_ml(SYNT_test_set_raw, mu=x_norm_train_mu,
                                                               d_std=x_norm_train_std,
                                                               flg='norm_for_supervised_ml', test_flg=True)

                    x_norm_train_DT, y_norm_train_DT, x_attack_train_DT, y_attack_train_DT = UNB_train_set_tmp
                    x_attack_train_DT, _, _ = z_score_np(x_attack_train_DT, mu=x_norm_train_mu, d_std=x_norm_train_std)
                    x_norm_val_DT, y_norm_val_DT, x_attack_val_DT, y_attack_val_DT = UNB_val_set_tmp
                    x_attack_val_DT, _, _ = z_score_np(x_attack_val, mu=x_norm_train_mu, d_std=x_norm_train_std)

                    x_norm_test_UNB_DT, y_norm_test_UNB_DT, x_attack_test_UNB_DT, y_attack_test_UNB_DT = UNB_test_set_tmp

                    x_norm_test_SYNT_DT, y_norm_test_SYNT_DT, x_attack_test_SYNT_DT, y_attack_test_SYNT_DT = SYNT_test_set_tmp

                    x_norm_test_MAWI_DT, y_norm_test_MAWI_DT, _, _ = MAWI_test_set_raw
                    x_norm_test_MAWI_DT, _, _ = z_score_np(x_norm_test_MAWI_DT, mu=x_norm_train_mu,
                                                           d_std=x_norm_train_std)

                else:
                    x_norm_train_DT, y_norm_train_DT, x_attack_train_DT, y_attack_train_DT = UNB_train_set_raw
                    x_norm_val_DT, y_norm_val_DT, x_attack_val_DT, y_attack_val_DT = UNB_val_set_raw
                    x_norm_test_UNB_DT, y_norm_test_UNB_DT, x_attack_test_UNB_DT, y_attack_test_UNB_DT = UNB_test_set_raw
                    x_norm_test_SYNT_DT, y_norm_test_SYNT_DT, x_attack_test_SYNT_DT, y_attack_test_SYNT_DT = SYNT_test_set_raw
                    x_norm_test_MAWI_DT, y_norm_test_MAWI_DT, x_attack_test_MAWI_DT, y_attack_test_MAWI_DT = MAWI_test_set_raw

                # balance_train_data_flg= 0
                if balance_train_data_flg:
                    x_train_DT, y_train_DT = balance_data(x_norm_train_DT, y_norm_train_DT, x_attack_train_DT,
                                                          y_attack_train_DT, random_state=random_state)
                else:
                    x_train_DT = np.concatenate([x_norm_train_DT, x_attack_train_DT])
                    y_train_DT = np.concatenate([y_norm_train_DT, y_attack_train_DT])
                    print(f'Without data balance, x_train.shape: {x_train_DT.shape}')
                    print(
                        f' in which, x_norm_train_DT.shape: {x_norm_train_DT.shape}, and x_attack_train_DT.shape: {x_attack_train_DT.shape}')
                # ### without val set for DT
                # x_val = np.concatenate([x_norm_val, x_attack_val])
                # x_val_labels = np.concatenate([y_val, y_attack_val])
                # unb test set
                print('\n-Step 3-2. train DT...')
                train_DT(x_train_DT, y_train_DT, case)

                print('\n-Step 3-3. evaluate DT...')
                print('evaluate DT on train set')
                print(f'x_train_DT.shape: {x_train_DT.shape}')
                test_DT(x_train_DT, y_train_DT, case)

                print('1) evaluate DT on unb test set ')
                x_test_UNB_DT = np.concatenate([x_norm_test_UNB_DT, x_attack_test_UNB_DT])
                y_test_UNB_DT = np.concatenate([y_norm_test_UNB_DT, y_attack_test_UNB_DT])
                print(f'x_test_UNB_DT.shape: {x_test_UNB_DT.shape}')
                test_DT(x_test_UNB_DT, y_test_UNB_DT, case)

                print('2) evaluate DT on SYNT test set ')
                x_test_SYNT_DT = np.concatenate([x_norm_test_SYNT_DT, x_attack_test_SYNT_DT])
                y_test_SYNT_DT = np.concatenate([y_norm_test_SYNT_DT, y_attack_test_SYNT_DT])
                print(f'x_test_SYNT_DT.shape: {x_test_SYNT_DT.shape}')
                test_DT(x_test_SYNT_DT, y_test_SYNT_DT, case)

                print('3) evaluate DT on mawi test set')
                x_test_MAWI_DT = x_norm_test_MAWI_DT
                y_test_MAWI_DT = y_norm_test_MAWI_DT
                print(f'x_test_MAWI_DT.shape: {x_test_MAWI_DT.shape}')
                test_DT(x_test_MAWI_DT, y_test_MAWI_DT, case)


            else:  # train on mawi
                print('train on mawi', case)
                pass

        else:  # Experiment 3

            if case[5] == '1':
                print('\n@@@Experiment 3 on SYNT:', case)
                # SYNT_train_set_raw, SYNT_val_set_raw, SYNT_test_set_raw, SYNT_test_set_2_raw = load_data(case)
                # train_set_raw, val_set_raw, test_set_raw, test_set_2_raw = SYNT_train_set_raw, SYNT_val_set_raw, SYNT_test_set_raw, SYNT_test_set_2_raw
            elif case[5] == '2':
                print('\n@@@Experiment 3 on unb:', case)
                # unb  because of unb attack and normal exist huge difference, so DT can easily distingusih them on test set 1 and test set 2.
                ### "x_norm, y_norm, x_attack_1, y_attack_1, x_attack_test_2, y_attack_test_2"
                # UNB_train_set_raw, UNB_val_set_raw, UNB_test_set_raw, UNB_test_set_2_raw = load_data(case)
            elif case[5] == '3':
                print('\n@@@Experiment 3 on mawi:', case)
                ### "x_norm, y_norm, x_attack_1, y_attack_1, x_attack_test_2, y_attack_test_2"
                # MAWI_train_set_raw, MAWI_val_set_raw, MAWI_test_set_raw, MAWI_test_set_2_raw = load_data(case)
            else:
                break

            print("Step 1. loading data...")
            ### "x_norm, y_norm, x_attack_1, y_attack_1, x_attack_test_2, y_attack_test_2"
            train_set_raw, val_set_raw, test_set_raw, test_set_2_raw = load_data(case, random_state=random_state)
            print('\nStep 2. train and evaluate AE on case:', case)
            print("\n+Step 2-1. preprocessing data for AE ...")
            if norm_flg:
                ### normal_for unsupervised_ml
                train_set_tmp, x_norm_train_mu, x_norm_train_std = normalize_for_ml(train_set_raw, mu='', d_std='',
                                                                                    flg='norm_for_unsupervised_ml')  # should be noticed that attack samples do not be normalized
                val_set_tmp, _, _ = normalize_for_ml(val_set_raw, mu=x_norm_train_mu, d_std=x_norm_train_std,
                                                     flg='norm_for_unsupervised_ml')
                test_set_tmp, _, _ = normalize_for_ml(test_set_raw, mu=x_norm_train_mu, d_std=x_norm_train_std,
                                                      flg='norm_for_unsupervised_ml', test_flg=True)

                test_set_2_tmp, _, _ = normalize_for_ml(test_set_2_raw, mu=x_norm_train_mu, d_std=x_norm_train_std,
                                                        flg='norm_for_unsupervised_ml', test_flg=True)
                x_norm_train, y_norm_train, x_attack_train, y_attack_train = train_set_tmp
                x_attack_train, _, _ = z_score_np(x_attack_train, mu=x_norm_train_mu, d_std=x_norm_train_std)
                x_norm_val, y_norm_val, x_attack_val, y_attack_val = val_set_tmp
                x_attack_val, _, _ = z_score_np(x_attack_val, mu=x_norm_train_mu, d_std=x_norm_train_std)
                x_norm_test, y_norm_test, x_attack_test, y_attack_test = test_set_tmp
                x_norm_test_2, y_norm_test_2, x_attack_test_2, y_attack_test_2 = test_set_2_tmp
            else:
                x_norm_train, y_norm_train, x_attack_train, y_attack_train = train_set_raw
                x_norm_val, y_norm_val, x_attack_val, y_attack_val = val_set_raw
                x_norm_test, y_norm_test, x_attack_test, y_attack_test = test_set_raw
                x_norm_test_2, y_norm_test_2, x_attack_test_2, y_attack_test_2 = test_set_2_raw

            num_features = x_norm_train.shape[1]
            print(
                f'**all_norm:{x_norm_train.shape[0]+x_norm_val.shape[0]+x_norm_test.shape[0]}, all_attack:{x_attack_train.shape[0]+x_attack_val.shape[0]+x_attack_test.shape[0]}')
            print(f'x_norm_train:{x_norm_train.shape}, x_attack_train:{x_attack_train.shape}')
            print(f'x_norm_val:{x_norm_val.shape}, x_attack_val:{x_attack_val.shape}')
            print(f'x_norm_test:{x_norm_test.shape}, x_attack_test:{x_attack_test.shape}')
            print(f'x_norm_test_2:{x_norm_test_2.shape}, x_attack_test_2:{x_attack_test_2.shape}')

            ### training for AE without attackack data
            print('\n+Step 2-1. train AE...')
            print(f'x_norm_train.shape: {x_norm_train.shape}')
            training_losses = train_AE(x_norm_train, x_norm_val, case, x_norm_train.shape[1], Epochs)
            if find_optimal_thres_flg:
                print('find the optimal threshold for AE by using training losss')
                optimal_thres_AE, factor, key, = get_optimal_thres(training_losses, factor=factor, key='loss')
            else:
                print('using the presetted threshold')
                optimal_thres_AE = optimal_thres_AE
                key = None
                factor = None
            print(f'optimal_thres_AE_val:{optimal_thres_AE}')
            if show_flg:
                title = f'{case}, key:{key}'
                plot_loss(training_losses['loss'], training_losses['val_loss'], title_flg=title_flg, title=title)

            ### evaluation
            print('\n+Step 2-2. evaluate AE...')
            print('\nevaluate AE on train set')
            x_train_set = np.concatenate([x_norm_train, x_attack_train])
            y_train_set = np.concatenate([y_norm_train, y_attack_train])
            y_pred_AE, reconstr_errs_arr = test_AE(x_train_set, y_train_set, x_norm_train, case, x_train_set.shape[1],
                                                   thres_AE=optimal_thres_AE)
            tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_train_set, y_pred_AE)
            ### evaluate AE
            # print('evaluate AE on test set')
            x_test_1 = np.concatenate([x_norm_test, x_attack_test])
            y_test_1 = np.concatenate([y_norm_test, y_attack_test])

            ### 2) Relation between Thresholds and FPR and FNR
            print(f'\n-Evaluate AE with different thresholds on test set with num = {num_features} features.')
            res_metrics_dict = {'tpr': [], 'fnr': [], 'fpr': [], 'tnr': [], 'acc': []}
            thres_lst = []
            max_reconstr_err, min_reconstr_err = get_max_reconstruction_error(
                reconstr_errs_arr)  # max and min val come from test set.
            for idx, thres in enumerate(np.linspace(min_reconstr_err, optimal_thres_AE * 2, num=50, endpoint=True)):
                print(f'*** idx={idx}, thres = {thres}')
                y_pred_AE, _ = test_AE(x_test_1, y_test_1, x_norm_train, case, x_test_1.shape[1], thres_AE=thres)
                tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_test_1, y_pred_AE)
                res_metrics_dict['tpr'].append(tpr)
                res_metrics_dict['fnr'].append(fnr)
                res_metrics_dict['fpr'].append(fpr)
                res_metrics_dict['tnr'].append(tnr)
                res_metrics_dict['acc'].append(acc)
                thres_lst.append(thres)
            out_file = f'figures/{case}_thres_res_metrics_of_{str(num_features)}_features.txt'
            save_thres_res_metrics(out_file, thres_lst, res_metrics_dict)
            title = os.path.split(out_file)[-1].split('.')[0]
            plot_AE_thresholds_metrics(out_file, title_flg=title_flg, title=title)

            print('\n1) evaluate AE on test set 1')
            print(f'x_test_1.shape: {x_test_1.shape}')
            y_pred_AE, reconstr_errs_arr = test_AE(x_test_1, y_test_1, x_norm_train, case, x_test_1.shape[1],
                                                   thres_AE=optimal_thres_AE)
            tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_test_1, y_pred_AE)

            print('\n2) evaluate AE on test set 2')
            x_test_2 = np.concatenate([x_norm_test, x_attack_test_2])
            y_test_2 = np.concatenate([y_norm_test, y_attack_test_2])
            print(f'x_test_2.shape: {x_test_2.shape}')
            y_pred_AE_2, reconstr_errs_arr_2 = test_AE(x_test_2, y_test_2, x_norm_train, case, x_test_2.shape[1],
                                                       thres_AE=optimal_thres_AE)
            tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_test_2, y_pred_AE_2)
            if show_flg:
                ### 1) Reconstruction error of normal and attack data
                print('\n--save and show the reconstruction errors of normal and attack samples')
                out_file = f'figures/{case}_recon_err_test_1.txt'
                out_file = save_reconstruction_errors(reconstr_errs_arr, case, x_test_1.shape[1], out_file)
                title = os.path.split(out_file)[-1].split('.')[0]
                plot_reconstruction_errors_from_txt(input_file=out_file, balance_data_flg=True, title_flg=title_flg,
                                                    title=title)
                out_file_2 = f'figures/{case}_recon_err_test2.txt'
                out_file_2 = save_reconstruction_errors(reconstr_errs_arr_2, case, x_test_2.shape[1], out_file_2)
                title_2 = os.path.split(out_file_2)[-1].split('.')[0]
                plot_reconstruction_errors_from_txt(input_file=out_file_2, balance_data_flg=True, title_flg=title_flg,
                                                    title=title_2)

            print('\nStep 3. train and evaluate DT on case:', case)
            ### evaluate DT
            print("\n-Step 3-1. preprocessing data for DT ...")
            # **** train set for supervised machine learning (DT)
            norm_flg = False
            if norm_flg:
                ### normal_for unsupervised_ml
                train_set_DT, x_train_mu_DT, x_train_std_DT = normalize_for_ml(train_set_raw, mu='', d_std='',
                                                                               flg='norm_for_supervised_ml')
                val_set_DT, _, _ = normalize_for_ml(val_set_raw, mu=x_train_mu_DT, d_std=x_train_std_DT,
                                                    flg='norm_for_supervised_ml')
                test_set_DT, _, _ = normalize_for_ml(test_set_raw, mu=x_train_mu_DT, d_std=x_train_std_DT,
                                                     flg='norm_for_supervised_ml')

                test_set_DT_2, _, _ = normalize_for_ml(test_set_2_raw, mu=x_train_mu_DT, d_std=x_train_std_DT,
                                                       flg='norm_for_supervised_ml')
                x_norm_train_DT, y_norm_train_DT, x_attack_train_DT, y_attack_train_DT = train_set_DT
                x_norm_val_DT, y_norm_val_DT, x_attack_val_DT, y_attack_val_DT = val_set_DT
                x_norm_test_DT, y_norm_test_DT, x_attack_test_DT, y_attack_test_DT = test_set_DT
                x_norm_test_2_DT, y_norm_test_2_DT, x_attack_test_2_DT, y_attack_test_2_DT = test_set_DT_2
            else:
                x_norm_train_DT, y_norm_train_DT, x_attack_train_DT, y_attack_train_DT = train_set_raw
                x_norm_val_DT, y_norm_val_DT, x_attack_val_DT, y_attack_val_DT = val_set_raw
                x_norm_test_DT, y_norm_test_DT, x_attack_test_DT, y_attack_test_DT = test_set_raw
                x_norm_test_2_DT, y_norm_test_2_DT, x_attack_test_2_DT, y_attack_test_2_DT = test_set_2_raw

            # balance_train_data_flg= 0
            if balance_train_data_flg:
                x_train_DT, y_train_DT = balance_data(x_norm_train_DT, y_norm_train_DT, x_attack_train_DT,
                                                      y_attack_train_DT, random_state=random_state)
            else:
                x_train_DT = np.concatenate([x_norm_train_DT, x_attack_train_DT])
                y_train_DT = np.concatenate([y_norm_train_DT, y_attack_train_DT])
                print(f'Without data balance, x_train.shape: {x_train_DT.shape}')
                print(
                    f' in which, x_norm_train_DT.shape: {x_norm_train_DT.shape}, and x_attack_train_DT.shape: {x_attack_train_DT.shape}')
            # ### without val set for DT
            # x_val = np.concatenate([x_norm_val, x_attack_val])
            # x_val_labels = np.concatenate([y_val, y_attack_val])
            # unb test set
            print('\n-Step 3-2. train DT...')
            train_DT(x_train_DT, y_train_DT, case)

            print('\n-Step 3-3. evaluate DT...')
            print('evaluate DT on train set')
            print(f'x_train_DT.shape: {x_train_DT.shape}')
            test_DT(x_train_DT, y_train_DT, case)

            print('1) evaluate DT on test set 1')
            x_test_1_DT = np.concatenate([x_norm_test_DT, x_attack_test_DT])
            y_test_1_DT = np.concatenate([y_norm_test_DT, y_attack_test_DT])
            print(f'x_test_1_DT.shape: {x_test_1_DT.shape}')
            test_DT(x_test_1_DT, y_test_1_DT, case)

            print('2) evaluate DT on test set 2')
            # test set 2
            x_test_2_DT = np.concatenate([x_norm_test_2_DT, x_attack_test_2_DT])
            y_test_2_DT = np.concatenate([y_norm_test_2_DT, y_attack_test_2_DT])
            print(f'x_test_2_DT.shape: {x_test_2_DT.shape}')
            test_DT(x_test_2_DT, y_test_2_DT, case)


if __name__ == '__main__':

    out_file = 'stdout_content.txt'
    # sys.stdout = open(out_file, mode='w', buffering=1)
    ###skip 'buffering' if you don't want the output to be flushed right away after written
    # sys.stdout = sys.__stdout__

    # open the file
    with open('log.txt', 'w') as f:
        # how to implement that saving console content to the file? need to be done

        ### parameters
        Epochs = 20  # when Epochs = 300, the better optimal_thres_AE=0.8~1.2
        find_optimal_thres_flg = False  # false: use the given optimal_thres_AE, otherwise, find the optimal one by using training loss.
        optimal_thres_AE = 0.95  # the val will be changed from train, training_loss*factor: factor =10,
        factor = 35

        flg = 0
        if flg == 0:  # for demo
            cases = ['uSc1C2_z-score']
            # cases = ['uSc1C1_z-score_20_14', 'uSc1C2_z-score_20_14', 'uSc1C3_z-score_20_14']  # Experiment 1
            main(Epochs, optimal_thres_AE, find_optimal_thres_flg, cases, factor)
        elif flg == 1:
            cases = ['uSc1C1_z-score_20_14', 'uSc1C2_z-score_20_14', 'uSc1C3_z-score_20_14']  # Experiment 1, 2, 3
            main(Epochs, optimal_thres_AE, find_optimal_thres_flg, cases, factor, sub_features_flg=True,
                 sub_features_lst=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 25])

        else:
            # cases = ['uSc1C1_z-score_20_14', 'uSc1C2_z-score_20_14', 'uSc1C3_z-score_20_14']  # Experiment 1
            # cases = ['uSc2C2_z-score_20_14' ]  # Experiment 2
            # cases = ['uSc3C1_z-score_20_14' ] # Experiment 3
            cases = ['uSc1C1_z-score_20_14', 'uSc1C2_z-score_20_14', 'uSc1C3_z-score_20_14', 'uSc2C2_z-score_20_14',
                     'uSc3C1_z-score_20_14', 'uSc3C2_z-score_20_14']

            main(Epochs, optimal_thres_AE, find_optimal_thres_flg, cases, factor)
