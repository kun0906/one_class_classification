"""
    Autoencoder class

"""
import os
import time
from collections import Counter

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *

from algorithms.detector import BaseDetector
from preprocess.normalization import calucalate_metrics, get_optimal_thres
from utils.balance import concat_path
from utils.visualization import plot_loss


class AutoencoderClass(BaseDetector):

    # Create the autoencoder for training and testing
    def __init__(self, in_dim=''):
        super(AutoencoderClass, self).__init__()
        self.alg_name = 'Autoencoder detector'

        def generate_model(in_dim='', h_dim='', latent_dim=''):
            print(f'in_dim:{in_dim}, h_dim:{in_dim}, latent_dim:{latent_dim}')

            # declaring the schematic of the autoencoder
            model = Sequential()
            model.add(Dense(h_dim, input_dim=in_dim))
            model.add(LeakyReLU())
            # model.add(Dropout(0.1))
            # after the first layer, you don't need to specify
            # the size of the input anymore:
            model.add(Dense(latent_dim))  # latent layer
            model.add(LeakyReLU())
            # model.add(Dropout(0.1))
            model.add(Dense(h_dim))
            model.add(LeakyReLU())
            model.add(Dense(in_dim))
            model.add(LeakyReLU())

            return model

        h_dim = self.AE_params_dict['h_dim']
        latent_dim = self.AE_params_dict['latent_dim']
        self.model = generate_model(in_dim=in_dim, h_dim=h_dim, latent_dim=latent_dim)
        self.model.compile(loss=euclidean_distance_loss, optimizer='adam')

        self.thres_AE = self.AE_params_dict['optimal_AE_thres']
        self.batch_size = self.AE_params_dict['batch_size']
        self.Epochs = self.AE_params_dict['Epochs']

        output_file = os.path.join(self.output_dir, 'models_dumping')
        self.model_file = os.path.join(output_file, f'{self.alg_name}.h5')  # creates a HDF5 file 'my_model.h5'

    def train(self, X, y=None, train_set_name='', val_set_dict=None):
        X_train = X
        y_train = y
        X_val = val_set_dict['X']
        y_val = val_set_dict['y']
        try:
            print(f'X_train.shape:{X_train.shape}, y_train:{Counter(y_train.reshape(-1,))},\n'
                  f'X_val.shape:{X_val.shape}, y_val:{Counter(y_val.reshape(-1,))}')
        except:
            print(f'X_train.shape:{X_train.shape}, y_train:{y_train},\n'
                  f'X_val.shape:{X_val.shape}, y_val:{y_val}')

        verbose = 0
        train_results = self.model.fit(x=X_train, y=X_train, validation_data=(X_val, X_val), epochs=self.Epochs,
                                       batch_size=self.batch_size, verbose=verbose)
        train_loss_dict = train_results.history
        print(f'train and val losses: {train_loss_dict}')

        if self.show_flg:
            train_loss_file = concat_path(self.output_dir + '/figures', f'{train_set_name}_train_loss.pdf')
            plot_loss(train_loss=train_loss_dict['loss'], val_loss=train_loss_dict['val_loss'],
                      out_file=train_loss_file,
                      title_flg=self.title_flg, title=f'{self.model_file}, key:')

    def test(self, X, y=None, test_set_name=''):

        X_test = X
        y_test = y
        print(f'X_test.shape:{X_test.shape}, y_test:{Counter(y_test.reshape(-1,))}')

        X_pred = self.model.predict(x=X_test, batch_size=self.batch_size)  # data_preds equals to input data
        # pred_arr_0 = distance(y_pred=X_pred, y_true=X_test, axis=1)   # euclidean_distance_loss.
        pred_error_arr = K.eval(
            euclidean_distance_loss(y_true=X_test, y_pred=X_pred, axis=1))  # calculated by columuns when asix = 1
        # assert pred_arr == pred_arr_0  # a.all() or a.any()

        self.y_pred = np.zeros(shape=(len(y_test),))
        self.y_pred[pred_error_arr < self.thres_AE] = 1  # predicted label.  1 means normal in our code.

        ### change the value to probability, use min-max method.
        probs = np.zeros([X_test.shape[0], 2])
        scaler = MinMaxScaler().fit(pred_error_arr.reshape(-1, 1))
        value = scaler.transform(pred_error_arr.reshape(-1, 1)).ravel().clip(0,
                                                                             1)  # normlize to [0,1]. Smaller the value is, more probability the normal is
        probs[:, 1] = 1 - value  # the probability of the sample are predicted as normal
        probs[:, 0] = 1 - probs[:, 1]
        self.y_proba = probs[:, 1]  # the probability of the sample are predicted as normal

        ###  reconstr_erros_arr is used for plot figures of reconstr_errors of normal and attack sampels.
        self.reconstr_errors_arr = np.concatenate([y_test.reshape(-1, 1), pred_error_arr.reshape(-1, 1)],
                                                  axis=1)  # concatenate by columns

        calucalate_metrics(y_true=y_test,
                           y_pred=self.y_pred)  # call confusion_matrix(y_true=y_test, y_pred=self.y_pred)

        # if self.show_flg:  # show reconstruction errors of normal and attack samples in test set
        #     out_file = os.path.join(self.output_dir,
        #                             f'figures/{experiment_str}={key}+recon_err={str(x_test.shape[1])}_features.txt')
        #     out_file = save_reconstruction_errors(self.reconstr_errs_arr, experiment_str, X_test.shape[1], out_file)
        #     print(f'out_file:{out_file}')
        #     title = key
        #     if 'mawi' not in experiment_str.lower():  # for mawi, it does not has attack samples.
        #         plot_reconstruction_errors_from_txt(input_file=out_file, balance_data_flg=False,
        #                                             random_state=self.random_state,
        #                                             title_flg=self.title_flg, title=title)

    def test_different_thres(self, X, y=None, test_set_name='', thres_lst=[]):

        X_test = X
        y_test = y
        print(f'X_test.shape:{X_test.shape}, y_test:{Counter(y_test.reshape(-1,))}')
        res_metrics_lst = {'tpr': [], 'fnr': [], 'fpr': [], 'tnr': [], 'acc': []}
        for i, thres in enumerate(thres_lst):
            print(f'idx: {i} AE thres: {thres}')

            # print(f'thres: {thres}')
            X_pred = self.model.predict(x=X_test, batch_size=self.batch_size)  # data_preds equals to input data
            # pred_arr_0 = distance(y_pred=X_pred, y_true=X_test, axis=1)   # euclidean_distance_loss.
            pred_error_arr = K.eval(
                euclidean_distance_loss(y_true=X_test, y_pred=X_pred, axis=1))  # calculated by columuns when asix = 1
            # assert pred_arr == pred_arr_0  # a.all() or a.any()

            y_pred = np.zeros(shape=(len(y_test),))
            y_pred[pred_error_arr < thres] = 1  # predicted label.  1 means normal in our code.

            # ### change the value to probability, use min-max method.
            # probs = np.zeros([X_test.shape[0], 2])
            # scaler = MinMaxScaler().fit(pred_error_arr.reshape(-1, 1))
            # value = scaler.transform(pred_error_arr.reshape(-1, 1)).ravel().clip(0,
            #                                                                      1)  # normlize to [0,1]. Smaller the value is, more probability the normal is
            # probs[:, 1] = 1 - value  # the probability of the sample are predicted as normal
            # probs[:, 0] = 1 - probs[:, 1]
            # y_proba = probs[:, 1]  # the probability of the sample are predicted as normal
            #
            # ###  reconstr_erros_arr is used for plot figures of reconstr_errors of normal and attack sampels.
            # reconstr_errors_arr = np.concatenate([y_test.reshape(-1, 1), pred_error_arr.reshape(-1, 1)],
            #                                           axis=1)  # concatenate by columns

            tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_true=y_test,
                                                         y_pred=y_pred)  # call confusion_matrix(y_true=y_test, y_pred=self.y_pred)

            res_metrics_lst['tpr'].append(float(tpr))
            res_metrics_lst['fnr'].append(float(fnr))
            res_metrics_lst['fpr'].append(float(fpr))
            res_metrics_lst['tnr'].append(float(tpr))
            res_metrics_lst['acc'].append(float(acc))
        plot_different_thres(thres_lst=thres_lst, res_metrics_lst=res_metrics_lst)


    def find_optimal_thres(self, X='', y=None, data_set_name=''):

        X_train = X
        y_train = y

        train_set_name = data_set_name

        verbose = 0
        train_results = self.model.fit(x=X_train, y=X_train, validation_data=None, epochs=self.Epochs,
                                       batch_size=self.batch_size, verbose=verbose)
        train_loss_dict = train_results.history
        print(f'train and val losses: {train_loss_dict}')

        factor_AE_thres = self.AE_params_dict['factor_AE_thres']
        optimal_AE_thres = self.AE_params_dict['optimal_AE_thres']
        find_optimal_thres_flg = self.AE_params_dict['find_optimal_thres_flg']
        if find_optimal_thres_flg:
            print(f'find the optimal threshold ({find_optimal_thres_flg}) for AE by using training losss')
            optimal_AE_thres, factor_AE_thres, key, = get_optimal_thres(train_loss_dict,
                                                                        factor_AE_thres=factor_AE_thres,
                                                                        key='loss')

            min_v = 1000000
            for i, thres_v in enumerate(
                    np.linspace(optimal_AE_thres / factor_AE_thres, optimal_AE_thres, factor_AE_thres)):
                print(f'idx: {i} --test AE on train set: \'{train_set_name}\', with optimal_AE_thres:{thres_v}')
                y_pred_label_AE, y_pred_probs_AE, _ = test_AE(X_train,
                                                              experiment='',
                                                              thres_AE=thres_v, AE_params_dict=self.AE_params_dict,
                                                              output_dir=self.output_dir)
                tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_true=y,
                                                             y_pred=y_pred_label_AE)
                if min_v > max(fnr, fpr):
                    min_v = max(fnr, fpr)
                    self.optimal_AE_thres = thres_v
        else:
            if type(optimal_AE_thres) == type(None):
                training_loss_value = train_loss_dict['loss'][-1]
                self.optimal_AE_thres = factor_AE_thres * training_loss_value
                print(
                    f'using the threshold obtained from train set (optimal_AE_thres {optimal_AE_thres} = factor '
                    f'({factor_AE_thres}) * training_loss ({training_loss_value}))')
            else:
                print(f'using the presetted threshold in params_dict[\'optimal_AE_thres\']: {self.optimal_AE_thres}.')
                key = None
                factor_AE_thres = None
        print(f'optimal_thres_AE_val:{optimal_AE_thres}, factor_AE_thres = {factor_AE_thres}, key = {key}')

    def dump_model(self, mode_file=''):
        ### overide
        if mode_file != '':
            self.model_file = mode_file
        print(f'dump mode_file: {self.model_file}')
        self.model.save(self.model_file)  # creates a HDF5 file 'my_model.h5'

    def load_model(self, mode_file=''):
        ### overide
        if mode_file != '':
            self.model_file = mode_file
        print(f'load mode_file: {self.model_file}')
        # returns a compiled model identical to the previous one
        self.model = load_model(self.model_file)


def plot_different_thres(thres_lst='', res_metrics_lst={}, output_file='_different_AE_thres.pdf', only_FPR_flg=False,
                         title_flg=True, title=''):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    if only_FPR_flg:
        ax.plot(thres_lst, res_metrics_lst['fpr'], 'r*-', label='FPR')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('Reconstruction errors')
        plt.ylabel('False Positive Rate')
        # plt.legend(loc='upper right')
        # plt.title(title)

        # sub_dir = os.path.split(input_file)[0]
        # output_pre_path = os.path.split(input_file)[-1].split('.')[0]
        # out_file = os.path.join(sub_dir, output_pre_path + '_AE_thres_for_paper.pdf')
        print(f'AE_thres: out_file:{output_file}')
        plt.savefig(output_file)  # should use before plt.show()

    else:
        ax.plot(thres_lst, res_metrics_lst['fpr'], 'r*-', label='FPR')
        ax.plot(thres_lst, res_metrics_lst['fnr'], 'b*-', label='FNR')
        # ax.plot(thres_lst, res_metrics_lst['tpr'], 'g*-', label='TPR')
        # ax.plot(thres_lst, res_metrics_lst['tnr'], 'c*-', label='TNR')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('AE Thresholds')
        plt.ylabel('Rate')
        plt.legend(loc='upper right')
        # plt.title(title)

        # sub_dir = os.path.split(input_file)[0]
        # output_pre_path = os.path.split(input_file)[-1].split('.')[0]
        # out_file = os.path.join(sub_dir, output_pre_path + '_AE_thres.pdf')
        print(f'AE_thres: out_file:{output_file}')
        plt.savefig(output_file)  # should use before plt.show()

    if title_flg:
        plt.title(title)

    plt.show()



def create_autoencoder(in_dim='', AE_params_dict=''):
    """

    :param feat_size: input_size
    :return:
    """
    # declaring the schematic of the autoencoder
    # inp_dim = int(in_dim)
    h1 = AE_params_dict['h_dim']
    h2 = AE_params_dict['latent_dim']
    print(f'in_dim:{in_dim}, h_dim:{h1}, latent_dim:{h2}')

    model = Sequential()
    model.add(Dense(h1, input_dim=in_dim))
    model.add(LeakyReLU())
    # model.add(Dropout(0.1))
    # after the first layer, you don't need to specify
    # the size of the input anymore:
    model.add(Dense(h2))  # latent layer
    model.add(LeakyReLU())
    # model.add(Dropout(0.1))
    model.add(Dense(h1))
    model.add(LeakyReLU())
    model.add(Dense(in_dim))
    model.add(LeakyReLU())

    return model


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
def distance(y_pred='predictions', y_true='labels', axis=1):
    """

    :param predictions: y_preds
    :param labels:  y_true
    :return:
    """
    distance = np.sqrt(np.sum(np.square(y_pred - y_true), axis=axis))  # for each y, sum all value to one.
    # distance = np.sqrt(np.sum(np.square(predictions - labels)) / len(labels))

    return distance


# To train the autoencoder model
def train_AE(train_set='', val_set='', experiment='', Epochs=2, batch_size=32, AE_params_dict='',
             output_dir='output_data', verbose=1):
    """
        only use normal samples to train AE.
    :param train_data:
    :param val_data:
    :param experiment:
    :param feat_size:
    :param Epochs:
    :return:
    """
    experiment_str = ''
    for key, value in experiment.items():
        experiment_str += value + '_'

    X_train = train_set['X']
    X_val = val_set['X']
    num_features = X_train.shape[1]
    model = create_autoencoder(in_dim=num_features, AE_params_dict=AE_params_dict)
    model.compile(loss=euclidean_distance_loss, optimizer='adam')

    st = time.time()
    training_results = model.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=Epochs,
                                 batch_size=batch_size,
                                 verbose=0)
    print('AE training for experiment(', experiment, '), time: ', time.time() - st, 's')

    score = model.evaluate(X_train, X_train, batch_size=batch_size, verbose=0)
    print("Train Loss = ", score, ', model.metrics_names', model.metrics_names)

    score = model.evaluate(X_val, X_val, batch_size=batch_size, verbose=0)
    print("Validation Loss = ", score, 'model.metrics_names', model.metrics_names)

    # # To save the weights of the model
    # if feat_size < 25:
    #     model.save_weights("models_dumping/corr_AE_" + str(feat_size) + experiment + ".hdf5")
    # else:
    #     model.save_weights("models_dumping/new_AE_" + experiment + ".hdf5")
    out_file = os.path.join(output_dir, "models_dumping/AE_" + str(num_features) + '_' + experiment_str + ".hdf5")
    model.save_weights(out_file)

    print(f'training_results:{training_results.history}')

    return training_results.history


# TO test the model
def test_AE(test_set='', experiment={}, thres_AE=1.5, AE_params_dict='', output_dir='output_data'):
    """

    :param data_test:
    :param data_test_labels:
    :param experiment:
    :param feat_size:
    :param thres_AE:
    :return:
    """
    experiment_str = ''
    for key, value in experiment.items():
        experiment_str += value + '_'

    X_test = test_set['X']
    num_features = X_test.shape[1]
    y_test = test_set['y']

    model = create_autoencoder(in_dim=num_features, AE_params_dict=AE_params_dict)
    model.compile(loss=euclidean_distance_loss, optimizer='adam')

    # # Load weights
    # if feat_size < 25:
    #     model.load_weights("models_dumping/corr_AE_" + str(feat_size) + experiment + ".hdf5")
    # else:
    #     model.load_weights("models_dumping/new_AE_" + experiment + ".hdf5")
    out_file = os.path.join(output_dir, "models_dumping/AE_" + str(num_features) + '_' + experiment_str + ".hdf5")
    model.load_weights(out_file)

    print(f'X_test.shape: {X_test.shape}, {Counter(y_test.reshape(-1,))}')
    # print(f'optimal_thres_AE {optimal_thres_AE} achieved from key={key}, factor = {factor}, key = {key}')
    print(f'--test AE on test set for {experiment_str} with optimal_thres_AE={thres_AE}')

    st = time.time()
    data_preds = model.predict(X_test)  # data_preds equals to input data
    print("AE Test time for ", experiment_str, " = : ", time.time() - st, 's')

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

    pred_arr = distance(data_preds, X_test, axis=1)
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
    probs = np.zeros([X_test.shape[0], 2])
    # pred_arr_train = distance(model.predict(data_train) , data_train, axis=1)
    scaler = MinMaxScaler().fit(pred_arr.reshape(-1, 1))
    probs[:, 1] = 1 - scaler.transform(pred_arr.reshape(-1, 1)).ravel().clip(0, 1)  # normlize to [0,1]
    probs[:, 0] = 1 - probs[:, 1]
    y_pred_probs_AE = probs[:, 1]

    pred_arr = np.reshape(pred_arr, (pred_arr.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    reconstr_errors_lst = np.concatenate([pred_arr, y_test], axis=1)  # concatenate by columns

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
