"""
    Autoencoder class

"""
import time

from tensorflow.python.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
import os
import numpy as np
from collections import Counter

class Autoencoder():

    # Create the autoencoder for training and testing
    def __init__(self, feat_size):
        pass


def create_autoencoder(in_dim=''):
    """

    :param feat_size: input_size
    :return:
    """
    # declaring the schematic of the autoencoder
    # inp_dim = int(in_dim)
    h1 = 24
    h2 = 16

    model = Sequential()
    model.add(Dense(h1, input_dim=in_dim))
    model.add(LeakyReLU())
    # model.add(Dropout(0.1))
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
def distance(predictions, labels, axis=1):
    """

    :param predictions: y_preds
    :param labels:  y_true
    :return:
    """
    distance = np.sqrt(np.sum(np.square(predictions - labels), axis=axis))  # for each y, sum all value to one.
    # distance = np.sqrt(np.sum(np.square(predictions - labels)) / len(labels))

    return distance


# To train the autoencoder model
def train_AE(train_set='', val_set= '', experiment='', Epochs=2, batch_size=32, output_dir='output_data', verbose=1):
    """
        only use normal samples to train AE.
    :param train_data:
    :param val_data:
    :param experiment:
    :param feat_size:
    :param Epochs:
    :return:
    """
    x_train = train_set['x']
    x_val = val_set['x']
    num_features = x_train.shape[1]
    model = create_autoencoder(in_dim = num_features)
    model.compile(loss=euclidean_distance_loss, optimizer='adam')

    st = time.time()
    training_results = model.fit(x_train, x_train, validation_data=(x_val, x_val), epochs=Epochs,
                                 batch_size=batch_size,
                                 verbose=0)
    print('AE training for experiment(', experiment, '), time: ', time.time() - st, 's')

    score = model.evaluate(x_train, x_train, batch_size=batch_size, verbose=0)
    print("Train Loss = ", score, ', model.metrics_names', model.metrics_names)

    score = model.evaluate(x_val, x_val, batch_size=batch_size, verbose=0)
    print("Validation Loss = ", score, 'model.metrics_names', model.metrics_names)

    # # To save the weights of the model
    # if feat_size < 25:
    #     model.save_weights("models_dumping/corr_AE_" + str(feat_size) + experiment + ".hdf5")
    # else:
    #     model.save_weights("models_dumping/new_AE_" + experiment + ".hdf5")
    output_file = os.path.join(output_dir, "models_dumping/AE_" + str(num_features) + '_' + experiment + ".hdf5")
    model.save_weights(output_file)

    print(f'training_results:{training_results}')

    return training_results.history

# TO test the model
def test_AE(test_set='', experiment='', thres_AE=1.5, output_dir='output_data'):
    """

    :param data_test:
    :param data_test_labels:
    :param experiment:
    :param feat_size:
    :param thres_AE:
    :return:
    """
    x_test = test_set['x']
    num_features = x_test.shape[1]
    y_test = test_set['y']

    model = create_autoencoder(in_dim=num_features)
    model.compile(loss=euclidean_distance_loss, optimizer='adam')

    # # Load weights
    # if feat_size < 25:
    #     model.load_weights("models_dumping/corr_AE_" + str(feat_size) + experiment + ".hdf5")
    # else:
    #     model.load_weights("models_dumping/new_AE_" + experiment + ".hdf5")
    output_file = os.path.join(output_dir,"models_dumping/AE_" + str(num_features) + '_' + experiment + ".hdf5")
    model.load_weights(output_file)

    print(f'x_test.shape: {x_test.shape}, {Counter(y_test.reshape(-1,))}, experiment: {experiment}')
    # print(f'optimal_thres_AE {optimal_thres_AE} achieved from key={key}, factor = {factor}, key = {key}')
    print(f'--test AE on test set for {experiment} with optimal_thres_AE={thres_AE}')

    st = time.time()
    data_preds = model.predict(x_test)  # data_preds equals to input data
    print("AE Test time for ", experiment, " = : ", time.time() - st)

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

    pred_arr = distance(data_preds, x_test, axis=1)
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
    probs = np.zeros([x_test.shape[0], 2])
    # pred_arr_train = distance(model.predict(data_train) , data_train, axis=1)
    scaler = MinMaxScaler().fit(pred_arr.reshape(-1, 1))
    probs[:, 1] = 1 - scaler.transform(pred_arr.reshape(-1, 1)).ravel().clip(0, 1)  # normlize to [0,1]
    probs[:, 0] = 1 - probs[:, 1]
    y_pred_probs_AE = probs[:, 1]

    pred_arr = np.reshape(pred_arr, (pred_arr.shape[0], 1))
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

