import time
import numpy as np
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from joblib import dump, load

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# from utils.metrics_measure import f1_score, calucalate_metrics


# To apply correlation
from preprocess.normalization import calucalate_metrics


def corr_apply(data_train, data_val, x_test_1, thresholddd):
    corr = data_train.df.corr()

    data_train = np.transpose(data_train.data)
    data_val = np.transpose(data_val)
    x_test_1 = np.transpose(x_test_1)
    # x_test_2 = np.transpose(x_test_2)

    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr.iloc[i, j]) >= thresholddd:
                if columns[j]:
                    columns[j] = False

    data_train_n, data_val_n, x_test_1_n, x_test_2_n = [], [], [], []
    for idx, keep in enumerate(columns):
        if keep == True:
            data_train_n.append(data_train[idx])
            data_val_n.append(data_val[idx])
            x_test_1_n.append(x_test_1[idx])
            # x_test_2_n.append(x_test_2[idx])

    return np.transpose(data_train_n), np.transpose(data_val_n), np.transpose(x_test_1_n)


# To train Decision tree
def train_DT(x_train='', y_train = '', experiment='', output_dir = ''):
    classifier = DecisionTreeClassifier()
    X, y = shuffle(x_train, y_train, random_state=42)

    classifier.fit(X, y)

    dump(classifier, "output_data/models_dumping/DTexperiment-" + str(experiment) + ".joblib")


def test_DT(x_test = '', y_test='', experiment='', output_dir = ''):
    clf = load("output_data/models_dumping/DTexperiment-" + str(experiment) + ".joblib")
    a = time.time()
    pred_class = clf.predict(x_test)
    pred_proba= clf.predict_proba(x_test)
    print("DT Testing Time : ", time.time() - a)

    # # test_label.reshape([test_label.shape[0],])
    # conf = confusion_matrix(test_label, pred_class)
    #
    # # print("F1 Score : ", f1_score(conf))
    # # print("FPR : ", (conf[1, 0] / (conf[1, 0] + conf[1, 1])))
    # # print("DR : ", (conf[0, 0] / (conf[0, 0] + conf[0, 1])))
    # # print(confusion_matrix(test_label, pred_class))
    # # print(classification_report(test_label, pred_class))

    calucalate_metrics(y_true=y_test, y_pred=pred_class)

    return pred_class, pred_proba


# TrainPCA
def train_PCA(x_train='', experiment='', n_components=14):
    a = time.time()
    pca = PCA(n_components=x_train.shape[1])
    pca = pca.fit(x_train)
    print("PCA Training time for experiment", experiment, " = ", time.time() - a)
    dump(pca, 'output_data/models_dumping/PCA_' + experiment + '.joblib')


def test_PCA(x_test= '', y_test='', experiment=''):
    model = load('output_data/models_dumping/PCA_' + experiment + '.joblib')
    a = time.time()
    data_pred = model.predict(
        x_test)  # 0 stands for inliers and 1 for outliers in pca . #  it should be considered as an outlier according to the fitted model. 0 stands for inliers and 1 for outliers.
    print("PCA Testing Time : ", time.time() - a)
    data_pred_labels = 1 - data_pred  # our defination of attack is 0, normal is 1.
    data_pred_probs = 1 - model.predict_proba(x_test)[:, 1]  # return the class 1 (normal) probability.
    # data_pred_probs = model.decision_function(x_test)
    # print('PCA experiment :  ', experiment, '\n',confusion_matrix(y_test, data_pred))
    # print('PCA experiment :  ', experiment, '\n',classification_report(y_test, data_pred))
    #
    # conf = confusion_matrix(y_test, data_pred)
    #
    # # print(classification_report(y_test,pred_class))
    #
    # print("F1 Score : ", f1_score(conf))
    # print("FPR : ", (conf[1, 0] / (conf[1, 0] + conf[1, 1])))
    # print("DR : ", (conf[0, 0] / (conf[0, 0] + conf[0, 1])))
    # try:
    #     print(conf)
    # except:
    #     print("FPR = 100")

    calucalate_metrics(y_test, data_pred_labels)

    # data_pred = model.decision_function(x_test)

    return data_pred_labels, data_pred_probs


# Train Isolation Forest
def train_IF(x_train='', experiment=''):
    clf = IsolationForest()
    a = time.time()
    clf.fit(x_train)
    print("IF Training time for experiment", experiment, " = ", time.time() - a)
    dump(clf, 'output_data/models_dumping/IF_' + experiment + '.joblib')


def test_IF(x_test='', y_test='', experiment=''):
    model = load('output_data/models_dumping/IF_' + experiment + '.joblib')
    a = time.time()
    data_pred = model.predict(x_test)
    print("IF Testing Time : ", time.time() - a)
    data_pred_labels = (data_pred + 1) / 2

    # data_pred_probs = model.predict_proba(x_test)[:, 1]  # return the class 1 probability.
    probs = np.zeros([x_test.shape[0], 2])
    pred_arr = model.decision_function(x_test)
    scaler = MinMaxScaler().fit(pred_arr.reshape(-1, 1))
    probs[:, 1] = scaler.transform(pred_arr.reshape(-1, 1)).ravel().clip(0, 1)  # normlize to [0,1], normal
    probs[:, 0] = 1 - probs[:, 1]  # attack
    data_pred_probs = probs[:, 1]

    # print('IF experiment :  ', experiment, '\n',confusion_matrix(y_test, data_pred))
    # print('IF experiment :  ', experiment, '\n',classification_report(y_test, data_pred))

    # conf = confusion_matrix(y_test, data_pred)
    #
    # # print(classification_report(y_test,pred_class))
    #
    # print("F1 Score : ", f1_score(conf))
    # print("FPR : ", (conf[1, 0] / (conf[1, 0] + conf[1, 1])))
    # print("DR : ", (conf[0, 0] / (conf[0, 0] + conf[0, 1])))
    # try:
    #     print(conf)
    # except:
    #     print("FPR = 100")

    calucalate_metrics(y_test, data_pred_labels)

    # data_pred = model.decision_function(x_test)

    return data_pred_labels, data_pred_probs


# TrainOCSVM
def train_OCSVM(x_train='', experiment=''):
    ocsvm = OCSVM(verbose=False)
    a = time.time()
    ocsvm = ocsvm.fit(x_train)
    print("SVM Training time for experiment", experiment, " = ", time.time() - a)
    dump(ocsvm, 'output_data/models_dumping/OCSVM_' + experiment + '.joblib')


def test_OCSVM(x_test='', y_test='', experiment=''):
    model = load('output_data/models_dumping/OCSVM_' + experiment + '.joblib')
    a = time.time()
    data_pred = model.predict(x_test)
    print("OCSVM Testing Time : ", time.time() - a)
    data_pred_labels = 1 - data_pred  # in our work, attack = 0, normal =1

    probs = np.zeros([x_test.shape[0], 2])
    pred_arr = model.decision_function(
        x_test)  # Signed distance is positive for an inlier and negative for an outlier.
    print(f'pred_arr:{pred_arr[:5]}, data_tes_labels:{data_pred_labels[:5]}')
    scaler = MinMaxScaler().fit(pred_arr.reshape(-1, 1))
    probs[:, 1] = 1 - scaler.transform(pred_arr.reshape(-1, 1)).ravel().clip(0, 1)  # normlize to [0,1], normal
    probs[:, 0] = 1 - probs[:, 1]  # attack
    data_pred_probs = probs[:, 1]

    # data_pred_probs = model.decision_function(x_test)
    # data_pred_probs = 1-model.predict_proba(x_test)[:, 1]  # return the class 1 probability.

    # print('OCSVM experiment :  ', experiment, '\n',confusion_matrix(y_test, data_pred))
    # print('OCSVM experiment :  ', experiment, '\n',classification_report(y_test, data_pred))

    # conf = confusion_matrix(y_test, data_pred)
    #
    # # print(classification_report(y_test,pred_class))
    #
    # print("F1 Score : ", f1_score(conf))
    # print("FPR : ", (conf[1, 0] / (conf[1, 0] + conf[1, 1])))
    # print("DR : ", (conf[0, 0] / (conf[0, 0] + conf[0, 1])))
    # try:
    #     print(conf)
    # except:
    #     print("FPR = 100")

    calucalate_metrics(y_test, data_pred_labels)

    # data_pred = model.decision_function(x_test)

    return data_pred_labels, data_pred_probs
