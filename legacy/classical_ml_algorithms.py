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

from utils.metrics_measure import f1_score, calucalate_metrics


# To apply correlation
def corr_apply(data_train, data_val, data_test_1, thresholddd):
    corr = data_train.df.corr()

    data_train = np.transpose(data_train.data)
    data_val = np.transpose(data_val)
    data_test_1 = np.transpose(data_test_1)
    # data_test_2 = np.transpose(data_test_2)

    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr.iloc[i, j]) >= thresholddd:
                if columns[j]:
                    columns[j] = False

    data_train_n, data_val_n, data_test_1_n, data_test_2_n = [], [], [], []
    for idx, keep in enumerate(columns):
        if keep == True:
            data_train_n.append(data_train[idx])
            data_val_n.append(data_val[idx])
            data_test_1_n.append(data_test_1[idx])
            # data_test_2_n.append(data_test_2[idx])

    return np.transpose(data_train_n), np.transpose(data_val_n), np.transpose(data_test_1_n)


# To train Decision tree
def train_DT(train_data, train_label, case):
    classifier = DecisionTreeClassifier()
    X, y = shuffle(train_data, train_label, random_state=42)

    classifier.fit(X, y)

    dump(classifier, "models_dumping/DTcase-" + str(case) + ".joblib")


def test_DT(test_data, test_label, case):
    clf = load("models_dumping/DTcase-" + str(case) + ".joblib")
    a = time.time()
    pred_class = clf.predict(test_data)
    print("DT Testing Time : ", time.time() - a)

    # # test_label.reshape([test_label.shape[0],])
    # conf = confusion_matrix(test_label, pred_class)
    #
    # # print("F1 Score : ", f1_score(conf))
    # # print("FPR : ", (conf[1, 0] / (conf[1, 0] + conf[1, 1])))
    # # print("DR : ", (conf[0, 0] / (conf[0, 0] + conf[0, 1])))
    # # print(confusion_matrix(test_label, pred_class))
    # # print(classification_report(test_label, pred_class))

    calucalate_metrics(test_label, pred_class)

    return pred_class


# TrainPCA
def train_PCA(train_data, case, n_components=14):
    a = time.time()
    pca = PCA(n_components=train_data.shape[1])
    pca = pca.fit(train_data)
    print("PCA Training time for case", case, " = ", time.time() - a)
    dump(pca, 'models_dumping/PCA_' + case + '.joblib')


def test_PCA(data_test, data_test_labels, case):
    model = load('models_dumping/PCA_' + case + '.joblib')
    a = time.time()
    data_pred = model.predict(
        data_test)  # 0 stands for inliers and 1 for outliers in pca . #  it should be considered as an outlier according to the fitted model. 0 stands for inliers and 1 for outliers.
    print("PCA Testing Time : ", time.time() - a)
    data_pred_labels = 1 - data_pred  # our defination of attack is 0, normal is 1.
    data_pred_probs = 1 - model.predict_proba(data_test)[:, 1]  # return the class 1 (normal) probability.
    # data_pred_probs = model.decision_function(data_test)
    # print('PCA Case :  ', case, '\n',confusion_matrix(data_test_labels, data_pred))
    # print('PCA Case :  ', case, '\n',classification_report(data_test_labels, data_pred))
    #
    # conf = confusion_matrix(data_test_labels, data_pred)
    #
    # # print(classification_report(data_test_labels,pred_class))
    #
    # print("F1 Score : ", f1_score(conf))
    # print("FPR : ", (conf[1, 0] / (conf[1, 0] + conf[1, 1])))
    # print("DR : ", (conf[0, 0] / (conf[0, 0] + conf[0, 1])))
    # try:
    #     print(conf)
    # except:
    #     print("FPR = 100")

    calucalate_metrics(data_test_labels, data_pred_labels)

    # data_pred = model.decision_function(data_test)

    return data_pred_labels, data_pred_probs


# Train Isolation Forest
def train_IF(train_data, case):
    clf = IsolationForest()
    a = time.time()
    clf.fit(train_data)
    print("IF Training time for case", case, " = ", time.time() - a)
    dump(clf, 'models_dumping/IF_' + case + '.joblib')


def test_IF(data_test, data_test_labels, case):
    model = load('models_dumping/IF_' + case + '.joblib')
    a = time.time()
    data_pred = model.predict(data_test)
    print("IF Testing Time : ", time.time() - a)
    data_pred_labels = (data_pred + 1) / 2

    # data_pred_probs = model.predict_proba(data_test)[:, 1]  # return the class 1 probability.
    probs = np.zeros([data_test.shape[0], 2])
    pred_arr = model.decision_function(data_test)
    scaler = MinMaxScaler().fit(pred_arr.reshape(-1, 1))
    probs[:, 1] = scaler.transform(pred_arr.reshape(-1, 1)).ravel().clip(0, 1)  # normlize to [0,1], normal
    probs[:, 0] = 1 - probs[:, 1]  # attack
    data_pred_probs = probs[:, 1]

    # print('IF Case :  ', case, '\n',confusion_matrix(data_test_labels, data_pred))
    # print('IF Case :  ', case, '\n',classification_report(data_test_labels, data_pred))

    # conf = confusion_matrix(data_test_labels, data_pred)
    #
    # # print(classification_report(data_test_labels,pred_class))
    #
    # print("F1 Score : ", f1_score(conf))
    # print("FPR : ", (conf[1, 0] / (conf[1, 0] + conf[1, 1])))
    # print("DR : ", (conf[0, 0] / (conf[0, 0] + conf[0, 1])))
    # try:
    #     print(conf)
    # except:
    #     print("FPR = 100")

    calucalate_metrics(data_test_labels, data_pred_labels)

    # data_pred = model.decision_function(data_test)

    return data_pred_labels, data_pred_probs


# TrainOCSVM
def train_OCSVM(train_data, case):
    ocsvm = OCSVM(verbose=False)
    a = time.time()
    ocsvm = ocsvm.fit(train_data)
    print("SVM Training time for case", case, " = ", time.time() - a)
    dump(ocsvm, 'models_dumping/OCSVM_' + case + '.joblib')


def test_OCSVM(data_test, data_test_labels, case):
    model = load('models_dumping/OCSVM_' + case + '.joblib')
    a = time.time()
    data_pred = model.predict(data_test)
    print("OCSVM Testing Time : ", time.time() - a)
    data_pred_labels = 1 - data_pred  # in our work, attack = 0, normal =1

    probs = np.zeros([data_test.shape[0], 2])
    pred_arr = model.decision_function(
        data_test)  # Signed distance is positive for an inlier and negative for an outlier.
    print(f'pred_arr:{pred_arr[:5]}, data_tes_labels:{data_pred_labels[:5]}')
    scaler = MinMaxScaler().fit(pred_arr.reshape(-1, 1))
    probs[:, 1] = 1 - scaler.transform(pred_arr.reshape(-1, 1)).ravel().clip(0, 1)  # normlize to [0,1], normal
    probs[:, 0] = 1 - probs[:, 1]  # attack
    data_pred_probs = probs[:, 1]

    # data_pred_probs = model.decision_function(data_test)
    # data_pred_probs = 1-model.predict_proba(data_test)[:, 1]  # return the class 1 probability.

    # print('OCSVM Case :  ', case, '\n',confusion_matrix(data_test_labels, data_pred))
    # print('OCSVM Case :  ', case, '\n',classification_report(data_test_labels, data_pred))

    # conf = confusion_matrix(data_test_labels, data_pred)
    #
    # # print(classification_report(data_test_labels,pred_class))
    #
    # print("F1 Score : ", f1_score(conf))
    # print("FPR : ", (conf[1, 0] / (conf[1, 0] + conf[1, 1])))
    # print("DR : ", (conf[0, 0] / (conf[0, 0] + conf[0, 1])))
    # try:
    #     print(conf)
    # except:
    #     print("FPR = 100")

    calucalate_metrics(data_test_labels, data_pred_labels)

    # data_pred = model.decision_function(data_test)

    return data_pred_labels, data_pred_probs
