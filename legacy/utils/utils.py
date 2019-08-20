import numpy as np
import math

from sklearn.model_selection import train_test_split


def zscore(x, m, sigma):
    '''

    :param x: an array or Series (containing float or int)
    :return: normalised array or Series(x - x.mean/x.std())
    '''
    if m > 0:
        mean = m
    else:
        mean = x.mean()
    if sigma > 0:
        std = sigma
    else:
        std = np.std(x)
    if std < 0.000000001:
        std = math.exp(-8)
    x_normal = (x - mean) / std

    return x_normal, mean, std


#
# def min_max(x, m, sigma):
#     '''
#
#     :param x: an array or Series (containing float or int)
#     :return: normalised array or Series(x - x.mean/x.std())
#     '''
#     if m > 0:
#         mean = m
#     else:
#         mean = np.amin(x)
#     if sigma > 0:
#         std = sigma
#     else:
#         std = np.amax(x)
#     x_normal = ((x - mean) / (std - mean))
#     return x_normal, mean, std


def normalise_data(X, features, m=[], sigma=[]):
    '''

    :param X: Pandas DataFrame consisting of columns to normalised
    :param features: list of column names to be normalised
    :return: DataFrame with the required columns normalized
    '''
    mean = m
    std = sigma
    new_mean = []
    new_std = []
    if len(mean) == 0 and len(std) == 0:
        for feature in features:
            X[feature], _mean, _std = zscore(X[feature], 0, 0)
            new_mean.append(_mean)
            new_std.append(_std)
        return X, new_mean, new_std
    else:
        for i in range(len(features)):
            X[features[i]], _, _ = zscore(X[features[i]], mean[i], std[i])
        return X, mean, std

#
# def train_val_test_split(X, train_frac=0.6, val_frac=0.4, test_frac=0.1, shuffle=False):
#     '''
#
#     :param X: Pandas Dataframe or Numpy array Dataset
#     :param train_frac: fraction of train split
#     :param val_frac:  fraction of validation split
#     :param test_frac: fraction of test split
#     :param shuffle: Flag to shuffle the data before the split
#     :return: Pandas Dataframe or Numpy array train, validation and test set of the input
#     '''
#
#     X_train, X_val = train_test_split(X, test_size=val_frac + test_frac)
#
#     val_frac = val_frac / (val_frac + test_frac)
#     if val_frac < 1.0:
#         X_val, X_test = train_test_split(X_val, train_size=val_frac)
#     else:
#         X_test = []
#
#     return X_train, X_val, X_test
