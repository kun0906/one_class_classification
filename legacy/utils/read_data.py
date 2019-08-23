import pandas as pd
import numpy as np
from utils.utils import normalise_data

file_normal = '/Users/mycomputer/Documents/Work/HSNL/dataset/public_dataset/Log_normal.txt'
file_attack_train = '/Users/mycomputer/Documents/Work/HSNL/dataset/synthetic_dataset/Sess_DDoS_Excessive_GET_POST.txt'
file_attack_test = '/Users/mycomputer/Documents/Work/HSNL/dataset/public_dataset/Log_doshulk.txt'


class Dataset:
    '''
    Class for loading the dataset and further preprocessing and return a clean, normalised dataset as a numpy array
    '''

    def __init__(self, normalise=True):
        '''
        load the data files as panda DataFrames and set variables
        '''

        self.data_normal = pd.read_csv(file_normal)
        feature_names = list(self.data_normal.columns)
        self.rel_features = feature_names[5:]

        self.attack_train = pd.read_csv(file_attack_train)[self.rel_features]

        # Indexed train and test datasets
        self.attack_test = pd.read_csv(file_attack_test)[self.rel_features]
        self.attack_train = self.attack_train[:8400]
        self.normal_train = self.data_normal[:8400][self.rel_features]
        self.normal_test = self.data_normal[8400:][self.rel_features]

        self.data_train = pd.concat([self.normal_train, self.attack_train])

        # features to normalise
        features_to_normalise = self.rel_features[:24] + self.rel_features[25:]  # Not normalise the feature  'isNew'

        # Normalising data using Z-score
        if normalise == True and len(features_to_normalise) != 0:
            self.data_train, mean, std = normalise_data(self.data_train, features_to_normalise)

            # normalizing test data
            self.normal_test, _, _ = normalise_data(self.normal_test, features_to_normalise, mean, std)
            self.attack_test, _, _ = normalise_data(self.attack_test, features_to_normalise, mean, std)

        # create labels
        self.normal_train_label = np.zeros(shape=[len(self.normal_train), 1])
        self.attack_train_label = np.ones(shape=[len(self.attack_train), 1])

        self.normal_test_label = np.zeros(shape=[len(self.normal_test), 1])
        self.attack_test_label = np.ones(shape=[len(self.attack_test), 1])

        self.data_train_label = np.concatenate([self.normal_train_label, self.attack_train_label])

        # converting datasets as ndarray for later
        self.data_train = np.array(self.data_train)
        self.normal_test = np.array(self.normal_test)
        self.attack_test = np.array(self.attack_test)
        self.data_test = np.concatenate([self.normal_test, self.attack_test])
        self.data_test_label = np.concatenate([self.normal_test_label, self.attack_test_label])
