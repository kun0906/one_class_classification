import pandas as pd
import numpy as np
from utils.utils import normalise_data

file_normal = '/Users/mycomputer/Documents/Work/HSNL/Dataset/Public_Dataset/Log_normal.txt'
file_attack_train = '/Users/mycomputer/Documents/Work/HSNL/Dataset/Simulated_DataSet/Sess_DDoS_Excessive_GET_POST.txt'
file_attack_test = '/Users/mycomputer/Documents/Work/HSNL/Dataset/Public_Dataset/Log_doshulk.txt'


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

        # features to normalise
        features_to_normalise = self.rel_features[:24] + self.rel_features[25:]  # Not normalise the feature  'isNew'

        # For autoencoder training
        self.auto_train = self.data_normal[:4200][self.rel_features]
        self.auto_train, mean, std = normalise_data(self.auto_train, features_to_normalise)

        # For decision tree training
        self.dt_normal_train = self.data_normal[4200:8400][self.rel_features]
        self.dt_normal_train, _, _ = normalise_data(self.dt_normal_train, features_to_normalise, mean, std)

        self.dt_attack_train = self.attack_train[:8400]
        self.dt_attack_train, _, _ = normalise_data(self.dt_attack_train, features_to_normalise, mean, std)

        # Create Labels for decision tree training
        self.normal_train_label = np.zeros(shape=[len(self.dt_normal_train), 1])
        self.attack_train_label = np.ones(shape=[len(self.dt_attack_train), 1])

        # For final testing
        self.attack_test = pd.read_csv(file_attack_test)[self.rel_features]
        self.normal_test = self.data_normal[8400:][self.rel_features]

        self.attack_test, _, _ = normalise_data(self.attack_test, features_to_normalise, mean, std)
        self.normal_test, _, _ = normalise_data(self.normal_test, features_to_normalise, mean, std)

        self.normal_test_label = np.zeros(shape=[len(self.normal_test), 1])
        self.attack_test_label = np.ones(shape=[len(self.attack_test), 1])

        # converting datasets as ndarray for later
        self.auto_train = np.array(self.auto_train)

        self.dt_normal_train = np.array(self.dt_normal_train)
        self.dt_attack_train = np.array(self.dt_attack_train)

        self.attack_test = np.array(self.attack_test)
        self.normal_test = np.array(self.normal_test)
