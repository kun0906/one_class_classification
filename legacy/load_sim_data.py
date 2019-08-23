import pandas as pd
import numpy as np
from utils.utils import normalise_data, train_val_test_split

# TODO - Add files to config files
# files to pick data from

file0 = '/Users/mycomputer/Documents/HSNL/hsnl_data/Sess_normal_0.txt'
file1 = '/Users/mycomputer/Documents/HSNL/hsnl_data/sess_DDoS_Excessive_GET_POST'
file2 = '/Users/mycomputer/Documents/HSNL/hsnl_data/sess_DDoS_Slow_POST_Two_Arm_HTTP_server_wait'

file3 = '/Users/mycomputer/Documents/HSNL/hsnl_data/sess_DDoS_Recursive_GET'
file4 = '/Users/mycomputer/Documents/HSNL/hsnl_data/sess_DDoS_SlowLoris_Two_Arm_HTTP_server_wait'

files = [file0, file1, file2, file3, file4]


class Dataset:
    '''
    Class for loading the dataset and further preprocessing and return a clean, normalised dataset as a numpy array
    '''

    def __init__(self, normalise=True, train_size=0.7, val_size=0.3, test_size=0.0, shuffle=False):
        '''
        load the data files as panda DataFrames and set variables
        '''

        self.data_normal = pd.read_csv('/Users/mycomputer/Documents/HSNL/hsnl_data/Sess_normal_0.txt')
        feature_names = list(self.data_normal.columns)
        self.rel_features = feature_names[5:]

        self.data_attack_1 = pd.read_csv(files[1])[self.rel_features]
        self.data_attack_2 = pd.read_csv(files[2])[self.rel_features]
        self.data_attack_3 = pd.read_csv(files[3])[self.rel_features]
        self.data_attack_4 = pd.read_csv(files[4])[self.rel_features]

        self.data_normal = self.data_normal[self.rel_features]
        self.data_attack_1 = self.data_attack_1[self.rel_features]
        self.data_attack_2 = self.data_attack_2[self.rel_features]
        self.data_attack_3 = self.data_attack_3[self.rel_features]
        self.data_attack_4 = self.data_attack_4[self.rel_features]

        start_2 = int(len(self.data_normal) * 0.5)
        start_3 = int(len(self.data_normal) * 0.9)

        self.data_normal_1 = self.data_normal[:start_2]
        self.data_normal_2 = self.data_normal[start_2: start_3]
        self.data_normal_3 = self.data_normal[start_3:]

        # features to normalise
        # TODO - Add relevant column names to config file
        features_to_normalise = self.rel_features[:24] + self.rel_features[25:]

        # splitting the data into train val and test
        self.data_train, self.data_val, self.data_test = train_val_test_split(self.data_normal_1,
                                                                              train_frac=train_size, \
                                                                              val_frac=val_size, \
                                                                              test_frac=test_size, \
                                                                              shuffle=shuffle)

        # Normalising data using Z-score
        if normalise == True and len(features_to_normalise) != 0:
            self.data_train, mean, std = normalise_data(self.data_train, features_to_normalise)
            self.data_val, _, _ = normalise_data(self.data_val, features_to_normalise, mean, std)
            self.data_normal_2, _, _ = normalise_data(self.data_normal_2, features_to_normalise, mean, std)
            self.data_normal_3, _, _ = normalise_data(self.data_normal_3, features_to_normalise, mean, std)

            self.data_attack_1, _, _ = normalise_data(self.data_attack_1, features_to_normalise, mean, std)
            self.data_attack_2, _, _ = normalise_data(self.data_attack_2, features_to_normalise, mean, std)
            self.data_attack_3, _, _ = normalise_data(self.data_attack_3, features_to_normalise, mean, std)
            self.data_attack_4, _, _ = normalise_data(self.data_attack_4, features_to_normalise, mean, std)

        # create labels
        self.data_train_label = np.ones(shape=[len(self.data_train), 1])
        self.data_val_label = np.ones(shape=[len(self.data_val), 1])

        self.data_normal_2_label = np.ones(shape=[len(self.data_normal_2), 1])
        self.data_normal_3_label = np.ones(shape=[len(self.data_normal_3), 1])

        self.data_attack_1_label = np.zeros(shape=[len(self.data_attack_1), 1])
        self.data_attack_2_label = np.zeros(shape=[len(self.data_attack_2), 1])
        self.data_attack_3_label = np.zeros(shape=[len(self.data_attack_3), 1])
        self.data_attack_4_label = np.zeros(shape=[len(self.data_attack_4), 1])

        # Appending all five test dataset cases
        if test_size > 0:
            self.data_test_label = np.ones(shape=[len(self.data_test), 1])

        # converting datasets as ndarray to be processed be tensorflow graph
        self.data_train = np.array(self.data_train)
        self.data_val = np.array(self.data_val)

        self.data_normal_2 = np.array(self.data_normal_2)
        self.data_normal_3 = np.array(self.data_normal_3)

        self.data_attack_1 = np.array(self.data_attack_1)
        self.data_attack_2 = np.array(self.data_attack_2)
        self.data_attack_3 = np.array(self.data_attack_3)
        self.data_attack_4 = np.array(self.data_attack_4)
