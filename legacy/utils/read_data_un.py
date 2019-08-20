import pandas as pd
import numpy as np
from utils.utils import normalise_data


class Dataset:
    '''
    Class for loading the dataset and further preprocessing and return a clean, normalised dataset as a numpy array
    '''

    def __init__(self, filename, label, normalize_flg=True, mean=None, std=None, start=0, end=None):
        '''
        load the data files as panda DataFrames and set variables
        '''

        # Reading Data
        self.data = pd.DataFrame()

        for file in filename:
            if end:
                self.temp_data = pd.read_csv(file)[start:end]
            else:
                self.temp_data = pd.read_csv(file)[start:]

            self.data = pd.concat([self.data, self.temp_data])

        self.origin_data = self.data.values

        feature_names = list(self.data.columns)
        self.rel_features = feature_names[5:]

        # features to normalise
        features_to_normalise = self.rel_features[:24] + self.rel_features[25:]  # Not normalising the feature 'isNew'

        # Setting the indexes if specified
        self.data = self.data[self.rel_features]
        self.df = self.data

        if normalize_flg == False:
            print('do not normalize.')
            pass
        else:
            # Normalising Data
            if not mean:
                self.data, self.mean, self.std = normalise_data(self.data, features_to_normalise)
            else:
                # print("Normalising for specified mean and std")
                self.data, _, _ = normalise_data(self.data, features_to_normalise, mean, std)

        # Creating Labels as specified
        if label == 0:
            self.data_label = np.zeros(shape=[len(self.data), 1])
        else:
            self.data_label = np.ones(shape=[len(self.data), 1])

        # converting dataset as ndarray
        self.data = np.array(self.data)
