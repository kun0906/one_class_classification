"""
    Purpose:
        includes all the configuration and some constants.

    python naming Conventions:   https://visualgit.readthedocs.io/en/latest/pages/naming_convention.html
        1) Constant names must be fully capitalized
        2) Class names should follow the UpperCaseCamelCase convention
        3) Package names should be all lower case. It is usually preferable to stick to 1 word names
        4) Module names should be all lower case. It is usually preferable to stick to 1 word names
        5) Function names should be all lower case

    Note:
        1) Avoid using names that are too general or too wordy. Strike a good balance between the two
        2) Use comments and doc strings for description of what is going on, not variable names

"""

### all the parameters used.

params_dict = {
    ### Unsupervised machine learning algorithms.
    'AE': {'Epochs': 50,
           'batch_size': 32,
           'h_dim': 24,  # the number of neurons of each hidden layer
           'latent_dim': 16,  # the number of neurons of latent layer.
           'find_optimal_thres_flg': True,  # use the average training loss of the last 4 Epochs
           # false: use the given optimal_thres_AE, otherwise, find the optimal one by using training loss.
           'optimal_AE_thres': 0.8,  # the val will be changed from train, training_loss * factor: factor =10,
           'factor_AE_thres': 5  # optimal_AE_thres= training_loss * factor: factor = 10,
           },

    'PCA': {},
    'OCSVM': {},
    'IF': {},

    ### Supervised machine learning algorithms.
    'DT': {},
    'balance_data_flg': False,  # for DT training

    ### common parameters for all algorithms.
    'input_dir': "input_data/dataset",
    'output_dir': "output_data",
    'norm_flg': True,  # normlize the data.
    'norm_method': 'min-max',  # 'z-score' or 'min-max'
    'test_set_percent': 0.2,  # train and test ratio.
    'random_state': 42,
    'show_flg': True,

    # 'sub_features_flg': False,  # if using the selected features to conduct experiments.
    'selected_features_lst': [],  # the selected features:[6, 7, 8, 10]
    'analyze_feature_flg': False,  # True: do feature selection.
    'feature_experiment_flg': False,  # True, conduct experiment on different features.
    'not_normalized_features_lst': [0,1],  # tcp and udp with one hot encoding, these features no need to normalize.

    'verbose': True,  # print all the information.
    'title_flg': True,  # if the figures have the title or not.

}

"""
   Step 1.  random control in order to achieve reproductive results

    cited from https://stackoverflow.com/questions/54047654/tensorflow-different-results-with-the-same-random-seed
"""

# Seed value
# Apparently you may use different seed values at each stage
random_state = params_dict['random_state']

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(random_state)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random

random.seed(random_state)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(random_state)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.set_random_seed(random_state)

# 5. Configure a new global `tensorflow` session
# from keras import backend as K
from tensorflow.python.keras import backend as K

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# Force Tensorflow to use a single thread
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# 6. PyTorch
# You can use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA):
import torch

torch.manual_seed(random_state)
# CuDNN
# When running on the CuDNN backend, two further options must be set:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""
    Step 2. 
"""
import sys

sys.stdout.flush()
# out_file = 'stdout_content.txt'
# # sys.stdout = open(out_file, mode='w', buffering=1)
# ###skip 'buffering' if you don't want the output to be flushed right away after written
# # sys.stdout = sys.__stdout__
#

#
# class Parameters():
#
#     def __init__(self):
#        all the parameters ...
#        ...
