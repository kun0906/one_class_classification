"""
    includes all the configuration and some constants.

"""

global random_state, norm_flg, test_size, Epochs, batch_size, h_dim, latent_dim, find_optimal_thres_flg, \
    optimal_thres_AE, factor_thres_AE,analyize_features_flg, sub_features_lst, feature_selection_experiment_flg

"""
   Step 1.  random control in order to achieve reproductive results

    cited from https://stackoverflow.com/questions/54047654/tensorflow-different-results-with-the-same-random-seed
"""

# Seed value
# Apparently you may use different seed values at each stage
random_state= 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(random_state)

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
#Force Tensorflow to use a single thread
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


"""
    Step 3. 
"""
input_dir = "input_data/dataset"
output_dir = "output_data"
norm_flg = True # normlize the data.
test_size = 0.2 # train and test ratio.
sub_features_flg = False # if using the selected features to conduct experiments.
sub_features_lst=[6,7,8,10] # the selected features
analyize_features_flg = False  # True: do feature selection.
feature_selection_experiment_flg=False # True, conduct experiment on different features.
no_normalized_features_idx=[]   # tcp and udp with one hot encoding, features no need to normalize.

"""
   Step 4.  neural network configuration
"""

Epochs = 2
batch_size =32
h_dim = 16   # the number of neurons of each hidden layer
latent_dim = 8 # the number of neurons of latent layer.

find_optimal_thres_flg = False  # false: use the given optimal_thres_AE, otherwise, find the optimal one by using training loss.
optimal_thres_AE = 0.95  # the val will be changed from train, training_loss*factor: factor =10,
factor_thres_AE = 35    #



"""

"""
show_flg = True  # plot the results
title_flg = True # if the figures have the title or not.
balance_train_data_flg=True # for DT training

