"""
    includes all the configuration and some constants.

"""

"""
   Step 1.  random control in order to achieve reproductive results

    cited from https://stackoverflow.com/questions/54047654/tensorflow-different-results-with-the-same-random-seed
"""

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

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
torch.manual_seed(seed_value)
# CuDNN
# When running on the CuDNN backend, two further options must be set:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



"""
   Step 2.  neural network configuration
"""

Epoches = 10
batch_size =32
h_dim = 16   # the number of neurons of each hidden layer
latent_dim = 8 # the number of neurons of latent layer.










