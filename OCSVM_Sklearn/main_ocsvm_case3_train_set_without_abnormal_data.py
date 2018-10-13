# -*- coding: utf-8 -*-
"""
    "Using one class classification (ocsvm) to detect abnormal traffic"
    # 0 is normal, 1 is abnormal

    Case3:
        sess_normal_0 + all_abnormal_data(sess_TDL4_HTTP_Requests_0 +sess_Rcv_Wnd_Size_0_0)

    Case1 and Case 3:
        Train set : (0.7 * all_normal_data)
        using "train_test_split(test_size = 0.7)" in sklearn to split "val_set and test_set"
        Val_set: 0.3*(all_normal_data*0.3 + all_abnormal_data)
        Test_set: 0.7*(all_normal_data*0.3+ all_abnormal_data)

        ### train_set=0.7*normal*0.9, test_set = 0.7*(abnormal+ 0.3*normal), val_set = 0.3*(abnormal+0.7*normal)

     Created at :
        2018/10/04

    Version:
        0.1.0

    Requirements:
        python 3.x
        Sklearn 0.20.0

    Author:

"""
import os
import time
from collections import Counter
from random import shuffle

import numpy as np
from sklearn.model_selection import train_test_split

from OCSVM_Sklearn.basic_svm import OCSVM
from Utilities.CSV_Dataloader import mix_normal_attack_and_label, open_file
from Utilities.common_funcs import load_data, dump_model, load_model, load_data_with_new_principle, normalizate_data


def ocsvm_main(input_file='csv', kernel='rbf', out_dir='./log', **kwargs):
    """

    :param input_data:
    :param kernel:
    :param out_path:
    :param kwargs:
    :return:
    """
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    st = time.time()
    print('It starts at ', start_time)

    # step 1 load Data and do preprocessing
    # train_set, val_set, test_set = load_data(input_file, norm_flg=True,
    #                                          train_val_test_percent=[0.7 * 0.9, 0.7 * 0.1, 0.3])
    train_set_without_abnormal_data, val_set, test_set = achieve_train_val_test_from_files(input_file, norm_flg=True,
                                                                                           train_val_test_percent=[
                                                                                               0.7 * 0.9,
                                                                                               0.7 * 0.1,
                                                                                               0.3])
    print('train_set:%s,val_set:%s,test_set:%s' % (
        Counter(train_set_without_abnormal_data[1]), Counter(val_set[1]), Counter(test_set[1])))

    # step 2.1 initialize OC-SVM
    ocsvm = OCSVM(train_set=train_set_without_abnormal_data, kernel=kernel, grid_search_cv_flg=True, val_set=val_set)

    # step 2.2 train OC-SVM model
    ocsvm.train(train)

    # step 3.1 dump model
    model_file = dump_model(ocsvm, os.path.join(out_dir, 'model.p'))

    # step 3.2 load model
    ocsvm_model = load_model(model_file)

    # step 4 evaluate model
    # model.evaluate(train_set, name='train_set')
    ocsvm_model.evaluate(test_set, name='test_set')

    end_time = time.strftime('%Y-%h-%d %H:%M:%S', time.localtime())
    print('It ends at ', end_time)
    print('All takes %.4f s' % (time.time() - st))


if __name__ == '__main__':
    # input_file = '../Data/Wednesday-workingHours-withoutInfinity-Sampled.pcap_ISCX.csv'
    normal_file = '../Data/sess_normal_0.txt'
    attack_file_1 = '../Data/sess_TDL4_HTTP_Requests_0.txt'
    attack_file_2 = '../Data/sess_Rcv_Wnd_Size_0_0.txt'
    input_file = {'normal_files': [normal_file], 'attack_files': [attack_file_1, attack_file_2]}
    ocsvm_main(input_file, kernel='rbf')
