# -*- coding: utf-8 -*-
"""
    "Using one class classification (ocsvm) to detect abnormal traffic"

    Case1:
        sess_normal_0 + sess_TDL4_HTTP_Requests_0
    Case2:
        sess_normal_0  + sess_Rcv_Wnd_Size_0_0

    Case1 and Case 2:
        Train set : (0.7 * all_normal_data  + 0.7* all_abnormal_data)*0.9
        Val_set: (0.7*all_normal_data + 0.7*all_abnormal_data)*0.1
        Test_set: 0.3*all_normal_data+ 0.3*all_abnormal_data


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

from history_files.basic_svm import OCSVM
from utils.CSV_dataloader import mix_normal_attack_and_label
from utils.common_funcs import load_data, dump_model, load_model


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

    # step 1. load input_data
    train_set, val_set, test_set = load_data(input_file, norm_flg=True,
                                             train_val_test_percent=[0.7 * 0.9, 0.7 * 0.1, 0.3])

    # step 2.1 initialize OC-SVM
    ocsvm = OCSVM(train_set=train_set, kernel=kernel, grid_search_cv_flg=True, val_set=val_set)

    # step 2.2 train OC-SVM model
    ocsvm.train()

    # step 3.1 dump model
    out_dir = '../log'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_path = os.path.join(out_dir, 'model.p')
    dump_model(ocsvm, model_path)

    # step 3.2 load model
    model = load_model(input_file=model_path)

    # step 4 evaluate model
    model.evaluate(train_set, name='train_set')
    model.evaluate(test_set, name='test_set')

    end_time = time.strftime('%Y-%h-%d %H:%M:%S', time.localtime())
    print('It ends at ', end_time)
    print('All takes %.4f s' % (time.time() - st))


if __name__ == '__main__':
    # dataset = 'mnist'
    # dataset = '../input_data/Wednesday-workingHours-withoutInfinity-Sampled.pcap_ISCX.csv'
    # ocsvm_main(dataset, kernel='rbf')

    # input_file = '../input_data/Wednesday-workingHours-withoutInfinity-Sampled.pcap_ISCX.csv'
    normal_file = '../input_data/sess_normal_0.txt'
    attack_file = '../input_data/sess_TDL4_HTTP_Requests_0.txt'
    # attack_file = '../input_data/sess_Rcv_Wnd_Size_0_0.txt'
    if 'TDL4' in attack_file:
        out_file = '../input_data/case1.csv'
    elif 'Rcv_Wnd' in attack_file:
        out_file = '../input_data/case2.csv'
    else:
        pass
    if not os.path.exists(out_file):
        st = time.time()
        (_, _), input_file = mix_normal_attack_and_label(normal_file, attack_file, out_file)
        print('mix dataset takes %.2f(s)' % (time.time() - st))
    else:
        input_file = out_file
    epochs = 10
    ocsvm_main(input_file, kernel='rbf')
