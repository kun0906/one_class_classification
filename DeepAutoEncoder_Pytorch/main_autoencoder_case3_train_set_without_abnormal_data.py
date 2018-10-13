# -*- coding: utf-8 -*-
"""
    "abnormal detection by reconstruction errors"

    Case3:
        sess_normal_0 + all_abnormal_data(sess_TDL4_HTTP_Requests_0 +sess_Rcv_Wnd_Size_0_0)

    Case1 and Case 3:
        Train set : (0.7 * all_normal_data)*0.9
        Val_set: (0.7*all_normal_data)*0.1 + 0.1*all_abnormal_data
        Test_set: 0.3*all_normal_data+ 0.9*all_abnormal_data

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

import numpy as np
import torch
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from Utilities.CSV_Dataloader import mix_normal_attack_and_label
from Utilities.common_funcs import load_data_with_new_principle, show_data, evaluate_model


def print_net(net, describe_str='Net'):
    """

    :param net:
    :param describe_str:
    :return:
    """
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print(describe_str, net)
    print('Total number of parameters: %d' % num_params)


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, epochs=2):
        """

        :param X: Features
        :param y: Labels
        :param epochs:
        """
        super().__init__()

        self.epochs = epochs
        self.learning_rate = 1e-3
        self.batch_size = 64

        self.show_flg = True

        self.num_features_in = in_dim
        self.h_size = 16
        self.num_latent_features = 10
        self.num_features_out = self.num_features_in

        self.encoder = nn.Sequential(
            nn.Linear(self.num_features_in, self.h_size * 8),
            nn.ReLU(True),
            nn.Linear(self.h_size * 8, self.h_size * 4),
            nn.ReLU(True),
            nn.Linear(self.h_size * 4, self.h_size * 2),
            nn.ReLU(True),
            nn.Linear(self.h_size * 2, self.num_latent_features))

        self.decoder = nn.Sequential(
            nn.Linear(self.num_latent_features, self.h_size * 2),
            nn.ReLU(True),
            nn.Linear(self.h_size * 2, self.h_size * 4),
            nn.ReLU(True),
            nn.Linear(self.h_size * 4, self.h_size * 8),
            nn.ReLU(True),
            nn.Linear(self.h_size * 8, self.num_features_in),
            nn.Sigmoid())

        if self.show_flg:
            print_net(self.encoder, describe_str='Encoder')
            print_net(self.decoder, describe_str='Decoder')

        self.criterion = nn.MSELoss(reduction='elementwise_mean')
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x1 = self.encoder(x)
        x = self.decoder(x1)
        return x

    def train(self, train_set, val_set):
        """

        :param train_set:
        :param val_set:
        :return:
        """
        if len(train_set) == 0 or len(val_set) == 0:
            print('data set is not right.')
            return -1
        print('train_set shape is %s, val_set size is %s' % (train_set[0].shape, val_set[0].shape))
        assert self.num_features_in == train_set[0].shape[1]

        X = train_set[0]
        self.train_set = Data.TensorDataset(torch.Tensor(X),
                                            torch.Tensor(X))  # only use features data to train autoencoder
        dataloader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        self.results = {'train_set': {'acc': [], 'auc': []}, 'val_set': {'acc': [], 'auc': []}}
        self.loss = []
        for epoch in range(self.epochs):
            for iter, (batch_X, _) in enumerate(dataloader):
                # # ===================forward=====================
                output = self.forward(batch_X)
                loss = self.criterion(output, batch_X)
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.loss.append(loss.data)
            self.T = self.loss[-1]
            val_set_acc, val_set_cm = self.evaluate(val_set, threshold=self.T.data)
            self.results['val_set']['acc'].append(val_set_acc)
            # ===================log========================
            print('epoch [{:d}/{:d}], loss:{:.4f}'.format(epoch + 1, self.epochs, loss.data))
            # if epoch % 10 == 0:
            #     # pic = to_img(output.cpu().Data)
            #     # save_image(pic, './mlp_img/image_{}.png'.format(epoch))
        self.T = self.loss[-1]

        if self.show_flg:
            show_data(self.loss, x_label='epochs', y_label='loss', fig_label='loss',
                      title='val_set evaluation on training process')

    def evaluate(self, test_set, threshold=0.1):
        """

        :param test_set:
        :param threshold:
        :return:
        """
        X = torch.Tensor(test_set[0])
        y_true = test_set[1]

        # self.T = torch.Tensor([0.0004452318244148046])  # based on the training loss.
        self.T = torch.Tensor([threshold])
        # Step 1. predict output
        AE_outs = self.forward(X)

        # Step 2. comparison with Threshold.
        y_preds = []
        num_abnormal = 0
        print('Threshold(T) is ', self.T.data.tolist())
        for i in range(X.shape[0]):
            # if torch.dist(AE_outs, X, 2) > self.T:
            # if torch.norm((AE_outs[i] - X[i]), 2) > self.T:
            if self.criterion(AE_outs[i], X[i]) > self.T:
                # print('abnormal sample.')
                y_preds.append('1')  # 0 is normal, 1 is abnormal
                num_abnormal += 1
            else:
                y_preds.append('0')

        # Step 3. achieve the evluation standards.
        print('No. of abnormal sample is ', num_abnormal)
        y_preds = np.asarray(y_preds, dtype=int)
        cm = confusion_matrix(y_pred=y_preds, y_true=y_true)
        print('Confusion matrix:\n', cm)
        acc = 100.0 * sum(y_true == y_preds) / len(y_true)
        print('Acc: %.2f%%' % acc)

        return acc, cm


def save_model(model, out_file='../log/autoencoder.pth'):
    """

    :param model:
    :param out_file:
    :return:
    """
    out_dir = os.path.split(out_file)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save(model, out_file)

    return out_file


def ae_main(input_file, epochs=2, out_dir='./log'):
    """

    :param input_file:
    :param epochs:
    :param out_dir:
    :return:
    """
    torch.manual_seed(1)
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    st = time.time()
    print('It starts at ', start_time)

    # step 1 load Data and do preprocessing
    # train_set, val_set, test_set = load_data(input_file, norm_flg=True,
    #                                          train_val_test_percent=[0.7 * 0.9, 0.7 * 0.1, 0.3])
    train_set_without_abnormal_data, val_set, test_set = load_data_with_new_principle(input_file, norm_flg=True,
                                                                                      train_val_test_percent=[0.7 * 0.9,
                                                                                                              0.7 * 0.1,
                                                                                                              0.3])
    print('train_set:%s,val_set:%s,test_set:%s' % (
        Counter(train_set_without_abnormal_data[1]), Counter(val_set[1]), Counter(test_set[1])))

    # step 2.1 model initialization
    AE_model = AutoEncoder(in_dim=train_set_without_abnormal_data[0].shape[1],
                           epochs=epochs)  # train_set (no_label and no abnormal traffic)

    # step 2.2 train model
    AE_model.train(train_set_without_abnormal_data, val_set)

    # step 3.1 dump model
    model_file = save_model(AE_model, out_file='../log/autoencoder.pth')

    # step 3.2 load model
    AE_model = torch.load(model_file)

    # step 4 evaluate model
    # re-evaluation on val_set to choose the best threshold.
    max_acc_thres = evaluate_model(AE_model, val_set, iters=10,
                                   fig_params={'x_label': 'evluation times', 'y_label': 'accuracy on val set',
                                               'fig_label': 'acc', 'title': 'accuracy on val set'})
    AE_model.T = torch.Tensor([max_acc_thres[1]])  #
    print('the best result on val_set is ', max_acc_thres)
    evaluate_model(AE_model, test_set, iters=1,
                   fig_params={'x_label': 'evluation times', 'y_label': 'accuracy on test set',
                               'fig_label': 'acc', 'title': 'accuracy on test set'})

    end_time = time.strftime('%Y-%h-%d %H:%M:%S', time.localtime())
    print('It ends at ', end_time)
    print('All takes %.4f s' % (time.time() - st))


if __name__ == '__main__':
    # input_file = '../Data/Wednesday-workingHours-withoutInfinity-Sampled.pcap_ISCX.csv'
    normal_file = '../Data/sess_normal_0.txt'
    attack_file = '../Data/sess_TDL4_HTTP_Requests_0.txt'
    # attack_file = '../Data/sess_Rcv_Wnd_Size_0_0.txt'
    if 'TDL4' in attack_file:
        out_file = '../Data/case1.csv'
    elif 'Rcv_Wnd' in attack_file:
        out_file = '../Data/case2.csv'
    else:
        pass
    if not os.path.exists(out_file):
        st = time.time()
        (_, _), input_file = mix_normal_attack_and_label(normal_file, attack_file, out_file)
        print('mix dataset takes %.2f(s)' % (time.time() - st))
    else:
        input_file = out_file
    epochs = 2
    ae_main(input_file, epochs)
