# -*- coding: utf-8 -*-
"""
    "abnormal detection by reconstruction errors"

    Case1:
        sess_normal_0 + sess_TDL4_HTTP_Requests_0
    Case2:
        sess_normal_0  + sess_Rcv_Wnd_Size_0_0

    Case1 and Case 2:
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
from collections import Counter

from sklearn.metrics import confusion_matrix

from Utilities.CSV_Dataloader import mix_normal_attack_and_label
from Utilities.common_funcs import load_data, load_data_with_new_principle, show_data

import time
import numpy as np

import torch
import torch.utils.data as Data
from torch import nn
from torch.utils.data import DataLoader


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
        # self.train_set_with_labels = train_set  # used for evaluation
        # X = np.asarray([x_t for (x_t, y_t) in zip(*train_set) if y_t == 0], dtype=float)
        # print('X.shape: ', X.shape)
        # # self.dataset = Data.TensorDataset(torch.Tensor(X), torch.Tensor(y))



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

        :param val_set:
        :return:
        """
        if len(train_set) == 0 or len(val_set) == 0:
            print('data set is not right.')
            return -1
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
            val_set_acc, val_set_cm = self.evaluate(val_set, Threshold=self.T.data)
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

    def evaluate(self, test_set, Threshold=0.1):
        """

        :param test_set:
        :return:
        """
        X = torch.Tensor(test_set[0])
        y_true = test_set[1]

        # self.T = torch.Tensor([0.0004452318244148046])  # based on the training loss.
        self.T = torch.Tensor([Threshold])
        ### predict output
        AE_outs = self.forward(X)

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
        print('No. of abnormal sample is ', num_abnormal)
        y_preds = np.asarray(y_preds, dtype=int)
        cm = confusion_matrix(y_pred=y_preds, y_true=y_true)
        print('Confusion matrix:\n', cm)
        acc = 100.0 * sum(y_true == y_preds) / len(y_true)
        print('Acc: %.2f%%' % acc)

        return acc, cm


def save_model(model, out_file='../log/autoencoder.pth'):
    out_dir = os.path.split(out_file)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save(model, out_file)

    return out_file


def evaluate_model(AE_model, test_set, iters=10):
    i = 0
    for thres in np.linspace(start=0, stop=(AE_model.T.data) * 5, num=iters, endpoint=True):
        print("Evaluation:%d/%d threshold = %f" % (i, iters, thres))
        test_set_acc, test_set_cm = AE_model.evaluate(test_set, Threshold=thres)
        i += 1

def ae_main(input_file, epochs=2, out_dir='./log'):
    """

    :param input_file: CSV
    :return:
    """
    torch.manual_seed(1)
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    st = time.time()
    print('It starts at ', start_time)

    # step 1 load Data and do preprocessing
    # train_set, val_set, test_set = load_data(input_file, norm_flg=True,
    #                                          train_val_test_percent=[0.7 * 0.9, 0.7 * 0.1, 0.3])
    train_set, val_set, test_set = load_data_with_new_principle(input_file, norm_flg=True,
                                             train_val_test_percent=[0.7 * 0.9, 0.7 * 0.1, 0.3])
    print('train_set:%s,val_set:%s,test_set:%s' % (Counter(train_set[1]), Counter(val_set[1]), Counter(test_set[1])))

    # step 2.1 model initialization
    AE_model = AutoEncoder(in_dim=train_set[0].shape[1], epochs=epochs)  # train_set (no_label and no abnormal traffic)

    # step 2.2 train model
    AE_model.train(train_set, val_set)

    # step 3.1 dump model
    model_file = save_model(AE_model, out_file='../log/autoencoder.pth')

    # step 3.2 load model
    AE_model = torch.load(model_file)

    # step 4 evaluate model
    evaluate_model(AE_model, test_set, iters=1)

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
    epochs = 5
    ae_main(input_file, epochs)
