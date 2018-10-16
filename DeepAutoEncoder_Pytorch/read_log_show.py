# -*- coding:utf-8 -*-
"""

"""
from Utilities.common_funcs import show_data


def parse_log(input_file):
    loss = {'train_loss': [], 'val_loss': []}
    with open(input_file, 'r') as in_f:
        line = in_f.readline().strip()
        while line:
            if not line.startswith('epoch ['):
                line = in_f.readline()
                continue
            line_arr = line.split(',')
            loss['train_loss'].append(float(line_arr[1].strip().split(':')[-1]))
            loss['val_loss'].append(float(line_arr[2].strip().split(':')[-1]))
            line = in_f.readline()

    return loss


if __name__ == '__main__':
    input_file = './outlog_9299217_4294967294.out'
    loss = parse_log(input_file)
    show_data(loss['train_loss'], x_label='epochs', y_label='loss', fig_label='', title='train_loss')
    show_data(loss['val_loss'], x_label='epochs', y_label='loss', fig_label='', title='val_loss')
    print(loss)
