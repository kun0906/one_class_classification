"""
    using pandas to load data from file

"""
import pandas as pd


def load_data(input_f):
    data = pd.read_csv(input_f)
    print(data.dtypes)


if __name__ == '__main__':
    input_f = '../Data/sess_DDoS_Excessive_GET_POST'
    load_data(input_f)
