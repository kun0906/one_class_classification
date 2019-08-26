"""
    autoencoder evaluation on offline experiments

"""
from configuration import *
from collections import OrderedDict

from algorithms.proposed_algorithms.experiment_1 import conduct_experiment_1
from algorithms.proposed_algorithms.experiment_2 import conduct_experiment_2
from algorithms.proposed_algorithms.experiment_3 import conduct_experiment_3


def execute_experiment(experiment='', Epochs=1, *kwarg, **kwargs):
    """

       :param Epochs:
       :param experiments:  For scenario 1 experiment 1
       :param show_flg:
       :param balance_train_data_flg:
       :return:
       """
    kwargs = kwargs['kwargs']
    random_state = kwargs['random_state']
    norm_flg = kwargs['norm_flg']
    optimal_AE_thres = kwargs['optimal_AE_thres']
    find_optimal_thres_flg = kwargs['find_optimal_thres_flg']
    factor_AE_thres = kwargs['factor_AE_thres']
    sub_features_lst = kwargs['sub_features_lst']


def main(params_dict={}):
    """
        params_dict={'AE':{'Epochs':'in_dim':''},

                    'PCA':{},
                    'OCSVM':{},
                    'IF':{},

                     'DT':{''},

                    'input_dir':"input_data/dataset"
                    'output_dir':"output_data"
                    'norm_flg': True # normlize the data.
                    'norm_method':'z-score' #  'z-score' or 'min-max'
                    'test_size_percent':0.2 # train and test ratio.
                    'random_state':42,
                    'show_flg':True,

                     ...
                     }

        datasets_dict={'train_set_dict':'', 'val_set_dict':'', 'test_set_dict':''},
            train_set_dict ={'train_set':{'x':, 'y':,}}
            val_set_dict ={'val_set':{'x':, 'y':,}}
            test_set_dict ={'test_set':{'x':, 'y':,},'test_set_2':{'x':, 'y':,},... }

    :return:
    """
    experiments_dict = OrderedDict({
        'experiment_1': OrderedDict(
            {'SYNT': 'uSc1C1_z-score_20_14', 'UNB': 'uSc1C2_z-score_20_14', 'MAWI': 'uSc1C3_z-score_20_14'}),
        'experiment_2': OrderedDict(
            {'SYNT': 'uSc2C1_z-score_20_14', 'UNB': 'uSc2C2_z-score_20_14', 'MAWI': 'uSc2C3_z-score_20_14'}),
        'experiment_3': OrderedDict(
            {'SYNT': 'uSc3C1_z-score_20_14', 'UNB': 'uSc3C2_z-score_20_14', }),
        'demo': {'demo': 'uSc1C3_z-score_20_14'}
    })
    '''
        sub_experiemnt = 'uSc1C1_z-score_20_14'
        sub_experiemnt[0] = u/s for Unsupervised or Supervised
        sub_experiemnt[3] = Scenario: Experiment 1, 2, 3
        sub_experiemnt[5] = Source: SYNT, UNB, MAWI
    '''
    mode = 'demo'
    # mode = 'all'
    if mode == 'all':
        for key, value_dict in experiments_dict.items():
            print(f'{key}: {value_dict}')
            for sub_key, sub_value in value_dict.items():
                print(f'-{key} => {sub_key}:{sub_value}')
                if key == 'experiment_1':
                    conduct_experiment_1(sub_experiment=(sub_key, sub_value), norm_flg=norm_flg,
                                         norm_method=norm_method,
                                         Epochs=Epochs,
                                         optimal_AE_thres=optimal_AE_thres,
                                         find_optimal_thres_flg=find_optimal_thres_flg,
                                         factor_AE_thres=factor_AE_thres, analyize_features_flg=analyize_features_flg,
                                         random_state=random_state, verbose=verbose
                                         )
                elif key == 'experiment_2':
                    conduct_experiment_2(sub_experiment=(sub_key, sub_value), norm_flg=norm_flg,
                                         norm_method=norm_method,
                                         Epochs=Epochs,
                                         optimal_AE_thres=optimal_AE_thres,
                                         find_optimal_thres_flg=find_optimal_thres_flg,
                                         factor_AE_thres=factor_AE_thres, analyize_features_flg=analyize_features_flg,
                                         random_state=random_state, verbose=verbose
                                         )
                elif key == 'experiment_3':
                    conduct_experiment_3(sub_experiment=(sub_key, sub_value), norm_flg=norm_flg,
                                         norm_method=norm_method,
                                         Epochs=Epochs,
                                         optimal_AE_thres=optimal_AE_thres,
                                         find_optimal_thres_flg=find_optimal_thres_flg,
                                         factor_AE_thres=factor_AE_thres, analyize_features_flg=analyize_features_flg,
                                         random_state=random_state, verbose=verbose
                                         )
                else:
                    print(f'{key:sub_key} or value:{sub_value} is not correct, please check again.')
                    pass
    elif mode == 'demo':
        sub_key = 'demo'
        sub_value = experiments_dict[sub_key][sub_key]
        if sub_value[3] == '1':  # conduct_experiment_1
            conduct_experiment_1(sub_experiment=(sub_key, sub_value), norm_flg=norm_flg,
                                 norm_method=norm_method,
                                 Epochs=Epochs,
                                 optimal_AE_thres=optimal_AE_thres, find_optimal_thres_flg=find_optimal_thres_flg,
                                 factor_AE_thres=factor_AE_thres, analyize_features_flg=analyize_features_flg,
                                 random_state=random_state, verbose=verbose
                                 )
        elif sub_value[3] == '2':  # conduct_experiment_2
            conduct_experiment_1(sub_experiment=(sub_key, sub_value), norm_flg=norm_flg,
                                 norm_method=norm_method,
                                 Epochs=Epochs,
                                 optimal_AE_thres=optimal_AE_thres, find_optimal_thres_flg=find_optimal_thres_flg,
                                 factor_AE_thres=factor_AE_thres, analyize_features_flg=analyize_features_flg,
                                 random_state=random_state, verbose=verbose
                                 )
        elif sub_value[3] == '3':  # conduct_experiment_3
            conduct_experiment_3(sub_experiment=(sub_key, sub_value), norm_flg=norm_flg,
                                 norm_method=norm_method,
                                 Epochs=Epochs,
                                 optimal_AE_thres=optimal_AE_thres, find_optimal_thres_flg=find_optimal_thres_flg,
                                 factor_AE_thres=factor_AE_thres, analyize_features_flg=analyize_features_flg,
                                 random_state=random_state, verbose=verbose
                                 )
        else:
            print(f'{key:sub_key} or value:{sub_value} is not correct, please check again.')
    else:
        print(f'mode:{mode} is not correct.')
        pass


if __name__ == '__main__':
    main(params_dict=params_dict)
