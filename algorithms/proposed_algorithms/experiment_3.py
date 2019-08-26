"""
    Experiment I

"""

from algorithms.proposed_algorithms.autoencoder import train_AE, test_AE
from algorithms.traditional_algorithms.classical_ml_algorithms import train_PCA, test_PCA, train_IF, test_IF, \
    train_OCSVM, test_OCSVM, train_DT, test_DT
from configuration import *
from utils.dataloader import *
from preprocess.feature_selection import *
from preprocess.normalization import *

from utils.visualization import *


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


def train_test_traditional_algorithms_experiment_3(experiment=('key', ''), train_set='train_set_dict', val_set='val_set_dict',
                               test_sets_dict='',
                                                   output_dir='', show_flg=True, verbose=1):
    """

    :param verbose:
    :return:
    """
    # x_train, y_train = train_set
    # print('\ndata is used to train and test traditional algorithms (PCA, OCSVM and IF) for {experiment}.')
    # # print(f'all_norm:{x_train.shape[0]+x_val.shape[0]+x_test.shape[0]}, all_attack:{x_attack_1.shape[0]}')
    # print(f'x_train:{x_train.shape}, y_train:{Counter(y_train.reshape(-1,).tolist())}')
    # # print(f'x_val:{x_val.shape}, y_val:{Counter(y_val.reshape(-1,))}')
    # # print(f'x_test:{x_test.shape}, y_test:{Counter(y_test.reshape(-1,))}')
    # # if type(x_attack) != type(None) and type(y_attack) != type(None):
    # #     print(f'x_remained_attack:{x_attack.shape}, y_remained_attack:{Counter(y_attack.reshape(-1,))}')

    experiment = experiment[0] + '-' + experiment[1]  # (key, value)
    y_test_pred_prob_dict = {'DT': [], 'PCA': [], 'IF': [], 'OCSVM': []}
    # ### 2. evaluate DT
    # # without attack data in training, so no DT model in this experiment.
    # print('\n2). no need to evaluate DT on test set')

    ### 0. evaluate DT
    print('\n1). train and evaluate DT...')
    x_train = np.concatenate([train_set['x_norm_train'], train_set['x_attack_train']])
    y_train = np.concatenate([train_set['y_norm_train'], train_set['y_attack_train']])

    train_DT(x_train=x_train, y_train = y_train, experiment=experiment)
    print(f'x_train.shape: {x_train.shape}')
    test_DT(x_test=x_train, y_test=y_train, experiment=experiment)

    for key, value_dict in test_sets_dict.items():
        x_test = np.concatenate([value_dict['x_norm_test'], value_dict['x_attack_test']])
        y_test = np.concatenate([value_dict['y_norm_test'], value_dict['y_attack_test']])
        print(f'normalize {key}, x_test.shape: {x_test.shape}, y_test:{Counter(y_test.reshape(-1,))}')
        y_pred_label_DT, y_pred_probs_DT = test_DT(x_test=x_test, y_test=y_test,
                                                                      experiment=experiment,
                                                                    output_dir=output_dir)
        y_test_pred_prob_dict['DT'].append((y_test, y_pred_probs_DT))

    # ### 1. evaluate PCA
    # print('\n1). train and evaluate PCA...')
    # train_PCA(x_train=x_train, experiment=experiment)
    # print(f'x_train.shape: {x_train.shape}')
    # test_PCA(x_test=x_train, y_test=y_train, experiment=experiment)
    #
    # for key, val in test_sets_dict.items():
    #     (x_test, y_test) = val
    #     print(f'normalize {key}, x_test.shape: {x_test.shape}, y_test:{Counter(y_test.reshape(-1,))}')
    #     y_pred_label_PCA, y_pred_probs_PCA = test_PCA(x_test=x_test, y_test=y_test, experiment=experiment)
    #     y_test_pred_prob_dict['PCA'].append((y_test, y_pred_probs_PCA))

    # # ### 2. evaluate IF
    # print('\n2). train and evaluate IF...')
    # train_IF(x_train=x_train, experiment=experiment)
    # print(f'x_train.shape: {x_train.shape}')
    # test_IF(x_test=x_train, y_test=y_train, experiment=experiment)
    #
    # for key, val in test_sets_dict.items():
    #     (x_test, y_test) = val
    #     print(f'evaluate on {key}, x_test.shape: {x_test.shape}, y_test:{Counter(y_test)}')
    #     y_pred_label_IF, y_pred_probs_IF = test_IF(x_test=x_test, y_test=y_test, experiment=experiment)
    #     y_test_pred_prob_dict['IF'].append((y_test, y_pred_probs_IF))
    #
    # ### 3. evaluate OC-SVM
    # print('\n3). train and evaluate OCSVM...')
    # train_OCSVM(x_train=x_train, experiment=experiment)
    # print(f'x_train.shape: {x_train.shape}')
    # test_OCSVM(x_train[:100], y_train[:100], experiment)
    #
    # for key, val in test_sets_dict.items():
    #     (x_test, y_test) = val
    #     print(f'evaluate on {key}, x_test.shape: {x_test.shape}, y_test:{Counter(y_test)}')
    #     y_pred_label_OCSVM, y_pred_probs_OCSVM = test_OCSVM(x_test=x_test, y_test=y_test, experiment=experiment)
    #     y_test_pred_prob_dict['OCSVM'].append((y_test, y_pred_probs_OCSVM))

    return y_test_pred_prob_dict, '', ''


def train_test_AE_experiment_3(experiment=('key', ''), train_set='train_set_dict', val_set='val_set_dict',
                               test_sets_dict='',
                               Epochs=1, find_optimal_thres_flg=True, optimal_AE_thres=-1,
                               factor_AE_thres='',
                               output_dir='output_data', show_flg=True, verbose=verbose):
    """

    :param verbose:
    :return:
    """
    # x_train, y_train = train_set
    # x_val, y_val = val_set
    # print(f'data is used to train and test AE for {experiment}.')
    # # print(f'all_norm:{x_train.shape[0]+x_val.shape[0]+x_test.shape[0]}, all_attack:{x_attack_1.shape[0]}')
    # print(f'x_train:{x_train.shape}, y_train:{Counter(y_train.reshape(-1,).tolist())}')
    # print(f'x_val:{x_val.shape}, y_val:{Counter(y_val.reshape(-1,))}')
    # # print(f'x_test:{x_test.shape}, y_test:{Counter(y_test.reshape(-1,))}')
    # # # if type(x_attack) != type(None) and type(y_attack) != type(None):
    # # #     print(f'x_remained_attack:{x_attack.shape}, y_remained_attack:{Counter(y_attack.reshape(-1,))}')

    experiment = experiment[0] + '-' + experiment[1]  # (key, value)
    print(f'-train AE on train set.')
    num_features = train_set['x_norm_train'].shape[1]
    training_losses = train_AE(x_train=train_set['x_norm_train'], x_val=val_set['x_norm_val'], experiment=experiment, in_dim=num_features,
                               Epochs=Epochs,
                               output_dir=output_dir, verbose=verbose)
    print(f'training_losses:{training_losses}')

    # if find_optimal_thres_flg:
    #     print(f'find the optimal threshold ({find_optimal_thres_flg}) for AE by using training losss')
    #     optimal_AE_thres, factor_AE_thres, key, = get_optimal_thres(training_losses, factor_AE_thres=factor_AE_thres,
    #                                                                 key='loss')
    # else:
    #     print('using the presetted threshold')
    #     optimal_AE_thres = optimal_AE_thres
    #     key = None
    #     factor_AE_thres = None
    # print(f'optimal_thres_AE_val:{optimal_AE_thres}, factor_AE_thres = {factor_AE_thres}, key = {key}')
    #
    # if show_flg:
    #     plot_loss(training_losses['loss'], training_losses['val_loss'], title_flg=title_flg,
    #               title=f'{experiment}, key:{key}')

    ### evaluation
    print(f'test AE...')
    print('--test AE on train set:')
    x_train = np.concatenate([train_set['x_norm_train'], train_set['x_attack_train']])
    y_train = np.concatenate([train_set['y_norm_train'], train_set['y_attack_train']])
    y_pred_label_AE, y_pred_probs_AE, _ = test_AE(x_test=x_train, y_test=y_train, experiment=experiment,
                                                  thres_AE=optimal_AE_thres,
                                                  in_dim=num_features, output_dir=output_dir)
    tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_true=y_train, y_pred=y_pred_label_AE)

    print('\nstep 3-2. evaluate AE on experiment:', optimal_AE_thres)
    y_test_pred_prob_dict = {'AE': []}

    print('--test AE on test set:')
    for key, value_dict in test_sets_dict.items():
        x_test = np.concatenate([value_dict['x_norm_test'], value_dict['x_attack_test']])
        y_test = np.concatenate([value_dict['y_norm_test'], value_dict['y_attack_test']])
        print(f'normalize {key}, x_test.shape: {x_test.shape}, y_test:{Counter(y_test.reshape(-1,))}')
        y_pred_label_AE, y_pred_probs_AE, reconstr_errs_arr = test_AE(x_test=x_test, y_test=y_test,
                                                                      experiment=experiment,
                                                                      thres_AE=optimal_AE_thres,
                                                                      in_dim=num_features, output_dir=output_dir)
        y_test_pred_prob_dict['AE'].append((y_test, y_pred_probs_AE))  # used for ROC, which need probability.

    if show_flg:  # show reconstruction errors of normal and attack samples in test set
        out_file = os.path.join(output_dir, f'figures/{experiment}_recon_err_of_{str(x_test.shape[1])}_features.txt')
        out_file = save_reconstruction_errors(reconstr_errs_arr, experiment, x_test.shape[1], out_file)
        title = os.path.split(out_file)[-1].split('.')[0]
        # if 'mawi' not in experiment.lower():  # for mawi, it does not has attack samples.
        plot_reconstruction_errors_from_txt(input_file=out_file, balance_data_flg=False,
                                            random_state=random_state,
                                            title_flg=title_flg, title=title)

    return y_test_pred_prob_dict, y_pred_label_AE, optimal_AE_thres


def normalize_all_data_experiment_3(train_set='train_set_dict', val_set='val_set_dict',
                       test_sets_dict=OrderedDict({'SYNT_test': '', 'UNB_test': '', 'MAWI_test': ''}),
                       selected_features_lst=[], norm_flg=norm_flg,
                       norm_method='z-score',
                       not_normalized_features_idx=[], verbose=verbose):
    """
    :param train_set_dict: {'x_norm_test': x_norm_test, 'y_norm_test': y_norm_test,
                               'x_attack_test': x_attack_test, 'y_attack_test': y_attack_test}
    :param selected_features_lst: if -1, use all features; otherwise, selected_features_lst = [5, 8, 10]
    :param norm_method: 'z-score' or 'min-max'
    :param not_normalized_features_idx: if [], normalized all features; otherwise, not_normalized_features_idx = [1, 3, 5]
    :return:
    """

    x_train = train_set['x_norm_train']
    x_val =    val_set['x_norm_val']

    # normlized_train_set_dict = OrderedDict({'x_norm_train': '', 'y_norm_train': '', 'x_attack_train': 'y_attack_train'})
    # normlized_val_set_dict = OrderedDict({'x_norm_val': '', 'y_norm_val': '', 'x_attack_val': 'y_attack_val'})
    normlized_train_set_dict = OrderedDict()
    normlized_val_set_dict = OrderedDict()
    normlized_test_sets_dict = OrderedDict()
    if norm_flg:
        print(f'if data will be normalized ?: {norm_flg}, norm_method: {norm_method}')
        if norm_method == 'z-score':
            x_train, x_train_mu, x_train_std = z_score_np(x_train, mu='', d_std='',
                                                          not_normalized_features_idx=not_normalized_features_idx)
            normlized_train_set_dict['x_norm_train'] = x_train
            normlized_train_set_dict['y_norm_train'] = train_set['y_norm_train']

            # no need to normlize x_attack_trian, because AE only uses normal samples to train.
            normlized_train_set_dict['x_attack_train'] = train_set['x_attack_train']
            normlized_train_set_dict['y_attack_train'] = train_set['y_attack_train']

            x_val_raw, _, _ = z_score_np(x_val, mu=x_train_mu, d_std=x_train_std,
                                         not_normalized_features_idx=not_normalized_features_idx)

            normlized_val_set_dict['x_norm_val'] = x_val_raw
            normlized_val_set_dict['y_norm_val'] = val_set['y_norm_val']

            # no need to normlize x_attack_trian, because AE only uses normal samples to train.
            x_val_raw, _, _ = z_score_np(val_set['x_attack_val'], mu=x_train_mu, d_std=x_train_std,
                                         not_normalized_features_idx=not_normalized_features_idx)
            normlized_val_set_dict['x_attack_val'] = x_val_raw
            normlized_val_set_dict['y_attack_val'] = val_set['y_attack_val']

            for key, value_dict in test_sets_dict.items():
                x_norm_test = value_dict['x_norm_test']
                y_norm_test = value_dict['y_norm_test']
                x_attack_test = value_dict['x_attack_test']
                y_attack_test = value_dict['y_attack_test']
                # print(f'normalize {key}, x_test.shape: {x_norm_test.shape}, y_test:{Counter(y_norm_test.reshape(-1,))}')
                x_norm_test, _, _ = z_score_np(x_norm_test, mu=x_train_mu, d_std=x_train_std,
                                              not_normalized_features_idx=not_normalized_features_idx)
                x_attack_test, _, _ = z_score_np(x_attack_test, mu=x_train_mu, d_std=x_train_std,
                                              not_normalized_features_idx=not_normalized_features_idx)
                normlized_test_sets_dict[key] ={'x_norm_test':x_norm_test, 'y_norm_test':y_norm_test,
                                                'x_attack_test':x_attack_test, 'y_attack_test':y_attack_test}
        elif norm_method == 'min-max':
            x_train, x_train_min, x_train_max = normalize_data_min_max(x_train, min_val='',
                                                                       max_val='',
                                                                       not_normalized_features_idx=not_normalized_features_idx)

            normlized_train_set_dict['x_norm_train'] = x_train
            normlized_train_set_dict['y_norm_train'] = train_set['y_norm_train']

            # no need to normlize x_attack_trian, because AE only uses normal samples to train.
            normlized_train_set_dict['x_attack_train'] = train_set['x_attack_train']
            normlized_train_set_dict['y_attack_train'] = train_set['y_attack_train']

            x_val_raw, _, _ = normalize_data_min_max(x_val, min_val=x_train_min,
                                                     max_val=x_train_max,
                                                     not_normalized_features_idx=not_normalized_features_idx)

            normlized_val_set_dict['x_norm_train'] = x_train
            normlized_val_set_dict['y_norm_train'] = train_set['y_norm_train']

            # no need to normlize x_attack_trian, because AE only uses normal samples to train.
            x_val_raw, _, _ = normalize_data_min_max(val_set['x_attack_val'], min_val=x_train_min,
                                                     max_val=x_train_max,
                                                     not_normalized_features_idx=not_normalized_features_idx)

            normlized_val_set_dict['x_attack_train'] = x_val_raw
            normlized_val_set_dict['y_attack_train'] = train_set['y_attack_train']

            for key, value_dict in test_sets_dict.items():
                x_norm_test = value_dict['x_norm_test']
                y_norm_test = value_dict['y_norm_test']
                x_attack_test = value_dict['x_attack_test']
                y_attack_test = value_dict['y_attack_test']
                # print(f'normalize {key}, x_test.shape: {x_norm_test.shape}, y_test:{Counter(y_norm_test.reshape(-1,))}')
                x_norm_test, _, _ = normalize_data_min_max(val_set['x_norm_test'], min_val=x_train_min,
                                                         max_val=x_train_max,
                                                         not_normalized_features_idx=not_normalized_features_idx)

                x_attack_test, _, _ = normalize_data_min_max(val_set['x_attack_test'], min_val=x_train_min,
                                                         max_val=x_train_max,
                                                         not_normalized_features_idx=not_normalized_features_idx)

                normlized_test_sets_dict[key] = {'x_norm_test': x_norm_test, 'y_norm_test': y_norm_test,
                                                 'x_attack_test': x_attack_test, 'y_attack_test': y_attack_test}
        else:
            print(f'norm_method {norm_method} is not correct.')
            return -1

        return normlized_train_set_dict, normlized_val_set_dict, normlized_test_sets_dict


def conduct_experiment_without_feature_selection(sub_experiment=('key', ''), train_set='train_set_dict',
                                                 val_set='val_set_dict',
                                                 test_sets_dict='test_sets_dict', output_dir='',
                                                 selected_features_lst=[],
                                                 Epochs=1, find_optimal_thres_flg=False, show_flg=True, verbose=1):
    print(f"-step 2. train and test AE without feature selection")
    num_features = train_set['x_norm_train'].shape[1]
    if len(selected_features_lst) > 0:
        print(f'using selected_features_lst (size={len(selected_features_lst)}): {selected_features_lst}')
        x_norm_train = select_sub_features_data(train_set['x_norm_train'], selected_features_lst)
        x_attack_train = select_sub_features_data(train_set['x_attack_train'], selected_features_lst)

        new_train_set_dict = {'x_norm_train': x_norm_train, 'y_norm_train': train_set['y_norm_train'],
                                         'x_attack_train': x_attack_train, 'y_attack_train': train_set['y_attack_train']}

        x_norm_val = select_sub_features_data(val_set['x_norm_val'], selected_features_lst)
        x_attack_val = select_sub_features_data(val_set['x_attack_val'], selected_features_lst)

        new_val_set_dict = {'x_norm_val': x_norm_val, 'y_norm_val': val_set['y_norm_val'],
                                         'x_attack_val': x_attack_val, 'y_attack_val': val_set['y_attack_val']}

        new_test_sets_dict=OrderedDict()
        for key, value_dict in test_sets_dict.items():
            x_norm_test = value_dict['x_norm_test']
            y_norm_test = value_dict['y_norm_test']
            x_attack_test = value_dict['x_attack_test']
            y_attack_test = value_dict['y_attack_test']
            # print(f'select sub features on {key}, x_test.shape: {x_test.shape}, y_test:{Counter(y_test)}')
            x_norm_test = select_sub_features_data(x_norm_test, selected_features_lst)
            x_attack_test = select_sub_features_data(x_attack_test, selected_features_lst)

            new_test_sets_dict[key] = {'x_norm_test': x_norm_test, 'y_norm_test': y_norm_test,
                                             'x_attack_test': x_attack_test, 'y_attack_test': y_attack_test}

    else:
        print(f'using all features (size={num_features})')
        new_train_set_dict = train_set
        new_val_set_dict = val_set
        new_test_sets_dict = test_sets_dict

    y_test_pred_proba_dict = {}  # save all results of y_test and y_pred_proba of AE, PCA, IF and OCSVM
    y_test_pred_proba_AE_dict, y_pred_AE_label, optimal_AE_thres = train_test_AE_experiment_3(experiment=sub_experiment,
                                                                                              train_set=new_train_set_dict,
                                                                                              val_set=new_val_set_dict,
                                                                                              test_sets_dict=new_test_sets_dict,
                                                                                              output_dir=output_dir,
                                                                                              Epochs=Epochs,
                                                                                              find_optimal_thres_flg=find_optimal_thres_flg,
                                                                                              verbose=verbose)
    y_test_pred_proba_dict.update(y_test_pred_proba_AE_dict)
    y_test_pred_proba_trad_ml_dict, _, _ = train_test_traditional_algorithms_experiment_3(experiment=sub_experiment,
                                                                                          train_set=new_train_set_dict,
                                                                                          val_set=new_val_set_dict,
                                                                                          test_sets_dict=new_test_sets_dict,
                                                                                          output_dir=output_dir,
                                                                                          verbose=verbose)
    y_test_pred_proba_dict.update(y_test_pred_proba_trad_ml_dict)
    out_file = f'output_data/figures/{sub_experiment[0]}-{sub_experiment[1]}_roc_data_of_{num_features}_features.txt'
    print(f'roc, out_file:{out_file}')
    # save_roc_to_txt(out_file, y_test_pred_proba_dict)
    # if show_flg:
    #     title = os.path.split(out_file)[-1].split('.')[0]
    #     plot_roc(out_file, balance_data_flg=True, title_flg=title_flg, title=title)


def conduct_experiment_3(sub_experiment=('key', 'value'), norm_flg=True, norm_method='z-score', Epochs=1,
                       optimal_AE_thres='', find_optimal_thres_flg=True, factor_AE_thres=1, selected_features_lst=[],
                       analyize_features_flg=False, output_dir='', random_state=42, verbose=1):
    print(f'-print sub_experiment {sub_experiment} parameters:\n'
          f'\'input_dir: {input_dir}, random_seed: {random_state}, normal_flg:{norm_flg}, '
          f'Epochs:{Epochs}, batch_size:{batch_size}, '
          f'optimal_AE_thres:{optimal_AE_thres}, find_optimal_thres_flg:{find_optimal_thres_flg}\'')

    print("-step 1. loading data.")
    train_set_dict, val_set_dict, test_sets_dict = \
        load_data_and_discretize_features(sub_experiment[-1], input_dir=input_dir, random_state=random_state)

    print("-step 1.1. preprocessing data.")
    train_set_dict, val_set_dict, test_sets_dict = \
        normalize_all_data_experiment_3(train_set=train_set_dict, val_set=val_set_dict,
                           test_sets_dict=test_sets_dict,
                           selected_features_lst=[], norm_flg=norm_flg,
                           norm_method=norm_method,
                           not_normalized_features_idx=[], verbose=verbose)

    conduct_experiment_without_feature_selection(sub_experiment=sub_experiment, train_set=train_set_dict,
                                                 val_set=val_set_dict,
                                                 test_sets_dict=test_sets_dict,
                                                 output_dir=output_dir,
                                                 Epochs=Epochs, find_optimal_thres_flg=find_optimal_thres_flg,
                                                 verbose=verbose)


def main():
    experiment_2_dict = OrderedDict(
        {'SYNT': 'uSc2C1_z-score_20_14', 'UNB': 'uSc2C2_z-score_20_14'})
    '''
        sub_experiemnt = 'uSc1C1_z-score_20_14'
        sub_experiemnt[0] = u/s for Unsupervised or Supervised
        sub_experiemnt[3] = Scenario: Experiment I, II, III
        sub_experiemnt[5] = Source: SYNT, UNB, MAWI
    '''
    key = 'debug'
    # key = 'all'
    if key == 'all':
        for sub_key, sub_value in experiment_2_dict.items():
            print(f'-experiment_1 => {sub_key}:{sub_value}')
            conduct_experiment_3(sub_experiment=(sub_key, sub_value), norm_flg=norm_flg, norm_method=norm_method,
                               Epochs=Epochs, output_dir=output_dir,
                               optimal_AE_thres=optimal_AE_thres, find_optimal_thres_flg=find_optimal_thres_flg,
                               factor_AE_thres=factor_AE_thres, analyize_features_flg=analyize_features_flg,
                               random_state=random_state, verbose=verbose
                               )
    elif key == 'debug':
        conduct_experiment_3(sub_experiment=('SYNT', 'uSc3C1_z-score_20_14'), norm_flg=norm_flg,
                           norm_method=norm_method,
                           Epochs=Epochs, output_dir=output_dir,
                           optimal_AE_thres=optimal_AE_thres, find_optimal_thres_flg=find_optimal_thres_flg,
                           factor_AE_thres=factor_AE_thres, analyize_features_flg=analyize_features_flg,
                           random_state=random_state, verbose=verbose
                           )
    else:
        pass


if __name__ == '__main__':
    main()
