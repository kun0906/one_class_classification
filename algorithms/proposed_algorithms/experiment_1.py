"""
    Experiment I

"""
from copy import deepcopy

from algorithms.proposed_algorithms.autoencoder import train_AE, test_AE
from algorithms.traditional_algorithms.classical_ml_algorithms import train_PCA, test_PCA, train_IF, test_IF, \
    train_OCSVM, test_OCSVM
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


def train_test_traditional_algorithms(experiment=('key', ''), x_train='', y_train='', x_val='', y_val='', x_test='',
                                      y_test='', x_attack='', y_attack='',
                                      output_dir='', show_flg=True, verbose=1):
    """

    :param verbose:
    :return:
    """
    print('\ndata is used to train and test traditional algorithms (PCA, OCSVM and IF) for {experiment}.')
    # print(f'all_norm:{x_train.shape[0]+x_val.shape[0]+x_test.shape[0]}, all_attack:{x_attack_1.shape[0]}')
    print(f'x_train:{x_train.shape}, y_train:{Counter(y_train.reshape(-1,).tolist())}')
    # print(f'x_val:{x_val.shape}, y_val:{Counter(y_val.reshape(-1,))}')
    print(f'x_test:{x_test.shape}, y_test:{Counter(y_test.reshape(-1,))}')
    if type(x_attack) != type(None) and type(y_attack) != type(None):
        print(f'x_remained_attack:{x_attack.shape}, y_remained_attack:{Counter(y_attack.reshape(-1,))}')

    experiment = experiment[0] + '-' + experiment[1]  # (key, value)
    y_test_pred_prob_dict = {'DT': [], 'PCA': [], 'IF': [], 'OCSVM': []}
    # ### 2. evaluate DT
    # # without attack data in training, so no DT model in this experiment.
    # print('\n2). no need to evaluate DT on test set')

    ### 1. evaluate PCA
    print('\n1). train and evaluate PCA...')
    train_PCA(x_train=x_train, experiment=experiment)
    print(f'x_train.shape: {x_train.shape}')
    test_PCA(x_test=x_train, y_test=y_train, experiment=experiment)

    print(f'x_test.shape: {x_test.shape}')
    y_pred_label_PCA, y_pred_probs_PCA = test_PCA(x_test=x_test, y_test=y_test, experiment=experiment)
    y_test_pred_prob_dict['PCA'] = (y_test, y_pred_probs_PCA)

    # # ### 2. evaluate IF
    # print('\n2). train and evaluate IF...')
    # train_IF(x_train=x_train, experiment=experiment)
    # print(f'x_train.shape: {x_train.shape}')
    # test_IF(x_test=x_train, y_test=y_train, experiment=experiment)
    #
    # print(f'x_test.shape: {x_test.shape}')
    # y_pred_label_IF, y_pred_probs_IF = test_IF(x_test=x_test, y_test=y_test, experiment=experiment)
    # y_test_pred_prob_dict['IF'] = (y_test, y_pred_probs_IF)

    # ### 3. evaluate OC-SVM
    # print('\n3). train and evaluate OCSVM...')
    # train_OCSVM(x_train=x_train, experiment=experiment)
    # print(f'x_train.shape: {x_train.shape}')
    # test_OCSVM(x_train[:100], y_train[:100], experiment)
    #
    # print(f'x_test.shape: {x_test.shape}')
    # y_pred_label_OCSVM, y_pred_probs_OCSVM = test_OCSVM(x_test=x_test, y_test=y_test, experiment=experiment)
    # y_test_pred_prob_dict['OCSVM'] = (y_test, y_pred_probs_OCSVM)

    return y_test_pred_prob_dict, '', ''


def train_test_AE(experiment=('key', []), train_set_dict ='', val_set_dict='', test_sets_dict='', Epochs=1, find_optimal_thres_flg=True, optimal_AE_thres=None,
                  factor_AE_thres='',
                  output_dir='output_data', show_flg=True, verbose=verbose):
    """

    :param verbose:
    :return:
    """
    print(f'data is used to train and test AE for {experiment}.')
    print_dict(train_set_dict)
    print_dict(val_set_dict)
    print_dict(test_sets_dict)

    experiment = experiment[0] + '-' + experiment[1]  # (key, value)
    print(f'-train AE on train set.')
    x_train = train_set_dict['train_set']['x']
    x_val = val_set_dict['val_set']['x']
    training_loss_dict = train_AE(train_set=train_set_dict['train_set'], val_set=val_set_dict['val_set'], experiment=experiment,
                               Epochs=Epochs,
                               output_dir=output_dir, verbose=verbose)

    if find_optimal_thres_flg:
        print(f'find the optimal threshold ({find_optimal_thres_flg}) for AE by using training losss')
        optimal_AE_thres, factor_AE_thres, key, = get_optimal_thres(training_loss_dict, factor_AE_thres=factor_AE_thres,
                                                                    key='loss')
    else:
        if type(optimal_AE_thres) == type(None):
            training_loss_value = training_loss_dict['loss'][-1]
            optimal_AE_thres = factor_AE_thres * training_loss_value
            print(f'using the threshold obtained from train set (optimal_AE_thres {optimal_AE_thres} = factor ({factor_AE_thres}) * training_loss ({training_loss_value}))')
        else:
            print(f'using the presetted threshold: {optimal_AE_thres}.')
            optimal_AE_thres = optimal_AE_thres
            key = None
            factor_AE_thres = None
    print(f'optimal_thres_AE_val:{optimal_AE_thres}, factor_AE_thres = {factor_AE_thres}, key = {key}')

    if show_flg:
        plot_loss(training_loss_dict['loss'], training_loss_dict['val_loss'], title_flg=title_flg,
                  title=f'{experiment}, key:{key}')

    ### evaluation
    print(f'test AE...')
    print('--test AE on train set:')
    y_pred_label_AE, y_pred_probs_AE, _ = test_AE(test_set=train_set_dict['train_set'], experiment=experiment,
                                                  thres_AE=optimal_AE_thres,
                                                  output_dir=output_dir)
    tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_true=train_set_dict['train_set']['y'], y_pred=y_pred_label_AE)

    print('\nstep 3-2. evaluate AE on experiment:', optimal_AE_thres)
    y_test_pred_prob_dict = {}
    y_pred_probs_AE_dict ={}
    optimal_AE_thres_dict ={}

    print('--test AE on test set:')
    for key, value_dict in test_sets_dict.items():
        y_pred_label_AE, y_pred_probs_AE, reconstr_errs_arr = test_AE(test_set=value_dict, experiment=experiment,
                                                                      thres_AE=optimal_AE_thres,
                                                                      output_dir=output_dir)
        x_test = value_dict['x']
        y_test = value_dict['y']
        tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_true=y_test, y_pred=y_pred_label_AE)
        y_test_pred_prob_dict[key] = {}
        y_test_pred_prob_dict[key]['AE'] = {}
        y_test_pred_prob_dict[key]['AE'] = (y_test, y_pred_label_AE, y_pred_probs_AE)  # used for ROC, which need probability.

        if show_flg:  # show reconstruction errors of normal and attack samples in test set
            out_file = os.path.join(output_dir, f'figures/{experiment}+test_set={key}+recon_err={str(x_test.shape[1])}_features.txt')
            out_file = save_reconstruction_errors(reconstr_errs_arr, experiment, x_test.shape[1], out_file)
            title = os.path.split(out_file)[-1].split('.')[0]
            if 'mawi' not in experiment.lower():  # for mawi, it does not has attack samples.
                plot_reconstruction_errors_from_txt(input_file=out_file, balance_data_flg=False,
                                                    random_state=random_state,
                                                    title_flg=title_flg, title=title)

    return y_test_pred_prob_dict, optimal_AE_thres


def conduct_experiment_without_feature_selection(sub_experiment=('key', ''), train_set_dict='', val_set_dict='',
                                                     test_sets_dict = '', output_dir='',
                                                 selected_features_lst=[], optimal_AE_thres='',
                                                 Epochs=1, find_optimal_thres_flg=False, show_flg=True, verbose=1):
    print(f"-step 2. train and test AE without feature selection")
    if len(selected_features_lst) > 0:
        print(f'using selected_features_lst (size={len(selected_features_lst)}): {selected_features_lst}')
        for value_dict in [train_set_dict, val_set_dict, test_sets_dict]:
            for key, value in value_dict.items():
                new_x = select_sub_features_data(value['x'], selected_features_lst)
                value_dict[key]['x'] = new_x
                value_dict[key]['y'] = value['y']
    else:
        num_features = train_set_dict['train_set']['x'].shape[1]
        print(f'using all features: {num_features}')

    """
        all_y_test_pred_proba_dict = {'test_set':{'AE':(y_true, y_pred_proba)}, 'test_set_2': {'AE':(,)}, {..}}
    """
    all_y_test_pred_proba_dict = {}  # save all results of y_test and y_pred_proba of AE, PCA, IF and OCSVM
    y_test_pred_proba_AE_dict, optimal_AE_thres = train_test_AE(experiment=sub_experiment,
                                                                                 train_set_dict=train_set_dict,
                                                                                 val_set_dict=val_set_dict,
                                                                                 test_sets_dict=test_sets_dict,
                                                                                 output_dir=output_dir,
                                                                                 Epochs=Epochs,
                                                                                 optimal_AE_thres=optimal_AE_thres,
                                                                                 find_optimal_thres_flg=find_optimal_thres_flg,
                                                                                 verbose=verbose)
    all_y_test_pred_proba_dict.update(y_test_pred_proba_AE_dict)
    y_test_pred_proba_trad_ml_dict, _ , _ = train_test_traditional_algorithms(experiment=sub_experiment,
                                                                              train_set_dict=train_set_dict,
                                                                              val_set_dict=val_set_dict,
                                                                              test_sets_dict=test_sets_dict,
                                                                  output_dir=output_dir, verbose=verbose)
    all_y_test_pred_proba_dict.update(y_test_pred_proba_trad_ml_dict)
    out_file = f'output_data/figures/{sub_experiment[0]}-{sub_experiment[1]}_roc_data_of_{x_train.shape[1]}_features.txt'
    print(f'roc, out_file:{out_file}')
    save_roc_to_txt(out_file, all_y_test_pred_proba_dict)
    if show_flg:
        for key, value_dict in all_y_test_pred_proba_dict.items():
            # title = os.path.split(out_file)[-1].split('.')[0]
            title = key
            plot_roc(value_dict, balance_data_flg=True, title_flg=title_flg, title=title)


def conduct_experiment_with_feature_selection(sub_experiment='',
                                                  train_set_dict='', val_set_dict='',test_sets_dict = '',
                                                  output_dir='', optimal_AE_thres='',
                                                  Epochs=Epochs, find_optimal_thres_flg=True,
                                                  verbose=1):
    ### Relation between the number of feature and FPR and FNR.
    sub_experiment_tuple = sub_experiment
    print(f"-step 2. analyize different number of features on sub_experiment:{sub_experiment}.")
    sub_experiment = sub_experiment[0] + '-' + sub_experiment[1]  # (key, value)

    x_train = train_set_dict['train_set']['x']
    all_sub_feature_sets_ascended_dict = conduct_feature_selection(x_train)
    out_file = f"output_data/figures/{sub_experiment}_sub_features_list.txt"
    out_file = save_features_selection_results(out_file, all_sub_feature_sets_ascended_dict)
    plot_features_selection_results(out_file, title_flg=True, title=f'feature_number')

    all_res_metrics_feat_list = []  # '', save all the sub_features results. each item is a dictionary.
    for idx, (key, value) in enumerate(all_sub_feature_sets_ascended_dict.items()):
        selected_features_lst = value
        if len(sub_features_lst) < 2:
            continue
        if len(selected_features_lst) % 10 != 0 :
            continue
        print(
            f'\n-{idx}/{len(all_sub_feature_sets_ascended_dict)}, conduct experiment on sub feature set: {selected_features_lst}')
        print(f'using selected_features_lst (size={len(selected_features_lst)}): {selected_features_lst}')

        new_train_set_dict = deepcopy(train_set_dict)
        new_val_set_dict = deepcopy(val_set_dict)
        new_test_sets_dict = deepcopy(test_sets_dict)

        for value_dict in [new_train_set_dict, new_val_set_dict, new_test_sets_dict]:
            for key, value in value_dict.items():
                new_x = select_sub_features_data(value['x'], selected_features_lst)
                value_dict[key]['x'] = new_x
                value_dict[key]['y'] = value['y']

        y_test_pred_proba_dict, optimal_AE_thres = train_test_AE(experiment=sub_experiment_tuple,
                                                                                  train_set_dict=new_train_set_dict,
                                                                                  val_set_dict=new_val_set_dict,
                                                                                  test_sets_dict=new_test_sets_dict,
                                                                                  output_dir=output_dir, Epochs=Epochs,
                                                                 optimal_AE_thres   = optimal_AE_thres,
                                                                                  find_optimal_thres_flg=find_optimal_thres_flg,
                                                                                  verbose=verbose)

        y_true = y_test_pred_proba_dict['test_set']['AE'][0]
        y_pred_label_AE = y_test_pred_proba_dict['test_set']['AE'][1]
        tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_true=y_true,
                                                     y_pred=y_pred_label_AE)  # confusion matrix just need y_pred_labels.
        ### 0) Relation between the number of feature and FPR and FNR.
        # print(f'\nsave the FPR and FNR reausts of AE with num ={num_features} features on test set.')
        res_metrics_feat_dict = {'tpr': [], 'fnr': [], 'fpr': [], 'tnr': [], 'acc': []}
        res_metrics_feat_dict['tpr'].append(tpr)
        res_metrics_feat_dict['fnr'].append(fnr)
        res_metrics_feat_dict['fpr'].append(fpr)
        res_metrics_feat_dict['tnr'].append(tnr)
        res_metrics_feat_dict['acc'].append(acc)
        all_res_metrics_feat_list.append([sub_features_lst, res_metrics_feat_dict, optimal_AE_thres])

    if show_flg:
        "Relation between the number of feature and FPR and FNR."
        out_file = f'output_data/figures/{sub_experiment}_all_num_features_res_metrics.txt'
        save_sub_features_res_metrics(out_file, all_res_metrics_feat_list, optimal_AE_thres)
        title = os.path.split(out_file)[-1].split('.')[0]
        plot_sub_features_metrics(out_file, title_flg=title_flg, title=title)


def conduct_experiment_1(sub_experiment=('key', 'value'), norm_flg=True, norm_method='z-score', Epochs=1,
                         optimal_AE_thres='', find_optimal_thres_flg=True, factor_AE_thres=1, selected_features_lst=[],
                         analyize_features_flg=False, random_state=42, verbose=1):
    print(f'-print sub_experiment {sub_experiment} parameters:\n'
          f'\'input_dir: {input_dir}, output_dir: {output_dir}, random_seed: {random_state}, normal_flg: {norm_flg}, '
          f'Epochs:{Epochs}, batch_size:{batch_size}, '
          f'optimal_AE_thres:{optimal_AE_thres}, find_optimal_thres_flg:{find_optimal_thres_flg}, '
          f'analyize_features_flg: {analyize_features_flg}\'')

    print("-step 1. loading data.")
    train_set_dict, val_set_dict, test_sets_dict = load_data_and_discretize_features(sub_experiment[-1],
                                                                                     input_dir=input_dir,
                                                                                     random_state=random_state)


    print("-step 1.1. preprocessing data.")
    train_set_dict, val_set_dict, test_sets_dict = normalize_all_data(train_set_dict, val_set_dict, test_sets_dict,
                                                                      selected_features_lst=[], norm_flg=norm_flg,
                                                                      norm_method=norm_method,
                                                                      not_normalized_features_idx=[], verbose=verbose)

    if not analyize_features_flg:  # without feature selection, so using all features to conduct experiment
        conduct_experiment_without_feature_selection(sub_experiment=sub_experiment,
                                                     train_set_dict=train_set_dict, val_set_dict=val_set_dict,
                                                     test_sets_dict = test_sets_dict, factor_AE_thres = factor_AE_thres,
                                                     output_dir=output_dir, optimal_AE_thres=optimal_AE_thres,
                                                     Epochs=Epochs, find_optimal_thres_flg=find_optimal_thres_flg,
                                                     verbose=verbose)
    else:  # do feature selection, so using different features to conduct experiment
        conduct_experiment_with_feature_selection(sub_experiment=sub_experiment,
                                                  train_set_dict=train_set_dict, val_set_dict=val_set_dict,
                                                  test_sets_dict = test_sets_dict,
                                                  factor_AE_thres = factor_AE_thres,
                                                  output_dir=output_dir, optimal_AE_thres=optimal_AE_thres,
                                                  Epochs=Epochs, find_optimal_thres_flg=find_optimal_thres_flg,
                                                  verbose=verbose)


def main():
    experiment_1_dict = OrderedDict(
        {'SYNT': 'uSc1C1_z-score_20_14', 'UNB': 'uSc1C2_z-score_20_14', 'MAWI': 'uSc1C3_z-score_20_14'})
    '''
        sub_experiemnt = 'uSc1C1_z-score_20_14'
        sub_experiemnt[0] = u/s for Unsupervised or Supervised
        sub_experiemnt[3] = Scenario: Experiment I, II, III
        sub_experiemnt[5] = Source: SYNT, UNB, MAWI
    '''
    # key = 'debug'
    key = 'all'
    if key == 'all':
        for sub_key, sub_value in experiment_1_dict.items():
            print(f'-experiment_1 => {sub_key}:{sub_value}')
            conduct_experiment_1(sub_experiment=(sub_key, sub_value), norm_flg=norm_flg, norm_method=norm_method,
                                 Epochs=Epochs,
                                 optimal_AE_thres=optimal_AE_thres, find_optimal_thres_flg=find_optimal_thres_flg,
                                 factor_AE_thres=factor_AE_thres, analyize_features_flg=analyize_features_flg,
                                 random_state=random_state, verbose=verbose
                                 )
    elif key == 'debug':
        conduct_experiment(sub_experiment=('SYNT', 'uSc2C1_z-score_20_14'), norm_flg=norm_flg, norm_method=norm_method,
                           Epochs=Epochs,
                           optimal_AE_thres=optimal_AE_thres, find_optimal_thres_flg=find_optimal_thres_flg,
                           factor_AE_thres=factor_AE_thres, analyize_features_flg=analyize_features_flg,
                           random_state=random_state, verbose=verbose
                           )
    else:
        pass


if __name__ == '__main__':
    main()
