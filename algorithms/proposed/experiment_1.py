"""
    Experiment I

"""
from copy import deepcopy

from algorithms.proposed.autoencoder import train_AE, test_AE
from algorithms.classical.ml_algs import train_PCA, test_PCA, train_IF, test_IF, \
    train_OCSVM, test_OCSVM
from configuration import *
from utils.dataloader import *
from preprocess.feature_selection import *
from preprocess.normalization import *

from utils.visualization import *


def train_test_classical_algs(experiment=('key', ''), datasets_dict='', params_dict=''):
    """
        "train and test classical machine learning algorithms (such as DT, PCA and IF)"
    :param verbose:
    :return:
    """
    print('\ndata is used to train and test traditional algorithms (PCA, OCSVM and IF) for {experiment}.')
    show_flg = params_dict['show_flg']
    verbose = params_dict['verbose']

    train_set_dict = datasets_dict['train_set_dict']
    val_set_dict = datasets_dict['val_set_dict']
    test_set_dict = datasets_dict['test_set_dict']

    experiment = experiment[0] + '-' + experiment[1]  # (key, value)
    y_test_pred_prob_dict = OrderedDict()
    # ### 2. evaluate DT
    # # without attack data in training, so no DT model in this experiment.
    # print('\n2). no need to evaluate DT on test set')

    ### 1. evaluate PCA
    print('\n1). train and evaluate PCA...')
    train_set = train_set_dict['train_set']
    train_PCA(train_set=train_set, experiment=experiment)
    # print(f'x_train.shape: {x_train.shape}')
    test_PCA(test_set=train_set, experiment=experiment)

    for key, value_dict in test_set_dict.items():
        x_test = value_dict['x']
        print(f'{key}, x_test.shape: {x_test.shape}')
        y_test = value_dict['y']
        y_pred_label_PCA, y_pred_probs_PCA = test_PCA(test_set=value_dict, experiment=experiment)
        y_test_pred_prob_dict[key] = {}
        y_test_pred_prob_dict[key]['PCA'] = (y_test, y_pred_label_PCA, y_pred_probs_PCA)

    # ### 2. evaluate IF
    print('\n2). train and evaluate IF...')
    train_IF(train_set=train_set, experiment=experiment)
    # print(f'x_train.shape: {x_train.shape}')
    test_IF(test_set = train_set, experiment=experiment)

    for key, value_dict in test_set_dict.items():
        x_test = value_dict['x']
        print(f'{key}, x_test.shape: {x_test.shape}')
        y_test = value_dict['y']
        y_pred_label_IF, y_pred_probs_IF = test_IF(test_set=value_dict, experiment=experiment)
        y_test_pred_prob_dict['IF'] = (y_test, y_pred_probs_IF)
        y_test_pred_prob_dict[key] = {}
        y_test_pred_prob_dict[key]['IF'] = (y_test, y_pred_label_IF, y_pred_probs_IF)

    # ### 3. evaluate OC-SVM
    # print('\n3). train and evaluate OCSVM...')
    # train_OCSVM(train_set=train_set, experiment=experiment)
    # # print(f'x_train.shape: {x_train.shape}')
    # test_OCSVM(test_set=train_set, experiment=experiment)
    #
    # for key, value_dict in test_set_dict.items():
    #     x_test = value_dict['x']
    #     print(f'{key}, x_test.shape: {x_test.shape}')
    #     y_test = value_dict['y']
    #     y_pred_label_OCSVM, y_pred_probs_OCSVM = test_OCSVM(test_set= value_dict, experiment=experiment)
    #     y_test_pred_prob_dict[key] = {}
    #     y_test_pred_prob_dict[key]['OCSVM'] = (y_test, y_pred_label_OCSVM, y_pred_probs_OCSVM)

    return y_test_pred_prob_dict, '', ''


def train_test_AE(experiment=('key', []), datasets_dict='', params_dict=''):
    """

    :param verbose:
    :return:
    """
    print(f'data is used to train and test AE for {experiment}.')
    AE_params_dict = params_dict['AE']
    show_flg = params_dict['show_flg']
    verbose = params_dict['verbose']

    train_set_dict = datasets_dict['train_set_dict']
    val_set_dict = datasets_dict['val_set_dict']
    test_set_dict = datasets_dict['test_set_dict']

    print_dict(train_set_dict)
    print_dict(val_set_dict)
    print_dict(test_set_dict)

    experiment = experiment[0] + '-' + experiment[1]  # (key, value)
    print(f'-train AE on train set.')
    x_train = train_set_dict['train_set']['x']
    x_val = val_set_dict['val_set']['x']
    output_dir = params_dict['output_dir']
    training_loss_dict = train_AE(train_set=train_set_dict['train_set'], val_set=val_set_dict['val_set'],
                                  experiment=experiment,
                                  Epochs=AE_params_dict['Epochs'],
                                  output_dir=output_dir, verbose=verbose)

    factor_AE_thres = AE_params_dict['factor_AE_thres']
    optimal_AE_thres = AE_params_dict['optimal_AE_thres']
    if AE_params_dict['find_optimal_thres_flg']:
        print(f'find the optimal threshold ({find_optimal_thres_flg}) for AE by using training losss')
        optimal_AE_thres, factor_AE_thres, key, = get_optimal_thres(training_loss_dict, factor_AE_thres=factor_AE_thres,
                                                                    key='loss')
    else:
        if type(optimal_AE_thres) == type(None):
            training_loss_value = training_loss_dict['loss'][-1]
            optimal_AE_thres = factor_AE_thres * training_loss_value
            print(
                f'using the threshold obtained from train set (optimal_AE_thres {optimal_AE_thres} = factor ({factor_AE_thres}) * training_loss ({training_loss_value}))')
        else:
            print(f'using the presetted threshold: {optimal_AE_thres}.')
            optimal_AE_thres = optimal_AE_thres
            key = None
            factor_AE_thres = None
    print(f'optimal_thres_AE_val:{optimal_AE_thres}, factor_AE_thres = {factor_AE_thres}, key = {key}')

    if show_flg:
        plot_loss(training_loss_dict['loss'], training_loss_dict['val_loss'], title_flg=params_dict['title_flg'],
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
    y_pred_probs_AE_dict = {}
    optimal_AE_thres_dict = {}

    print('--test AE on test set:')
    for key, value_dict in test_set_dict.items():
        print(f'\n-----test on \'{key}\'')
        y_pred_label_AE, y_pred_probs_AE, reconstr_errs_arr = test_AE(test_set=value_dict, experiment=experiment,
                                                                      thres_AE=optimal_AE_thres,
                                                                      output_dir=output_dir)
        x_test = value_dict['x']
        y_test = value_dict['y']
        tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_true=y_test, y_pred=y_pred_label_AE)
        y_test_pred_prob_dict[key] = {}
        y_test_pred_prob_dict[key]['AE'] = {}
        y_test_pred_prob_dict[key]['AE'] = (
            y_test, y_pred_label_AE, y_pred_probs_AE)  # used for ROC, which need probability.

        if show_flg:  # show reconstruction errors of normal and attack samples in test set
            out_file = os.path.join(output_dir,
                                    f'figures/{experiment}+test_set={key}+recon_err={str(x_test.shape[1])}_features.txt')
            out_file = save_reconstruction_errors(reconstr_errs_arr, experiment, x_test.shape[1], out_file)
            title = os.path.split(out_file)[-1].split('.')[0]
            if 'mawi' not in experiment.lower():  # for mawi, it does not has attack samples.
                plot_reconstruction_errors_from_txt(input_file=out_file, balance_data_flg=False,
                                                    random_state=random_state,
                                                    title_flg=params_dict['title_flg'], title=title)

    return y_test_pred_prob_dict, optimal_AE_thres


def conduct_experiment_without_feature_selection(sub_experiment=('key', ''), datasets_dict={}, params_dict={}):
    """

    :param sub_experiment:
    :param datasets_dict:
    :param params_dict:
    :return:
    """

    print(f"-step 2. train and test AE without feature selection")
    selected_features_lst = params_dict['selected_features_lst']
    if len(selected_features_lst) > 0:
        print(f'using selected_features_lst (size={len(selected_features_lst)}): {selected_features_lst}')
        for key_set, value_dict in datasets_dict.items():
            for sub_key, sub_value in value_dict.items():
                new_x = select_sub_features_data(sub_value['x'], selected_features_lst)
                value_dict[sub_key]['x'] = new_x
                value_dict[sub_key]['y'] = sub_value['y']
        num_features = len(selected_features_lst)
    else:
        num_features = datasets_dict['train_set_dict']['train_set']['x'].shape[1]
        print(f'using all features: {num_features}')

    """
        all_y_test_pred_proba_dict = {'test_set':{'AE':(y_true, y_pred_proba)}, 'test_set_2': {'AE':(,)}, {..}}
    """
    all_y_test_pred_proba_dict = {}  # save all results of y_test and y_pred_proba of AE, PCA, IF and OCSVM
    y_test_pred_proba_AE_dict, optimal_AE_thres = train_test_AE(experiment=sub_experiment, datasets_dict=datasets_dict,
                                                                params_dict=params_dict)

    y_test_pred_proba_trad_ml_dict, _, _ = train_test_classical_algs(experiment=sub_experiment,
                                                                             datasets_dict=datasets_dict,
                                                                             params_dict=params_dict)

    for key, value in y_test_pred_proba_AE_dict.items():
        all_y_test_pred_proba_dict[key]={}
        all_y_test_pred_proba_dict[key].update(value)
        all_y_test_pred_proba_dict[key].update(y_test_pred_proba_trad_ml_dict[key])

    out_file = f'output_data/figures/{sub_experiment[0]}-{sub_experiment[1]}_roc_data_of_{num_features}_features.txt'
    print(f'roc, out_file:{out_file}')
    save_roc_to_txt(out_file, all_y_test_pred_proba_dict)
    if params_dict['show_flg']:
        for key, value_dict in all_y_test_pred_proba_dict.items():
            # title = os.path.split(out_file)[-1].split('.')[0]
            title = key
            out_file= os.path.splitext(out_file)[0] + '_'+ key +'_roc.pdf'
            plot_roc(value_dict, out_file=out_file, balance_data_flg=True, title_flg=params_dict['title_flg'], title=title)


def conduct_experiment_with_feature_selection(sub_experiment='', datasets_dict='datasets_dict', params_dict=''):
    """

    :param sub_experiment:
    :param datasets_dict:
    :param params_dict:
    :return:
    """
    train_set_dict = datasets_dict['train_set_dict']
    val_set_dict = datasets_dict['val_set_dict']
    test_set_dict = datasets_dict['test_set_dict']

    sub_experiment_tuple = sub_experiment
    print(f"-step 2. analyize different number of features on sub_experiment:{sub_experiment}.")
    sub_experiment = sub_experiment[0] + '-' + sub_experiment[1]  # (key, value)

    x_train = train_set_dict['train_set']['x']
    all_sub_feature_sets_ascended_dict = conduct_feature_selection(x_train)
    out_file = f"output_data/figures/{sub_experiment}_sub_features_list.txt"
    out_file = save_features_selection_results(out_file, all_sub_feature_sets_ascended_dict)
    plot_features_selection_results(out_file, title_flg=True, title=f'feature_number')

    all_res_metrics_feat_list = []  # '', save all the sub_features results. each item is a dictionary.
    for idx, (key, value_selected_features_lst) in enumerate(all_sub_feature_sets_ascended_dict.items()):
        if len(value_selected_features_lst) < 2:
            continue
        if len(value_selected_features_lst) % 10 != 0:
            continue
        print(
            f'\n-{idx}/{len(all_sub_feature_sets_ascended_dict)}, conduct experiment on sub feature set: {value_selected_features_lst}')
        # print(f'using selected_features_lst (size={len(selected_features_lst)}): {selected_features_lst}')

        new_train_set_dict = deepcopy(train_set_dict)
        new_val_set_dict = deepcopy(val_set_dict)
        new_test_sets_dict = deepcopy(test_set_dict)

        for value_dict in [new_train_set_dict, new_val_set_dict, new_test_sets_dict]:
            for key, value in value_dict.items():
                new_x = select_sub_features_data(value['x'], value_selected_features_lst)
                value_dict[key]['x'] = new_x
                value_dict[key]['y'] = value['y']

        y_test_pred_proba_dict, optimal_AE_thres = train_test_AE(experiment=sub_experiment_tuple,
                                                                 datasets_dict=datasets_dict, params_dict=params_dict)

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
        all_res_metrics_feat_list.append([value_selected_features_lst, res_metrics_feat_dict, optimal_AE_thres])

    if params_dict['show_flg']:
        "Relation between the number of feature and FPR and FNR."
        out_file = f'output_data/figures/{sub_experiment}_all_num_features_res_metrics.txt'
        save_sub_features_res_metrics(out_file, all_res_metrics_feat_list, optimal_AE_thres)
        title = os.path.split(out_file)[-1].split('.')[0]
        plot_sub_features_metrics(out_file, title_flg=params_dict['title_flg'], title=title)


def conduct_experiment_1(sub_experiment={}, params_dict=''):
    """

    :param sub_experiment:  sub_experiment_dict = {'experiment_name': sub_key, 'experiment_value':sub_value, 'train_set_name':'unb', 'val_set_name':'unb'}
    :param params_dict:
    :return:
    """
    print(f'\n-conduct sub_experiment {sub_experiment}')

    print("-step 1. loading data")
    datasets_dict = load_data_and_discretize_features(sub_experiment[-1], input_dir=params_dict['input_dir'],
                                                      random_state=params_dict['random_state'])

    print("-step 1.1. preprocessing data")
    datasets_dict = normalize_all_data(datasets_dict=datasets_dict, params_dict=params_dict)

    analyize_features_flg = params_dict['analyize_features_flg']
    if not analyize_features_flg:  # without feature selection, so using all features to conduct experiment
        conduct_experiment_without_feature_selection(sub_experiment=sub_experiment, datasets_dict=datasets_dict,
                                                     params_dict=params_dict)
    else:  # do feature selection, so using different features to conduct experiment
        conduct_experiment_with_feature_selection(sub_experiment=sub_experiment, datasets_dict=datasets_dict,
                                                  params_dict=params_dict)

