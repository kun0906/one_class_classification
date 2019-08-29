# Authors: xxx <xxx@xxx>
#
# License: xxx

"""Main function:
Evaluate autoencoder and classical machine learning algorithms on datasets in offline

"""

from algorithms.classical.ml_algs import train_PCA, test_PCA, train_IF, test_IF
from algorithms.proposed.autoencoder import train_AE, test_AE
from configuration import *
from preprocess.feature_selection import *
from preprocess.normalization import *
from utils.balance import balance_dict
from utils.dataloader import *
from utils.visualization import *


def train_test_classical_algs(experiment_dict={}, datasets_dict='', params_dict=''):
    """Train and test classical algorithms (such as PCA and OCSVM) on dataset


    Args:
       experiment_dict: {'experiment_name': sub_key, 'experiment_value': sub_value, 'train_set_name': 'unb',
                               'val_set_name': 'unb'}
       datasets_dict: {'train_set_dict':, 'val_set_dict':, 'test_set_dict':,}
       params_dict:   all parameters used in this experiment

    Returns:
       classical_results_dict: {}
       optimal_AE_thres:
       y_pred:

    """
    print(f'\ndata is used to train and test traditional algorithms (PCA, OCSVM and IF) for {experiment_dict}.')
    # show_flg = params_dict['show_flg']
    # verbose = params_dict['verbose']

    train_set_name = experiment_dict['train_set_name']
    val_set_name = experiment_dict['val_set_name']

    train_set_dict = datasets_dict['train_set_dict']
    val_set_dict = datasets_dict['val_set_dict']
    test_set_dict = datasets_dict['test_set_dict']

    experiment_str = ''
    for key, value in experiment_dict.items():
        experiment_str += value + '_'

    y_test_pred_prob_dict = OrderedDict()
    # ### 2. evaluate DT
    # # without attack data in training, so no DT model in this experiment.
    # print('\n2). no need to evaluate DT on test set')

    ### 1. evaluate PCA
    print('\n1). train and evaluate PCA...')
    train_set = train_set_dict[train_set_name]
    train_PCA(train_set=train_set, experiment=experiment_str)
    # print(f'x_train.shape: {x_train.shape}')
    test_PCA(test_set=train_set, experiment=experiment_str)

    for key, value_dict in test_set_dict.items():
        if params_dict['balance_data_flg']:
            value_dict = balance_dict(value_dict)
        x_test = value_dict['X']
        print(f'{key}, x_test.shape: {x_test.shape}')
        y_test = value_dict['y']
        y_pred_label_PCA, y_pred_probs_PCA = test_PCA(test_set=value_dict, experiment=experiment_str)
        if key not in y_test_pred_prob_dict.keys():
            y_test_pred_prob_dict[key] = {}
            y_test_pred_prob_dict[key]['PCA'] = (y_test, y_pred_label_PCA, y_pred_probs_PCA)

    # ### 2. evaluate IF
    print('\n2). train and evaluate IF...')
    train_IF(train_set=train_set, experiment=experiment_str)
    # print(f'x_train.shape: {x_train.shape}')
    test_IF(test_set=train_set, experiment=experiment_str)

    for key, value_dict in test_set_dict.items():
        if params_dict['balance_data_flg']:
            value_dict = balance_dict(value_dict)
        x_test = value_dict['X']
        print(f'{key}, x_test.shape: {x_test.shape}')
        y_test = value_dict['y']
        y_pred_label_IF, y_pred_probs_IF = test_IF(test_set=value_dict, experiment=experiment_str)
        if key not in y_test_pred_prob_dict.keys():
            y_test_pred_prob_dict[key] = {}
            y_test_pred_prob_dict[key]['IF'] = (y_test, y_pred_label_IF, y_pred_probs_IF)
        else:
            y_test_pred_prob_dict[key]['IF'] = (y_test, y_pred_label_IF, y_pred_probs_IF)

    # ### 3. evaluate OC-SVM
    # print('\n3). train and evaluate OCSVM...')
    # train_OCSVM(train_set=train_set, experiment=experiment)
    # # print(f'x_train.shape: {x_train.shape}')
    # test_OCSVM(test_set=train_set, experiment=experiment)
    #
    # for key, value_dict in test_set_dict.items():
    #     if params_dict['balance_data_flg']:
    #         value_dict = balance_dict(value_dict)
    #     x_test = value_dict['X']
    #     print(f'{key}, x_test.shape: {x_test.shape}')
    #     y_test = value_dict['y']
    #     y_pred_label_OCSVM, y_pred_probs_OCSVM = test_OCSVM(test_set= value_dict, experiment=experiment)
    #     if key not in y_test_pred_prob_dict.keys():
    #         y_test_pred_prob_dict[key] = {}
    #         y_test_pred_prob_dict[key]['OCSVM'] = (y_test, y_pred_label_OCSVM, y_pred_probs_OCSVM)
    #     else:
    #         y_test_pred_prob_dict[key]['OCSVM'] = (y_test, y_pred_label_OCSVM, y_pred_probs_OCSVM)

    return y_test_pred_prob_dict, '', ''


def train_test_AE(experiment_dict={}, datasets_dict='', params_dict=''):
    """Train and test AE on dataset


    Args:
       experiment_dict: {'experiment_name': sub_key, 'experiment_value': sub_value, 'train_set_name': 'unb',
                               'val_set_name': 'unb'}
       datasets_dict: {'train_set_dict':, 'val_set_dict':, 'test_set_dict':,}
       params_dict:   all parameters used in this experiment

    Returns:
       AE_results_dict: {}
       optimal_AE_thres:
       y_pred:

    """
    print(f'\ndata is used to train and test AE for {experiment_dict}.')
    train_set_name = experiment_dict['train_set_name']
    val_set_name = experiment_dict['val_set_name']

    AE_params_dict = params_dict['AE']
    show_flg = params_dict['show_flg']
    verbose = params_dict['verbose']

    train_set_dict = datasets_dict['train_set_dict']
    val_set_dict = datasets_dict['val_set_dict']
    test_set_dict = datasets_dict['test_set_dict']

    print_dict(train_set_dict)
    print_dict(val_set_dict)
    print_dict(test_set_dict)

    experiment_str = ''
    for key, value in experiment_dict.items():
        experiment_str += value + '_'

    print(f'-train AE on train set. experiment_str:{experiment_str}')
    output_dir = params_dict['output_dir']
    training_loss_dict = train_AE(train_set=train_set_dict[train_set_name], val_set=val_set_dict[val_set_name],
                                  experiment=experiment_dict,
                                  Epochs=AE_params_dict['Epochs'],
                                  AE_params_dict=params_dict['AE'],
                                  output_dir=output_dir, verbose=verbose)

    factor_AE_thres = AE_params_dict['factor_AE_thres']
    optimal_AE_thres = AE_params_dict['optimal_AE_thres']
    find_optimal_thres_flg = AE_params_dict['find_optimal_thres_flg']
    if find_optimal_thres_flg:
        print(f'find the optimal threshold ({find_optimal_thres_flg}) for AE by using training losss')
        optimal_AE_thres, factor_AE_thres, key, = get_optimal_thres(training_loss_dict, factor_AE_thres=factor_AE_thres,
                                                                    key='loss')

        for i, thres_v in enumerate(np.linspace(optimal_AE_thres / factor_AE_thres, optimal_AE_thres, factor_AE_thres)):
            print(f'idx: {i} --test AE on train set: \'{train_set_name}\', with optimal_AE_thres:{thres_v}')
            y_pred_label_AE, y_pred_probs_AE, _ = test_AE(test_set=train_set_dict[train_set_name],
                                                          experiment=experiment_dict,
                                                          thres_AE=thres_v, AE_params_dict=params_dict['AE'],
                                                          output_dir=output_dir)
            tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_true=train_set_dict[train_set_name]['y'],
                                                         y_pred=y_pred_label_AE)
    else:
        if type(optimal_AE_thres) == type(None):
            training_loss_value = training_loss_dict['loss'][-1]
            optimal_AE_thres = factor_AE_thres * training_loss_value
            print(
                f'using the threshold obtained from train set (optimal_AE_thres {optimal_AE_thres} = factor ({factor_AE_thres}) * training_loss ({training_loss_value}))')
        else:
            print(f'using the presetted threshold in params_dict[\'optimal_AE_thres\']: {optimal_AE_thres}.')
            optimal_AE_thres = optimal_AE_thres
            key = None
            factor_AE_thres = None
    print(f'optimal_thres_AE_val:{optimal_AE_thres}, factor_AE_thres = {factor_AE_thres}, key = {key}')

    if show_flg:
        plot_loss(training_loss_dict['loss'], training_loss_dict['val_loss'], title_flg=params_dict['title_flg'],
                  title=f'{experiment_str}, key:{key}')

    ### evaluation
    print(f'test AE...')
    print(f'--test AE on train set: \'{train_set_name}\'')
    y_pred_label_AE, y_pred_probs_AE, _ = test_AE(test_set=train_set_dict[train_set_name], experiment=experiment_dict,
                                                  thres_AE=optimal_AE_thres, AE_params_dict=params_dict['AE'],
                                                  output_dir=output_dir)
    tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_true=train_set_dict[train_set_name]['y'], y_pred=y_pred_label_AE)

    # print('\nstep 3-2. evaluate AE on experiment:', optimal_AE_thres)
    y_test_pred_prob_dict = {}
    y_pred_probs_AE_dict = {}
    optimal_AE_thres_dict = {}

    print('--test AE on test set:')
    for key, value_dict in test_set_dict.items():
        print(f'\n-----test on \'{key}\'')
        if params_dict['balance_data_flg']:
            print(f'--balance \'{key}\'')
            value_dict = balance_dict(value_dict)

        y_pred_label_AE, y_pred_probs_AE, reconstr_errs_arr = test_AE(test_set=value_dict, experiment=experiment_dict,
                                                                      thres_AE=optimal_AE_thres,
                                                                      AE_params_dict=params_dict['AE'],
                                                                      output_dir=output_dir)
        x_test = value_dict['X']
        y_test = value_dict['y']
        tpr, fnr, fpr, tnr, acc = calucalate_metrics(y_true=y_test, y_pred=y_pred_label_AE)
        y_test_pred_prob_dict[key] = {}
        y_test_pred_prob_dict[key]['AE'] = {}
        y_test_pred_prob_dict[key]['AE'] = (
            y_test, y_pred_label_AE, y_pred_probs_AE)  # used for ROC, which need probability.

        if show_flg:  # show reconstruction errors of normal and attack samples in test set
            out_file = os.path.join(output_dir,
                                    f'figures/{experiment_str}={key}+recon_err={str(x_test.shape[1])}_features.txt')
            out_file = save_reconstruction_errors(reconstr_errs_arr, experiment_str, x_test.shape[1], out_file)
            print(f'out_file:{out_file}')
            title = key
            if 'mawi' not in experiment_str.lower():  # for mawi, it does not has attack samples.
                plot_reconstruction_errors_from_txt(input_file=out_file, balance_data_flg=False,
                                                    random_state=random_state,
                                                    title_flg=params_dict['title_flg'], title=title)

    return y_test_pred_prob_dict, optimal_AE_thres


def conduct_experiment_without_feature_analysis(experiment_dict={}, datasets_dict={}, params_dict={}):
    """

    :param sub_experiment:  sub_experiment_dict = {'experiment_name': sub_key, 'experiment_value':sub_value, 'train_set_name':'unb', 'val_set_name':'unb'}
    :param datasets_dict:
    :param params_dict:
    :return:
    """
    print('-preprocessing data')
    datasets_dict = preprocessing_data(datasets_dict=datasets_dict, local_params_dict=experiment_dict,
                                       params_dict=params_dict)

    print(f"-train and test AE without feature selection: {experiment_dict}")
    # train_set_name = experiment_dict['train_set_name'] + '_train_set'
    """
        all_y_test_pred_proba_dict = {'test_set':{'AE':(y_true, y_pred_proba)}, 'test_set_2': {'AE':(,)}, {..}}
    """
    all_results_dict = {}  # save all results of y_test and y_pred_proba of AE, PCA, IF and OCSVM
    AE_results_dict, optimal_AE_thres = train_test_AE(experiment_dict=experiment_dict,
                                                      datasets_dict=datasets_dict,
                                                      params_dict=params_dict)

    classical_ml_results_dict, _, _ = train_test_classical_algs(experiment_dict=experiment_dict,
                                                                datasets_dict=datasets_dict,
                                                                params_dict=params_dict)

    for key, value in AE_results_dict.items():
        if key not in all_results_dict.keys():
            all_results_dict[key] = {}
        all_results_dict[key].update(value)
        all_results_dict[key].update(classical_ml_results_dict[key])

    experiment_str = ''
    for key, value in experiment_dict.items():
        experiment_str += value + '_'

    num_features = datasets_dict['train_set_dict'][experiment_dict['train_set_name']]['X'].shape[1]
    out_file = f'output_data/figures/{experiment_str}_roc_data_of_{num_features}_features.txt'
    print(f'roc, out_file:{out_file}')
    save_roc_to_txt(out_file, all_results_dict)
    if params_dict['show_flg']:
        for key, value_dict in all_results_dict.items():
            # title = os.path.split(out_file)[-1].split('.')[0]
            title = key
            new_out_file = os.path.splitext(out_file)[0] + '_' + key + '_roc.pdf'
            plot_roc(value_dict, out_file=new_out_file, balance_data_flg=True, title_flg=params_dict['title_flg'],
                     title=title)


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

    x_train = train_set_dict[train_set_name]['X']
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
                new_x = select_sub_features_data(value['X'], value_selected_features_lst)
                value_dict[key]['X'] = new_x
                value_dict[key]['y'] = value['y']

        AE_result_dict, optimal_AE_thres = train_test_AE(experiment=sub_experiment_tuple,
                                                         datasets_dict=datasets_dict, params_dict=params_dict)

        y_true = AE_result_dict['test_set']['AE'][0]
        y_pred_label_AE = AE_result_dict['test_set']['AE'][1]
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
        all_results_lst.append([value_selected_features_lst, res_metrics_feat_dict, optimal_AE_thres])

    if params_dict['show_flg']:
        "Relation between the number of feature and FPR and FNR."
        out_file = f'output_data/figures/{sub_experiment}_all_num_features_res_metrics.txt'
        save_sub_features_res_metrics(out_file, all_results_lst, optimal_AE_thres)
        title = os.path.split(out_file)[-1].split('.')[0]
        plot_sub_features_metrics(out_file, title_flg=params_dict['title_flg'], title=title)


def preprocessing_data(datasets_dict, local_params_dict='', params_dict=''):
    """

    :param datasets_dict:
    :param local_params_dict:
    :param params_dict:
    :return:
    """
    train_set_name = local_params_dict['train_set_name']  # synt_train_set.
    norm_flg = params_dict['norm_flg']
    if norm_flg:
        datasets_dict = normalize_all_data(datasets_dict=datasets_dict,
                                           train_set_name=train_set_name,
                                           params_dict=params_dict)

    selected_features_lst = params_dict['selected_features_lst']
    # print(f"using selected feature lst: {selected_features_lst}. If selected_features_lst =[], using all features.")
    if len(selected_features_lst) > 0:
        print(f'using selected_features_lst (size={len(selected_features_lst)}): {selected_features_lst}')
        for key_set, value_dict in datasets_dict.items():
            for sub_key, sub_value in value_dict.items():
                new_x = select_sub_features_data(sub_value['X'], selected_features_lst)
                value_dict[sub_key]['X'] = new_x
                value_dict[sub_key]['y'] = sub_value['y']
        # num_features = len(selected_features_lst)
    else:
        num_features = datasets_dict['train_set_dict'][train_set_name]['X'].shape[1]
        print(f'using all features: {num_features}')

    return datasets_dict


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
            train_set_dict ={'train_set':{'X':, 'y':,}}
            val_set_dict ={'val_set':{'X':, 'y':,}}
            test_set_dict ={'test_set':{'X':, 'y':,},'test_set_2':{'X':, 'y':,},... }

    :return:
    """

    print(f'step 1. load params_dict:{params_dict}')

    experiments_dict = OrderedDict({
        'experiment_1': OrderedDict(
            {'SYNT': 'uSc1C1_20_14', 'UNB': 'uSc1C2_20_14', 'MAWI': 'uSc1C3_20_14'}),
        'experiment_2': OrderedDict(
            {'SYNT': 'uSc2C1_20_14', 'UNB': 'uSc2C2_20_14', 'MAWI': 'uSc2C3_20_14'}),
        'experiment_3': OrderedDict(
            {'SYNT': 'uSc3C1_20_14', 'UNB': 'uSc3C2_20_14', }),
    })
    '''
        sub_experiemnt = 'uSc1C1_20_14'
        sub_experiemnt[0] = u/s for Unsupervised or Supervised
        sub_experiemnt[3] = Scenario: Experiment 1, 2, 3
        sub_experiemnt[5] = Source: SYNT, UNB, MAWI
    '''

    experiment_name = 'experiment_1'  # conduct experiment 1
    dataset_key = 'UNB'  # train and validate on 'SYNT' data, test on all three datasets (SYNT, UNB, MAWI).
    local_params_dict = OrderedDict(
        {'experiment_name': experiment_name,
         'experiment_value': experiments_dict[experiment_name][dataset_key],
         'train_set_name': dataset_key + '_train_set',
         'val_set_name': dataset_key + '_val_set'
         })

    print(f'\nload local_params_dict: {local_params_dict}')

    print("\nstep 2: loading data")
    datasets_dict = get_dataset(local_params_dict['experiment_value'], input_dir=params_dict['input_dir'],
                                random_state=params_dict['random_state'])

    if not params_dict['analyze_feature_flg']:  # without feature selection, so using all features to conduct experiment
        print("\nstep 3: train and test detection models on data")
        conduct_experiment_without_feature_analysis(experiment_dict=local_params_dict, datasets_dict=datasets_dict,
                                                    params_dict=params_dict)
    else:  # do feature selection, so using different features to conduct experiment
        conduct_experiment_with_feature_selection(experiment_dict=local_params_dict, datasets_dict=datasets_dict,
                                                  params_dict=params_dict)

    # conduct_experiment(experiment_dict=local_params_dict, params_dict=params_dict)

    # def obtain_experiment_parameters(params_str='demo:experiment_1:SYNT:'):
    #
    #     if experiment_key == 'experiment_1':
    #         if dataset_key == 'SYNT':
    #             train_set_name = 'synt'   # train on synt_train_set, validate on synt_val_set
    #             val_set_name= 'synt'
    #             experiment_dict = OrderedDict(
    #                 {'experiment_name': experiment_key, 'experiment_value': experiment_value, 'train_set_name': train_set_name,
    #                  'val_set_name': val_set_name})
    #
    #     return experiment_dict
    #     # elif sub_key == 'experiment_2':
    #     #     if sub_value[5] == '1':
    #     #         sub_experiment_dict = OrderedDict(
    #     #             {'experiment_name': sub_key, 'experiment_value': sub_value, 'train_set_name': 'synt',
    #     #              'val_set_name': 'synt'})
    #     #     elif sub_value[5] =='2':
    #     #         sub_experiment_dict = OrderedDict(
    #     #             {'experiment_name': sub_key, 'experiment_value': sub_value, 'train_set_name': 'unb',
    #     #              'val_set_name': 'unb'})
    #     #     elif sub_value[5] =='3':
    #     #         sub_experiment_dict = OrderedDict(
    #     #             {'experiment_name': sub_key, 'experiment_value': sub_value, 'train_set_name': 'mawi',
    #     #              'val_set_name': 'mawi'})
    #     # elif sub_key == 'experiment_3':
    #     #     if sub_value[5] == 'synt':
    #     #         sub_experiment_dict = OrderedDict(
    #     #             {'experiment_name': sub_key, 'experiment_value': sub_value, 'train_set_name': 'synt',
    #     #              'val_set_name': 'synt'})
    #     #     elif sub_value[5] == 'unb':
    #     #         sub_experiment_dict = OrderedDict(
    #     #             {'experiment_name': sub_key, 'experiment_value': sub_value, 'train_set_name': 'unb',
    #     #              'val_set_name': 'unb'})
    #     #     elif sub_value[5] == 'mawi':
    #     #         sub_experiment_dict = OrderedDict(
    #     #             {'experiment_name': sub_key, 'experiment_value': sub_value, 'train_set_name': 'mawi',
    #     #              'val_set_name': 'mawi'})
    #
    #     else:
    #         print(f'key:{sub_key} or value:{sub_value} is not correct, please check again.')
    #         return -1
    #
    #
    # demo_dict =OrderedDict({
    #     'experiment_1': {'SYNT':'uSc1C2_20_14'},
    #     'experiment_2': {'SYNT':'uSc2C3_20_14'},
    #     'experiment_3': 'uSc3C2_20_14'
    # })
    #
    #
    #
    # mode = 'demo:experiment_1:SYNT'
    # print(f'\nmode:\'{mode}\'')
    # # mode = 'all'
    # if mode == 'all':
    #     for key, value_dict in experiments_dict.items():
    #         print(f'{key}: {value_dict}')
    #         for sub_key, sub_value in value_dict.items():
    #             print(f'-{key} => {sub_key}:{sub_value}')
    #             if key == 'experiment_1':
    #                 conduct_experiment_1(sub_experiment=(sub_key, sub_value), params_dict=params_dict)
    #             elif key == 'experiment_2':
    #                 conduct_experiment_2(sub_experiment=(sub_key, sub_value), params_dict=params_dict)
    #             elif key == 'experiment_3':
    #                 conduct_experiment_3(sub_experiment=(sub_key, sub_value), params_dict=params_dict)
    #             else:
    #                 print(f'key:{sub_key} or value:{sub_value} is not correct, please check again.')
    #                 pass
    # elif mode.startswith('demo'):
    #     key = demo_dict.split(':')
    #     experiment_key = key[1]
    #     dataset_key = key[2]
    #     experiment_value = demo_dict[experiment_key][dataset_key]
    #
    #
    #
    #     conduct_experiment(experiment_dict=sub_experiment_dict, params_dict=params_dict)
    # else:
    #     print(f'mode:{mode} is not correct.')
    #     return -1


if __name__ == '__main__':
    # umap_plot_data(case='uSc1C1_20_14')
    main(params_dict=params_dict)
