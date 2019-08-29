# Authors: xxx <xxx@xxx>
#
# License: xxx

"""Main function:
Evaluate autoencoder and classical machine learning algorithms on datasets in offline

"""
import os

from parameter import Configuration
from algorithms.classical.ml_algs import DTClass, IFClass, PCAClass
from algorithms.proposed.autoencoder import AutoencoderClass
from preprocess.normalization import save_roc_to_txt
from utils.balance import balance_dict, concat_path
from utils.dataset import Dataset
from utils.visualization import plot_roc_curve


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

    all_results_lst = []  # '', save all the sub_features results. each item is a dictionary.
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


def main():
    """
        datasets_dict={'train_set_dict':'', 'val_set_dict':'', 'test_set_dict':''},
            train_set_dict ={'train_set':{'X':, 'y':,}}
            val_set_dict ={'val_set':{'X':, 'y':,}}
            test_set_dict ={'test_set':{'X':, 'y':,},'test_set_2':{'X':, 'y':,},... }

    :return:
    """

    conf_inst = Configuration()  # return an instance or object of Configuration class.
    print(f'All parameters used: {conf_inst}')

    dataset_key = 'SYNT'
    case = conf_inst.experiments_dict['experiment_1'][dataset_key]
    datasets_inst = Dataset(case=case, input_dir=conf_inst.input_dir,
                            random_state=conf_inst.random_state)

    datasets_inst.selected_sub_features(features_lst=conf_inst.features_lst)
    if conf_inst.norm_flg:
        datasets_inst.normalize_data(norm_method=conf_inst.norm_method, train_set_key=dataset_key,
                                     not_normalized_features_lst=conf_inst.not_normalized_features_lst)

    datasets_dict = datasets_inst.datasets_dict
    train_set_dict = datasets_dict['train_set_dict'][f'{dataset_key}_train_set']  # train_set_dict= {'X':, 'y':}
    val_set_dict = datasets_dict['val_set_dict'][f'{dataset_key}_val_set']  # val_set_dict={'X':, 'y':}

    ################## algorithm AE #################################################
    AE_inst = AutoencoderClass(in_dim=datasets_inst.num_features)
    print(f'\n{AE_inst.alg_name} train and test')

    AE_inst.train(X=train_set_dict['X'], y=None, val_set_dict=val_set_dict)
    AE_inst.dump_model()
    ### find optimal thres on train set
    # AE_inst.find_optimal_thres(X=train_set_dict['X'], y=train_set_dict['y'])

    print(f'test {AE_inst.alg_name} on {dataset_key}_train_set with AE_thres: {AE_inst.thres_AE}')
    AE_inst.test(X=train_set_dict['X'], y=train_set_dict['y'])

    all_results_dict = {}   # save all results in dict
    for key, value_dict in datasets_dict['test_set_dict'].items():
        print(f'test {AE_inst.alg_name} on {key} with AE_thres: {AE_inst.thres_AE}')
        X_test, y_test = value_dict['X'], value_dict['y']
        test_set_name= key
        if conf_inst.balance_flg:
            print(f'--balance \'{test_set_name}\'')
            value_dict = {'X': X_test, 'y': y_test}
            value_dict = balance_dict(value_dict)
            X_test = value_dict['X']
            y_test = value_dict['y']

        AE_inst.test(X=X_test, y=y_test)
        if key not in all_results_dict.keys():
            all_results_dict[key]={}
        all_results_dict[key]['AE'] = {'y_true':y_test, 'y_pred': AE_inst.y_pred, 'y_proba': AE_inst.y_proba,
                                        'reconstr_errors': AE_inst.reconstr_errors_arr}

    ################## algorithm PCA #################################################
    PCA_inst = PCAClass()
    print(f'\n{PCA_inst.alg_name} train and test')

    PCA_inst.train(X=train_set_dict['X'])
    PCA_inst.dump_model()

    print(f'test {PCA_inst.alg_name} on {dataset_key}_train_set')
    PCA_inst.test(X=train_set_dict['X'], y=train_set_dict['y'])
    for key, value_dict in datasets_dict['test_set_dict'].items():
        print(f'test {PCA_inst.alg_name} on {key}')
        X_test = value_dict['X']
        y_test = value_dict['y']
        test_set_name = key
        if conf_inst.balance_flg:
            print(f'--balance \'{test_set_name}\'')
            value_dict = {'X': X_test, 'y': y_test}
            value_dict = balance_dict(value_dict)
            X_test = value_dict['X']
            y_test = value_dict['y']

        PCA_inst.test(X=X_test, y=y_test)
        if key not in all_results_dict.keys():
            all_results_dict[key]={}
        all_results_dict[key]['PCA'] = {'y_true':y_test,'y_pred': PCA_inst.y_pred, 'y_proba': PCA_inst.y_proba}

    ################## algorithm IF #################################################
    IF_inst = IFClass()
    print(f'\n{IF_inst.alg_name} train and test')

    IF_inst.train(X=train_set_dict['X'])
    IF_inst.dump_model()

    print(f'test {IF_inst.alg_name} on {dataset_key}_train_set')
    IF_inst.test(X=train_set_dict['X'], y=train_set_dict['y'])
    for key, value_dict in datasets_dict['test_set_dict'].items():
        print(f'test {IF_inst.alg_name} on {key}')
        X_test = value_dict['X']
        y_test = value_dict['y']
        test_set_name = key
        if conf_inst.balance_flg:
            print(f'--balance \'{test_set_name}\'')
            value_dict = {'X': X_test, 'y': y_test}
            value_dict = balance_dict(value_dict)
            X_test = value_dict['X']
            y_test = value_dict['y']

        IF_inst.test(X=X_test, y=y_test)
        if key not in all_results_dict.keys():
            all_results_dict[key]={}
        all_results_dict[key]['IF'] = {'y_true':y_test,'y_pred': IF_inst.y_pred, 'y_proba': IF_inst.y_proba}

    ################## algorithm DT #################################################
    DT_inst = DTClass()
    print(f'\n{DT_inst.alg_name} train and test')

    DT_inst.train(X=train_set_dict['X'], y=train_set_dict['y'])
    DT_inst.dump_model()

    print(f'test {DT_inst.alg_name} on {dataset_key}_trainset')
    DT_inst.test(X=train_set_dict['X'], y=train_set_dict['y'])

    for key, value_dict in datasets_dict['test_set_dict'].items():
        print(f'test {DT_inst.alg_name} on {key}')
        X_test = value_dict['X']
        y_test = value_dict['y']
        test_set_name = key
        if conf_inst.balance_flg:
            print(f'--balance \'{test_set_name}\'')
            value_dict = {'X': X_test, 'y': y_test}
            value_dict = balance_dict(value_dict)
            X_test = value_dict['X']
            y_test = value_dict['y']

        DT_inst.test(X=X_test, y=y_test)
        if key not in all_results_dict.keys():
            all_results_dict[key]={}
        all_results_dict[key]['DT'] = {'y_true':y_test,'y_pred': DT_inst.y_pred, 'y_proba': DT_inst.y_proba}


    ################## ROC of all result #################################################
    # save_roc_to_txt(out_file=roc_out_file, all_results_dict=all_results_dict)
    for key, value_dict in all_results_dict.items():
        test_set_name = key
        roc_out_file = concat_path(conf_inst.output_dir+'/figures', f'{test_set_name}_roc.pdf')
        plot_roc_curve(all_results_dict=value_dict, out_file=roc_out_file,  title_flg=conf_inst.title_flg, title='roc')

    #
    # experiment_name = 'experiment_1'  # conduct experiment 1
    # dataset_key = 'SYNT'  # train and validate on 'SYNT' data, test on all three datasets (SYNT, UNB, MAWI).
    # local_params_dict = OrderedDict(
    #     {'experiment_name': experiment_name,
    #      'experiment_value': experiments_dict[experiment_name][dataset_key],
    #      'train_set_name': dataset_key + '_train_set',
    #      'val_set_name': dataset_key + '_val_set'
    #      })
    #
    # print(f'\nload local_params_dict: {local_params_dict}')

    # print("\nstep 2: loading data")
    # datasets_dict = get_dataset(local_params_dict['experiment_value'], input_dir=params_dict['input_dir'],
    #                             random_state=params_dict['random_state'])
    #
    # if not params_dict['analyze_feature_flg']:  # without feature selection, so using all features to conduct experiment
    #     print("\nstep 3: train and test detection models on data")
    #     conduct_experiment_without_feature_analysis(experiment_dict=local_params_dict, datasets_dict=datasets_dict,
    #                                                 params_dict=params_dict)
    # else:  # do feature selection, so using different features to conduct experiment
    #     conduct_experiment_with_feature_selection(experiment_dict=local_params_dict, datasets_dict=datasets_dict,
    #                                               params_dict=params_dict)

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
    main()
