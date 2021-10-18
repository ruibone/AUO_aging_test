#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
from tqdm.auto import tqdm

from sklearn.linear_model import LogisticRegression, RidgeCV, Ridge
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor,    AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
import optuna

from library.Data_Preprocessing import Balance_Ratio
from library.Imbalance_Sampling import label_divide
from library.Aging_Score_Contour import score1
from library.AdaBoost import train_set, multiple_set, multiple_month, line_chart, cf_matrix, AUC, PR_curve,      multiple_curve, PR_matrix, best_threshold, all_optuna, optuna_history, AdaBoost_creator 
from library.XGBoost import XGBoost_creator
from library.LightGBM import LightGBM_creator
from library.CatBoost import CatBoost_creator
from library.Random_Forest import RandomForest_creator
from library.Extra_Trees import ExtraTrees_creator
'''
os.chdir('C:/Users/user/Desktop/Darui_R08621110')  
os.getcwd()
'''

# ## 

# ### optimize base learner

# In[ ]:


def month_param(num_set, date, month_list, model_list, iter_list, filename, mode, TPE_multi):
    
    sampler = 'multivariate-TPE' if TPE_multi else 'univariate-TPE'
    month_dict = {}
    for month in month_list:
        
        model_dict = {}
        for i, model in enumerate(model_list):
                
            with open(f'hyperparameter/{date}/{filename}_m{month}_{model}{mode}_{sampler}_{iter_list[i]}.data', 'rb') as f:
                model_param = pickle.load(f)
            model_dict[model] = model_param
        
        month_dict[f'm{month}'] = model_dict

    return month_dict


def all_param(num_set, date, model_list, iter_list, filename, mode, TPE_multi):
    
    sampler = 'multivariate-TPE' if TPE_multi else 'univariate-TPE'
    model_dict = {}
    for i, model in enumerate(model_list):

        with open(f'hyperparameter/{date}/{filename}_{model}{mode}_{sampler}_{iter_list[i]}.data', 'rb') as f:
                set_dict = pickle.load(f)
                
        model_dict[model] = set_dict
        
    return model_dict


def optimize_base(num_set, train_data, mode, TPE_multi, base_list, iter_list, filename):
    
    best_param = {}
    month_list = list(train_data.keys())
    
    for i in tqdm(month_list):
        
        best_param[f'{i}'] = {}
        if 'XGBoost' in base_list:
            print('\nStarting for XGBoost:\n')
            model_index = base_list.index('XGBoost')
            best_param[f'{i}'][f'XGBoost'], _ = all_optuna(num_set = num_set, 
                                                           all_data = train_data[f'{i}'], 
                                                           mode = mode, 
                                                           TPE_multi = TPE_multi, 
                                                           n_iter = iter_list[model_index],
                                                           filename = f'{filename}_{i}_XGBoost',
                                                           creator = XGBoost_creator)

        if 'LightGBM' in base_list:
            print('\nStarting for LightGBM:\n')
            model_index = base_list.index('LightGBM')
            best_param[f'{i}'][f'LightGBM'], _ = all_optuna(num_set = num_set, 
                                                            all_data = train_data[f'{i}'], 
                                                            mode = mode, 
                                                            TPE_multi = TPE_multi, 
                                                            n_iter = iter_list[model_index],
                                                            filename = f'{filename}_{i}_LightGBM',
                                                            creator = LightGBM_creator)
        
        if 'AdaBoost' in base_list:
            print('\nStarting for AdaBoost:\n')
            model_index = base_list.index('AdaBoost')
            best_param[f'{i}'][f'AdaBoost'], _ = all_optuna(num_set = num_set, 
                                                            all_data = train_data[f'{i}'], 
                                                            mode = mode, 
                                                            TPE_multi = TPE_multi, 
                                                            n_iter = iter_list[model_index],
                                                            filename = f'{filename}_{i}_AdaBoost',
                                                            creator = AdaBoost_creator)
            
        if 'CatBoost' in base_list:
            print('\nStarting for CatBoost:\n')
            model_index = base_list.index('CatBoost')
            best_param[f'{i}'][f'CatBoost'], _ = all_optuna(num_set = num_set, 
                                                            all_data = train_data[f'{i}'], 
                                                            mode = mode, 
                                                            TPE_multi = TPE_multi, 
                                                            n_iter = iter_list[model_index],
                                                            filename = f'{filename}_{i}_CatBoost',
                                                            creator = CatBoost_creator)
            
        if 'RandomForest' in base_list:
            print('\nStarting for RandomForest:\n')
            model_index = base_list.index('RandomForest')
            best_param[f'{i}'][f'RandomForest'], _ = all_optuna(num_set = num_set, 
                                                                all_data = train_data[f'{i}'], 
                                                                mode = mode, 
                                                                TPE_multi = TPE_multi, 
                                                                n_iter = iter_list[model_index],
                                                                filename = f'{filename}_{i}_RandomForest',
                                                                creator = RandomForest_creator)

        if 'ExtraTrees' in base_list:
            print('\nStarting for ExtraTrees:\n')
            model_index = base_list.index('ExtraTrees')
            best_param[f'{i}'][f'ExtraTrees'], _ = all_optuna(num_set = num_set, 
                                                              all_data = train_data[f'{i}'], 
                                                              mode = mode, 
                                                              TPE_multi = TPE_multi, 
                                                              n_iter = iter_list[model_index],
                                                              filename = f'{filename}_{i}_ExtraTrees',
                                                              creator = ExtraTrees_creator)
            
    return best_param


# ### transform data by base learner

# In[ ]:


def stratified_data(train_data, cv):
    
    good = train_data[train_data.GB == 0]
    bad = train_data[train_data.GB == 1]
    good_index = random.sample(good.index.to_list(), k = len(good))
    bad_index = random.sample(bad.index.to_list(), k = len(bad))
    
    train_x_dict = {}
    train_y_dict = {}
    valid_x_dict = {}
    valid_y_dict = {}
    for i in range(cv):
        
        if (i+1) == cv:
            good_valid_index = good_index[int(np.floor((i/cv)*len(good))): ]
            bad_valid_index = bad_index[int(np.floor((i/cv)*len(bad))): ]
        else:
            good_valid_index = good_index[int(np.floor((i/cv)*len(good))) : int(np.floor(((i+1)/cv)*len(good)))]
            bad_valid_index = bad_index[int(np.floor((i/cv)*len(bad))) : int(np.floor(((i+1)/cv)*len(bad)))]
        good_train_index = [x for x in good_index if x not in good_valid_index]
        bad_train_index = [x for x in bad_index if x not in bad_valid_index]
        
        good_train = good.loc[good_train_index]
        good_valid = good.loc[good_valid_index]
        bad_train = bad.loc[bad_train_index]
        bad_valid = bad.loc[bad_valid_index]
        train = pd.concat([good_train, bad_train], axis = 0)
        valid = pd.concat([good_valid, bad_valid], axis = 0)
        train_x_dict[i], train_y_dict[i], valid_x_dict[i], valid_y_dict[i] = label_divide(train, valid, 
                                                                                          train_only = False)

    return train_x_dict, train_y_dict, valid_x_dict, valid_y_dict


def transform_train(train_data, num_set, mode, base_param, cv):
    
    month_list = list(base_param.keys())
    model_list = list(base_param[month_list[0]].keys())
    set_dict = {}
    for x in range(num_set):
        set_dict[f'set{x}'] = pd.DataFrame()
        
    for month in tqdm(month_list):
        
        for i in tqdm(range(num_set)):
            
            train_x_dict, train_y_dict, valid_x_dict, valid_y_dict = stratified_data(train_data[month][f'set{i}'], cv = cv)
            all_cv = pd.DataFrame()
            for j in range(cv):
                
                model_predict = pd.DataFrame()
                if mode == 'C':

                    if 'XGBoost' in model_list:                     
                        clf = XGBClassifier(**base_param[month]['XGBoost'][f'set{i}'], n_jobs = -1)
                        clf.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = clf.predict_proba(valid_x_dict[j])
                        predict = pd.DataFrame({'X': predict_y[:, 0]})
                        model_predict = pd.concat([model_predict, predict], axis = 1)

                    if 'LightGBM' in model_list:                        
                        clf = LGBMClassifier(**base_param[month]['LightGBM'][f'set{i}'])
                        clf.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = clf.predict_proba(valid_x_dict[j])
                        predict = pd.DataFrame({'L': predict_y[:, 0]})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                    if 'CatBoost' in model_list:
                        clf = CatBoostClassifier(**base_param[month]['CatBoost'][f'set{i}'])
                        clf.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = clf.predict_proba(valid_x_dict[j])
                        predict = pd.DataFrame({'C': predict_y[:, 0]})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                    if 'AdaBoost' in model_list:
                        tree_param = {
                            'base_estimator': DecisionTreeClassifier(
                                max_depth = base_param[month]['AdaBoost'][f'set{i}']['max_depth']
                            )}
                        boost_param = dict(
                            (key, base_param[month]['AdaBoost'][f'set{i}'][key]) for key in ['learning_rate', 'n_estimators']
                        )
                        boost_param.update(tree_param)
                        clf = AdaBoostClassifier(**boost_param)
                        clf.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = clf.predict_proba(valid_x_dict[j])
                        predict = pd.DataFrame({'A': predict_y[:, 0]})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                    if 'RandomForest' in model_list:
                        clf = RandomForestClassifier(**base_param[month]['RandomForest'][f'set{i}'])
                        clf.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = clf.predict_proba(valid_x_dict[j])
                        predict = pd.DataFrame({'R': predict_y[:, 0]})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                    if 'ExtraTrees' in model_list:
                        clf = ExtraTreesClassifier(**base_param[month]['ExtraTrees'][f'set{i}'])
                        clf.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = clf.predict_proba(valid_x_dict[j])
                        predict = pd.DataFrame({'E': predict_y[:, 0]})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                elif mode == 'R':
                    
                    if 'XGBoost' in model_list:
                        reg = XGBRegressor(**base_param[month]['XGBoost'][f'set{i}'], n_jobs = -1)
                        reg.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = reg.predict(valid_x_dict[j])
                        predict = pd.DataFrame({'X': predict_y})
                        model_predict = pd.concat([model_predict, predict], axis = 1)

                    if 'LightGBM' in model_list:
                        reg = LGBMRegressor(**base_param[month]['LightGBM'][f'set{i}'])
                        reg.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = reg.predict(valid_x_dict[j])
                        predict = pd.DataFrame({'L': predict_y})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                    if 'CatBoost' in model_list:
                        reg = CatBoostRegressor(**base_param[month]['CatBoost'][f'set{i}'])
                        reg.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = reg.predict(valid_x_dict[j])
                        predict = pd.DataFrame({'C': predict_y})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                    if 'AdaBoost' in model_list:
                        tree_param = {
                            'base_estimator': DecisionTreeRegressor(
                                max_depth = base_param[month]['AdaBoost'][f'set{i}']['max_depth']
                            )}
                        boost_param = dict(
                            (key, base_param[month]['AdaBoost'][f'set{i}'][key]) for key in ['learning_rate', 'n_estimators']
                        )
                        boost_param.update(tree_param)
                        reg = AdaBoostRegressor(**boost_param)
                        reg.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = reg.predict(valid_x_dict[j])
                        predict = pd.DataFrame({'A': predict_y})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                    if 'RandomForest' in model_list:
                        reg = RandomForestRegressor(**base_param[month]['RandomForest'][f'set{i}'])
                        reg.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = reg.predict(valid_x_dict[j])
                        predict = pd.DataFrame({'R': predict_y})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                    
                    if 'ExtraTrees' in model_list:
                        reg = ExtraTreesRegressor(**base_param[month]['ExtraTrees'][f'set{i}'])
                        reg.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = reg.predict(valid_x_dict[j])
                        predict = pd.DataFrame({'E': predict_y})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                test_label = valid_y_dict[j].reset_index(drop = True)
                done_cv = pd.concat([model_predict, test_label], axis = 1)
                all_cv = pd.concat([all_cv, done_cv], axis = 0)
                
            set_dict[f'set{i}'] = pd.concat([set_dict[f'set{i}'], all_cv], axis = 0)
    
    return set_dict


def transform_test(train_data, test_data, num_set, mode, base_param):
    
    month_list = list(base_param.keys())
    model_list = list(base_param[month_list[0]].keys())
    test_dict = {}
    for i in tqdm(range(num_set)):
        
        month_test = pd.DataFrame()
        for month in tqdm(month_list):

            train_x, train_y, test_x, test_y = label_divide(train_data[f'set{i}'], test_data, train_only = False)
            model_predict = pd.DataFrame()
            if mode == 'C':

                if 'XGBoost' in model_list:
                    clf = XGBClassifier(**base_param[month]['XGBoost'][f'set{i}'], n_jobs = -1)
                    clf.fit(train_x, train_y)
                    predict_y = clf.predict_proba(test_x)
                    predict = pd.DataFrame({'X': predict_y[:, 0]})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'LightGBM' in model_list:
                    clf = LGBMClassifier(**base_param[month]['LightGBM'][f'set{i}'])
                    clf.fit(train_x, train_y)
                    predict_y = clf.predict_proba(test_x)
                    predict = pd.DataFrame({'L': predict_y[:, 0]})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'CatBoost' in model_list:
                    clf = CatBoostClassifier(**base_param[month]['CatBoost'][f'set{i}'])
                    clf.fit(train_x, train_y)
                    predict_y = clf.predict_proba(test_x)
                    predict = pd.DataFrame({'C': predict_y[:, 0]})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'AdaBoost' in model_list:
                    tree_param = {
                        'base_estimator': DecisionTreeClassifier(
                            max_depth = base_param[month]['AdaBoost'][f'set{i}']['max_depth']
                        )}
                    boost_param = dict(
                        (key, base_param[month]['AdaBoost'][f'set{i}'][key]) for key in ['learning_rate', 'n_estimators']
                    )
                    boost_param.update(tree_param)
                    clf = AdaBoostClassifier(**boost_param)
                    clf.fit(train_x, train_y)
                    predict_y = clf.predict_proba(test_x)
                    predict = pd.DataFrame({'A': predict_y[:, 0]})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'RandomForest' in model_list:
                    clf = RandomForestClassifier(**base_param[month]['RandomForest'][f'set{i}'])
                    clf.fit(train_x, train_y)
                    predict_y = clf.predict_proba(test_x)
                    predict = pd.DataFrame({'R': predict_y[:, 0]})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'ExtraTrees' in model_list:
                    clf = ExtraTreesClassifier(**base_param[month]['ExtraTrees'][f'set{i}'])
                    clf.fit(train_x, train_y)
                    predict_y = clf.predict_proba(test_x)
                    predict = pd.DataFrame({'E': predict_y[:, 0]})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

            elif mode == 'R':

                if 'XGBoost' in model_list:
                    reg = XGBRegressor(**base_param[month]['XGBoost'][f'set{i}'], n_jobs = -1)
                    reg.fit(train_x, train_y)
                    predict_y = reg.predict(test_x)
                    predict = pd.DataFrame({'X': predict_y})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'LightGBM' in model_list:
                    reg = LGBMRegressor(**base_param[month]['LightGBM'][f'set{i}'])
                    reg.fit(train_x, train_y)
                    predict_y = reg.predict(test_x)
                    predict = pd.DataFrame({'L': predict_y})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'CatBoost' in model_list:
                    reg = CatBoostRegressor(**base_param[month]['CatBoost'][f'set{i}'])
                    reg.fit(train_x, train_y)
                    predict_y = reg.predict(test_x)
                    predict = pd.DataFrame({'C': predict_y})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'AdaBoost' in model_list:
                    tree_param = {
                        'base_estimator': DecisionTreeRegressor(
                            max_depth = base_param[month]['AdaBoost'][f'set{i}']['max_depth']
                        )}
                    boost_param = dict(
                        (key, base_param[month]['AdaBoost'][f'set{i}'][key]) for key in ['learning_rate', 'n_estimators']
                    )
                    boost_param.update(tree_param)
                    reg = AdaBoostRegressor(**boost_param)
                    reg.fit(train_x, train_y)
                    predict_y = reg.predict(test_x)
                    predict = pd.DataFrame({'A': predict_y})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'RandomForest' in model_list:
                    reg = RandomForestRegressor(**base_param[month]['RandomForest'][f'set{i}'])
                    reg.fit(train_x, train_y)
                    predict_y = reg.predict(test_x)
                    predict = pd.DataFrame({'R': predict_y})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'ExtraTrees' in model_list:
                    reg = ExtraTreesRegressor(**base_param[month]['ExtraTrees'][f'set{i}'])
                    reg.fit(train_x, train_y)
                    predict_y = reg.predict(test_x)
                    predict = pd.DataFrame({'E': predict_y})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

            month_test = pd.concat([month_test, model_predict], axis = 1)
        month_done = pd.concat([month_test, test_y], axis = 1)
        test_dict[f'set{i}'] = month_done
        
    return test_dict


# ### meta learner

# In[ ]:


def LR(train_x, test_x, train_y, test_y, config):
    
    clf = LogisticRegression(**config)
    clf.fit(train_x, train_y)
    coef = clf.coef_
    predict_y = clf.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result, coef


def RidgeR(train_x, test_x, train_y, test_y, config):
    
    reg = Ridge(**config)
    reg.fit(train_x, train_y)
    coef = reg.coef_
    predict_y = reg.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result, coef


def runall_LR(num_set, trainset_x, testset_x, trainset_y, testset_y, config):
    
    table_set = pd.DataFrame()
    coef_set = pd.DataFrame()
    judge = list(config.keys())[0]

    for i in tqdm(range(num_set)):
        print('\n', f'Dataset {i}:')
        
        if isinstance(config[judge], dict) :
            best_config = config[f'set{i}']
        else :
            best_config = config
        model_name = trainset_x[f'set{i}'].columns.to_list()

        result, coef = LR(trainset_x[f'set{i}'], testset_x[f'set{i}'], trainset_y[f'set{i}'], testset_y[f'set{i}'], 
                          best_config)
        table = cf_matrix(result, trainset_y[f'set{i}'])
        coef_df = pd.DataFrame({f'set{i}': coef.flatten()})
        table_set = pd.concat([table_set, table]).rename(index = {0: f'dataset {i}'})
        coef_set = pd.concat([coef_set, coef_df], axis = 1)
    coef_set.index = model_name
    
    return table_set, coef_set


def runall_RidgeR(num_set, trainset_x, testset_x, trainset_y, testset_y, config, thres_target = 'Recall', 
                    threshold = False):
    
    table_set = pd.DataFrame()
    coef_set = pd.DataFrame()
    pr_dict = {}
    judge = list(config.keys())[0]

    for i in range(num_set):
        print('\n', f'Dataset {i}:')
        
        if isinstance(config[judge], dict) :
            best_config = config[f'set{i}']
        else :
            best_config = config
        model_name = trainset_x[f'set{i}'].columns.to_list()

        predict, coef = RidgeR(trainset_x[f'set{i}'], testset_x[f'set{i}'], trainset_y[f'set{i}'], testset_y[f'set{i}'], 
                           best_config)
        pr_matrix = PR_matrix(predict, trainset_y[f'set{i}'])
        pr_dict[f'set{i}'] = pr_matrix
        
        best_data, best_thres = best_threshold(pr_matrix, target = thres_target, threshold = threshold)
        table_set = pd.concat([table_set, best_data]).rename(index = {best_data.index.values[0]: f'dataset {i}'})
        coef_df = pd.DataFrame({f'set{i}': coef.flatten()})
        coef_set = pd.concat([coef_set, coef_df], axis = 1)
    coef_set.index = model_name
        
    return pr_dict, table_set, coef_set


# ### feature importance

# In[ ]:


def correlation_plot(target_data):
    
    correlation = target_data.iloc[:, :-1].corr()
    plot = sn.heatmap(correlation, annot = True, cmap = 'magma')
    plot.set_title('Correlation Coefficient of Base Learner Outputs')
    
    return correlation 
    
    
def vif(target_data):
    
    corr = target_data.corr()
    vif = round(corr / (1 - corr), 2)   
    return vif


def forest_importance(target_data, mode = 'R'):
    
    colname = target_data.columns.to_list()[:-1]
    X, Y = label_divide(target_data, None, 'GB', train_only = True)
    if mode == 'C':
        clf = RandomForestClassifier(max_depth = 5, n_estimators = 500)
    elif mode == 'R':
        clf = RandomForestRegressor(max_depth = 5, n_estimators = 500)
    clf.fit(X, Y)
    importance = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis = 0)
    importances = pd.DataFrame(dict(importance = importance, std = std), index = colname)
    importances = importances.sort_values('importance', ascending = True)
    
    plt.figure()
    plt.barh(importances.index, importances['importance'].values, color = 'darkgreen', 
             xerr = importances['std'].values, ecolor = 'limegreen')
    plt.title('Feature Importance by Random Forest')
    plt.xlabel('importance')
    plt.ylabel('model')
    
    return importances
    
    
def forest_permutation(target_data, mode = 'R'):
    
    colnames = target_data.columns.to_list()[:-1]
    X, Y = label_divide(target_data, None, 'GB', train_only = True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    if mode == 'C':
        clf = RandomForestClassifier(max_depth = 5, n_estimators = 300)
    elif mode == 'R':
        clf = RandomForestRegressor(max_depth = 5, n_estimators = 300)
    clf.fit(X_train, y_train)
    
    importance = permutation_importance(clf, X_test, y_test, n_repeats = 20, n_jobs = 2)
    importances = pd.DataFrame(dict(importance = importance.importances_mean, std = importance.importances_std), 
                               index = colnames)
    importances = importances.sort_values('importance', ascending = True)
    
    plt.figure()
    plt.barh(importances.index, importances['importance'].values, color = 'firebrick', 
             xerr = importances['std'].values, ecolor = 'coral')
    plt.title('Feature Importance by Permutation (Random Forest)')
    plt.xlabel('importance')
    plt.ylabel('model')
    
    return importances


#####!!!!! forest_shap can only run for regressor !!!!!#####
def forest_shap(target_data, mode = 'R'):
    
    colnames = target_data.columns.to_list()[:-1]
    X, Y = label_divide(target_data, None, 'GB', train_only = True)
    if mode == 'C':
        clf = RandomForestClassifier(max_depth = 5, n_estimators = 300)
    elif mode == 'R':
        clf = RandomForestRegressor(max_depth = 5, n_estimators = 300)
    clf.fit(X, Y)
    
    explainer = shap.Explainer(clf)
    shap_value = explainer(X)
    values = abs(shap_value.values).mean(axis = 0)
    values = values / sum(values)
    shap_df = pd.DataFrame(dict(value = values), index = colnames).sort_values('value', ascending = True)
    
    plt.figure()
    shap.plots.bar(shap_value)
    shap.plots.beeswarm(shap_value)
    
    return shap_df


def GLM_coefficient(target_data, mode = 'R'):
    
    colnames = target_data.columns.to_list()[:-1]
    X, Y = label_divide(target_data, None, 'GB', train_only = True)
    if mode == 'C':
        clf = LogisticRegression()
        clf.fit(X, Y)
        coefficient = abs(clf.coef_[0,:])
    elif mode == 'R':
        clf = Ridge()
        clf.fit(X, Y)
        coefficient = abs(clf.coef_)
    coef_df = pd.DataFrame(dict(GLM = coefficient), index = colnames).sort_values('GLM', ascending = True)
    
    return coef_df


#####!!!!! forest_shap can only run for regressor !!!!!#####
def rank_importance(target_data, mode = 'R'):
    
    correlation = correlation_plot(target_data)
    coefficient = GLM_coefficient(target_data, mode = mode).GLM
    forest = forest_importance(target_data, mode = mode).importance
    permutation = forest_permutation(target_data, mode = mode).importance
    shapvalue = forest_shap(target_data, mode = mode).value
    rank_df = pd.DataFrame()
    rank_df['GLM'] = coefficient.rank(ascending = False)
    rank_df['forest'] = forest.rank(ascending = False)
    rank_df['permutation'] = permutation.rank(ascending = False)
    rank_df['SHAP'] = shapvalue.rank(ascending = False)
    rank_df['total_rank'] = rank_df.apply(sum, axis = 1).rank()
    rank_df = rank_df.sort_values('total_rank', ascending = True)
    
    return rank_df


# ### optuna

# In[ ]:


def stackingCV_creator(train_data, mode, num_valid = 3) :
    
    def objective(trial) :
        # hyperparameters randomize setting
        if mode == 'C' :
            meta_learner = 'Logistic Regression'
            
            if meta_learner == 'Logistic Regression' :      
                param = {
                    'solver': 'lbfgs',
                    'C': trial.suggest_categorical('C', [100, 10 ,1 ,0.1, 0.01]),
                    'penalty': trial.suggest_categorical('penalty', ['none', 'l2']),
                    'n_jobs': -1
                }

            elif meta_learner == 'Extra Trees' :
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500, step = 100),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 32, step = 5),
                    'max_depth': trial.suggest_int('max_depth', 3, 21, step = 3),
                    'n_jobs': -1
                }     

        elif mode == 'R' :
            meta_learner = 'RidgeCV'
            
            if meta_learner == 'RidgeCV' :
                param = {
                    'alpha': trial.suggest_float('alpha', 0, 1, step = 0.1)
                }
            
            elif meta_learner == 'Extra Trees' :
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500, step = 100),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 32, step = 5),
                    'max_depth': trial.suggest_int('max_depth', 3, 21, step = 3),
                    'n_jobs': -1
                }
        
        # objective function
        result_list = []
        for i in range(num_valid):

            train_x, train_y = label_divide(train_data, None, 'GB', train_only = True)
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.25)

            if mode == 'C':
                result, _ = LR(train_x, valid_x, train_y, valid_y, param)
                table = cf_matrix(result, valid_y)
                recall = table['Recall']
                aging = table['Aging Rate']
                result_list.append(recall - 0.1*aging)

            elif mode == 'R':
                result, _ = RidgeR(train_x, valid_x, train_y, valid_y, param)
                pr_matrix = PR_matrix(result, valid_y)
                auc = AUC(pr_matrix['Recall'], pr_matrix['Aging Rate'])
                result_list.append((-1)*auc)

        return np.mean(result_list)
    
    return objective

'''
# ## 

# ### loading training & testing data

# In[ ]:


### training data ### 
training_month = [2, 3, 4]

data_dict, trainset_x, trainset_y = multiple_month(training_month, num_set = 10, filename = 'dataset')

print('\nCombined training data:\n')
run_train = multiple_set(num_set = 10)
run_train_x, run_train_y = train_set(run_train, num_set = 10)

### testing data ###
run_test = pd.read_csv('test_runhist.csv').iloc[:, 2:]
run_test_x, run_test_y = label_divide(run_test, None, 'GB', train_only = True)
print('\n', 'Dimension of testing data:', run_test.shape)


# ## base learner

# ### optimize the base learners by one-month data

# #### for training data transformation

# In[ ]:


##### by optuna ##### 
base_param_monthC = optimize_base(num_set = 10, 
                                  train_data = data_dict, 
                                  mode = 'C', 
                                  TPE_multi = True, 
                                  base_list = ['XGBoost', 'LightGBM'],
                                  iter_list = [200, 200],
                                  filename = 'runhist_array_4criteria_m2m5')
 
base_param_monthR = optimize_base(num_set = 10, 
                                  train_data = data_dict, 
                                  mode = 'R', 
                                  TPE_multi = True, 
                                  base_list = ['XGBoost', 'LightGBM'],
                                  iter_list = [200, 200],
                                  filename = 'runhist_array_4criteria_m2m5')


# In[ ]:


##### 'OR' by loading from stackingCV scheme 2 #####
base_param_monthC = month_param(num_set = 10, 
                                date = '20211019', 
                                month_list = [2, 3, 4], 
                                model_list = ['XGBoost', 'LightGBM'], 
                                iter_list = [200, 200], 
                                filename = 'runhist_array_4criteria_m2m5', 
                                mode = 'C', 
                                TPE_multi = True)
base_param_monthR = month_param(num_set = 10, 
                                date = '20211019', 
                                month_list = [2, 3, 4], 
                                model_list = ['XGBoost', 'LightGBM'], 
                                iter_list = [200, 200], 
                                filename = 'runhist_array_4criteria_m2m5', 
                                mode = 'R', 
                                TPE_multi = True)


# #### for testing data transformation

# In[ ]:


##### by optuna ##### 
base_param_allC = optimize_base(num_set = 10, 
                                train_data = {'all': run_train}, 
                                mode = 'C', 
                                TPE_multi = True, 
                                base_list = ['XGBoost', 'LightGBM'], 
                                iter_list = [200, 200],
                                filename = 'runhist_array_4criteria_m2m5')

base_param_allR = optimize_base(num_set = 10, 
                                train_data = {'all': run_train}, 
                                mode = 'R', 
                                TPE_multi = True, 
                                base_list = ['XGBoost', 'LightGBM'], 
                                iter_list = [200, 200],
                                filename = 'runhist_array_4criteria_m2m5')


# In[ ]:


##### 'OR' by loading from stackingCV scheme 1 #####
base_param_allC = all_param(num_set = 10, 
                           date = '20211019', 
                           model_list = ['XGBoost', 'LightGBM'], 
                           iter_list = [200, 200], 
                           filename = 'runhist_array_m2m5_4selection', 
                           mode = 'C', 
                           TPE_multi = True)
base_param_allR = all_param(num_set = 10, 
                           date = '20211019', 
                           model_list = ['XGBoost', 'LightGBM'], 
                           iter_list = [200, 200], 
                           filename = 'runhist_array_m2m5_4selection', 
                           mode = 'R', 
                           TPE_multi = True)


# ### data transform for scheme 3

# In[ ]:


train_firstC = transform_train(data_dict, 
                               num_set = 10, 
                               mode = 'C', 
                               base_param = base_param_monthC, 
                               cv = 5)
test_firstC = transform_test(run_train, 
                             run_test, 
                             num_set = 10, 
                             mode = 'C', 
                             base_param = dict(all = base_param_allC))
train_firstC_x, train_firstC_y = train_set(train_firstC, num_set = 10)
test_firstC_x, test_firstC_y = train_set(test_firstC, num_set = 10) 

train_firstR = transform_train(data_dict, 
                               num_set = 10, 
                               mode = 'R', 
                               base_param = base_param_monthR, 
                               cv = 5)
test_firstR = transform_test(run_train, 
                             run_test, 
                             num_set = 10,
                             mode = 'R',
                             base_param = dict(all = base_param_allR))
train_firstR_x, train_firstR_y = train_set(train_firstR, num_set = 10)
test_firstR_x, test_firstR_y = train_set(test_firstR, num_set = 10) 


# ## meta learner

# ### searching for best hyperparameters

# In[ ]:


best_paramC, _ = all_optuna(num_set = 10, 
                            all_data = train_firstC, 
                            mode = 'C', 
                            TPE_multi = True, 
                            n_iter = 10,
                            filename = f'runhist_array_4criteria_m2m5_StackingCV3',
                            creator = stackingCV_creator
)

best_paramR, _ = all_optuna(num_set = 10, 
                            all_data = train_firstR, 
                            mode = 'R', 
                            TPE_multi = True, 
                            n_iter = 10,
                            filename = f'runhist_array_4criteria_m2m5_StackingCV3',
                            creator = stackingCV_creator
)


# ### feature selection by feature importance

# In[ ]:


rank_importance(train_firstR['set7'], mode = 'R')


# ### classifier

# In[ ]:


table_setC, coefC = runall_LR(10, train_firstC_x, test_firstC_x, train_firstC_y, test_firstC_y, best_paramC)
line_chart(table_setC, title = 'StackingCV Classifier (scheme 3)')


# In[ ]:


print(coefC)
table_setC


# ### regressor

# In[ ]:


pr_dict, table_setR, coefR = runall_RidgeR(10, train_firstR_x, test_firstR_x, train_firstR_y, test_firstR_y, 
                                           best_paramR, thres_target = 'Recall', threshold = 0.7)
line_chart(table_setR, title = 'StackingCV Regressor (scheme 3)')


# In[ ]:


multiple_curve(4, 3, pr_dict, table_setR, target = 'Aging Rate')
multiple_curve(4, 3, pr_dict, table_setR, target = 'Precision')
print(coefR)
table_setR


# ### export

# In[ ]:


savedate = '20211019'
TPE_multi = True

table_setC['sampler'] = 'multivariate-TPE' if TPE_multi else 'univariate-TPE'
table_setR['sampler'] = 'multivariate-TPE' if TPE_multi else 'univariate-TPE'
table_setC['model'] = 'StackingCV 3'
table_setR['model'] = 'StackingCV 3'
with pd.ExcelWriter(f'{savedate}_Classifier.xlsx', mode = 'a') as writer:
    table_setC.to_excel(writer, sheet_name = 'StackingCV_3')
with pd.ExcelWriter(f'{savedate}_Regressor.xlsx', mode = 'a') as writer:
    table_setR.to_excel(writer, sheet_name = 'StackingCV_3')

'''