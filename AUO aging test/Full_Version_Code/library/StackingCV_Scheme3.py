#!/usr/bin/env python
# coding: utf-8

# In[142]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import random
import pickle
from tqdm.auto import tqdm

from sklearn.linear_model import LogisticRegression, RidgeCV, Ridge
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor,    AdaBoostClassifier, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import shap
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
import torch
from torch import nn
from torch.utils.data import DataLoader
import optuna

from library.Data_Preprocessing import Balance_Ratio, train_col
from library.Imbalance_Sampling import label_divide
from library.Aging_Score_Contour import score1
from library.AdaBoost import train_set, multiple_set, multiple_month, line_chart, cf_matrix, AUC, PR_curve,      multiple_curve, PR_matrix, best_threshold, all_optuna, optuna_history, AdaBoost_creator 
from library.XGBoost import XGBoost_creator
from library.LightGBM import LightGBM_creator
from library.CatBoost import CatBoost_creator
from library.RandomForest import RandomForest_creator
from library.ExtraTrees import ExtraTrees_creator
from library.NeuralNetwork import RunhistSet, NeuralNetworkC, trainingC, NeuralNetwork_creator
'''
os.chdir('C:/Users/user/Desktop/Darui_R08621110')  
os.getcwd()
'''

# ## 

# ### Optimize Base Learners

# In[114]:


# load hyperparameters of base learners finished by scheme 2 (for training data transformation)
def month_param(date, month_list, model_list, iter_dict, filename, mode, TPE_multi):
    
    sampler = 'multivariate-TPE' if TPE_multi else 'univariate-TPE'
    month_dict = {}
    for month in month_list:
        
        model_dict = {}
        for model in model_list:
                
            with open(f'hyperparameter/{date}/{filename}_m{month}_{model}{mode}_{sampler}_{iter_dict[model]}.data', 'rb') as f:
                model_param = pickle.load(f)
            model_dict[model] = model_param
        
        month_dict[f'm{month}'] = model_dict

    return month_dict


# load hyperparameters of base learners finished by scheme 1 (for testing data transformation)
def all_param(date, model_list, iter_dict, filename, mode, TPE_multi):
    
    sampler = 'multivariate-TPE' if TPE_multi else 'univariate-TPE'
    model_dict = {}
    for model in model_list:

        with open(f'hyperparameter/{date}/{filename}_{model}{mode}_{sampler}_{iter_dict[model]}.data', 'rb') as f:
                set_dict = pickle.load(f)           
        model_dict[model] = set_dict
    
    done_dict = dict(all = model_dict)
        
    return done_dict


# search for the best hyperparameters of base learners
def optimize_base(train_data, mode, TPE_multi, base_list, iter_dict, filename):
    
    best_param = {}
    month_list = list(train_data.keys())
    available_model = ['AdaBoost', 'XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'ExtraTrees', 'NeuralNetwork']
    
    for i in tqdm(month_list):
        
        best_param[f'{i}'] = {}
        for model in base_list:
            
            if model not in available_model:
                raise('Invalid Model !')
            elif model == available_model[0]:
                creator = AdaBoost_creator
            elif model == available_model[1]:
                creator = XGBoost_creator
            elif model == available_model[2]:
                creator = LightGBM_creator
            elif model == available_model[3]:
                creator = CatBoost_creator
            elif model == available_model[4]:
                creator = RandomForest_creator
            elif model == available_model[5]:
                creator = ExtraTrees_creator
            elif model == available_model[6]:
                creator = NeuralNetwork_creator
            print(f'\nStarting for {model}:\n')
        
            best_param[f'{i}'][model], _ = all_optuna(all_data = train_data[f'{i}'], 
                                                      mode = mode, 
                                                      TPE_multi = TPE_multi, 
                                                      n_iter = iter_dict[model],
                                                      filename = f'{filename}_{i}_{model}',
                                                      creator = creator)
            
    return best_param


# ### Transform Data by Base Learners

# In[117]:


# concept of strtified cross-validation
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
        train_x_dict[i], train_y_dict[i], valid_x_dict[i], valid_y_dict[i] = label_divide(train, valid, train_only = False)

    return train_x_dict, train_y_dict, valid_x_dict, valid_y_dict


# input training data to the base learners and output the outcome
def transform_train(train_data, mode, base_param, cv):
    
    month_list = list(base_param.keys())
    model_list = list(base_param[month_list[0]].keys())
    set_list = list(base_param[month_list[0]][model_list[0]].keys())
    set_dict = {}
    for x in set_list:
        set_dict[x] = pd.DataFrame()
        
    for month in tqdm(month_list):
        for i in tqdm(set_list):
            
            train_x_dict, train_y_dict, valid_x_dict, valid_y_dict = stratified_data(train_data[month][i], cv = cv)
            all_cv = pd.DataFrame()
            for j in range(cv):
                
                model_predict = pd.DataFrame()
                if mode == 'C':
                    
                    if 'NeuralNetwork' in model_list:
                        temp_train = RunhistSet(train_x_dict[j], train_y_dict[j])
                        temp_valid = RunhistSet(valid_x_dict[j], valid_y_dict[j])
                        train_loader = DataLoader(temp_train, 
                                                  batch_size = base_param[month]['NeuralNetwork'][i]['batch_size'], 
                                                  shuffle = True)
                        valid_loader = DataLoader(temp_valid, batch_size = len(valid_x_dict[j]), shuffle = False)
                        nn_model = NeuralNetworkC(dim = train_x_dict[j].shape[1])
                        optimizer = torch.optim.Adam(nn_model.parameters(), 
                                                     lr = base_param[month]['NeuralNetwork'][i]['learning_rate'], 
                                                     weight_decay = base_param[month]['NeuralNetwork'][i]['weight_decay'])
                        criterion = nn.CrossEntropyLoss(
                            weight = torch.tensor([1-base_param[month]['NeuralNetwork'][i]['bad_weight'], 
                                                   base_param[month]['NeuralNetwork'][i]['bad_weight']])).to('cpu')
                        network, _, _ = trainingC(nn_model, train_loader, train_loader, optimizer, criterion, epoch = 100, 
                                                  early_stop = 10)
                        for x, y in valid_loader:
                            output = network(x)
                            _, predict_y = torch.max(output.data, 1)
                        predict = pd.DataFrame({'N': predict_y.numpy()})
                        model_predict = pd.concat([model_predict, predict], axis = 1)

                    if 'XGBoost' in model_list:                     
                        clf = XGBClassifier(**base_param[month]['XGBoost'][i], use_label_encoder = False, n_jobs = -1)
                        clf.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = clf.predict_proba(valid_x_dict[j])
                        predict = pd.DataFrame({'X': predict_y[:, 0]})
                        model_predict = pd.concat([model_predict, predict], axis = 1)

                    if 'LightGBM' in model_list:                        
                        clf = LGBMClassifier(**base_param[month]['LightGBM'][i])
                        clf.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = clf.predict_proba(valid_x_dict[j])
                        predict = pd.DataFrame({'L': predict_y[:, 0]})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                    if 'CatBoost' in model_list:
                        clf = CatBoostClassifier(**base_param[month]['CatBoost'][i])
                        clf.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = clf.predict_proba(valid_x_dict[j])
                        predict = pd.DataFrame({'C': predict_y[:, 0]})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                    if 'AdaBoost' in model_list:
                        tree_param = {
                            'base_estimator': DecisionTreeClassifier(
                                max_depth = base_param[month]['AdaBoost'][i]['max_depth']
                            )}
                        boost_param = dict(
                            (key, base_param[month]['AdaBoost'][i][key]) for key in ['learning_rate', 'n_estimators']
                        )
                        boost_param.update(tree_param)
                        clf = AdaBoostClassifier(**boost_param)
                        clf.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = clf.predict_proba(valid_x_dict[j])
                        predict = pd.DataFrame({'A': predict_y[:, 0]})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                    if 'RandomForest' in model_list:
                        clf = RandomForestClassifier(**base_param[month]['RandomForest'][i])
                        clf.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = clf.predict_proba(valid_x_dict[j])
                        predict = pd.DataFrame({'R': predict_y[:, 0]})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                    if 'ExtraTrees' in model_list:
                        clf = ExtraTreesClassifier(**base_param[month]['ExtraTrees'][i])
                        clf.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = clf.predict_proba(valid_x_dict[j])
                        predict = pd.DataFrame({'E': predict_y[:, 0]})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                elif mode == 'R':
                    
                    if 'XGBoost' in model_list:
                        reg = XGBRegressor(**base_param[month]['XGBoost'][i], n_jobs = -1)
                        reg.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = reg.predict(valid_x_dict[j])
                        predict = pd.DataFrame({'X': predict_y})
                        model_predict = pd.concat([model_predict, predict], axis = 1)

                    if 'LightGBM' in model_list:
                        reg = LGBMRegressor(**base_param[month]['LightGBM'][i])
                        reg.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = reg.predict(valid_x_dict[j])
                        predict = pd.DataFrame({'L': predict_y})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                    if 'CatBoost' in model_list:
                        reg = CatBoostRegressor(**base_param[month]['CatBoost'][i])
                        reg.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = reg.predict(valid_x_dict[j])
                        predict = pd.DataFrame({'C': predict_y})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                    if 'AdaBoost' in model_list:
                        tree_param = {
                            'base_estimator': DecisionTreeRegressor(
                                max_depth = base_param[month]['AdaBoost'][i]['max_depth']
                            )}
                        boost_param = dict(
                            (key, base_param[month]['AdaBoost'][i][key]) for key in ['learning_rate', 'n_estimators']
                        )
                        boost_param.update(tree_param)
                        reg = AdaBoostRegressor(**boost_param)
                        reg.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = reg.predict(valid_x_dict[j])
                        predict = pd.DataFrame({'A': predict_y})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                    if 'RandomForest' in model_list:
                        reg = RandomForestRegressor(**base_param[month]['RandomForest'][i])
                        reg.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = reg.predict(valid_x_dict[j])
                        predict = pd.DataFrame({'R': predict_y})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                    
                    if 'ExtraTrees' in model_list:
                        reg = ExtraTreesRegressor(**base_param[month]['ExtraTrees'][i])
                        reg.fit(train_x_dict[j], train_y_dict[j])
                        predict_y = reg.predict(valid_x_dict[j])
                        predict = pd.DataFrame({'E': predict_y})
                        model_predict = pd.concat([model_predict, predict], axis = 1)
                        
                test_label = valid_y_dict[j].reset_index(drop = True)
                done_cv = pd.concat([model_predict, test_label], axis = 1)
                all_cv = pd.concat([all_cv, done_cv], axis = 0)
                
            set_dict[i] = pd.concat([set_dict[i], all_cv], axis = 0)
    
    return set_dict


# input testing data to the base learners and output the outcome
def transform_test(train_data, test_data, mode, base_param):
    
    month_list = list(base_param.keys())
    model_list = list(base_param[month_list[0]].keys())
    set_list = list(base_param[month_list[0]][model_list[0]].keys())
    test_dict = {}
    for i in tqdm(set_list):
        
        month_test = pd.DataFrame()
        for month in month_list:

            train_x, train_y, test_x, test_y = label_divide(train_data[i], test_data, train_only = False)
            model_predict = pd.DataFrame()
            if mode == 'C':

                if 'NeuralNetwork' in model_list:
                    temp_train = RunhistSet(train_x, train_y)
                    temp_test = RunhistSet(test_x, test_y)
                    train_loader = DataLoader(temp_train, 
                                              batch_size = base_param[month]['NeuralNetwork'][i]['batch_size'], 
                                              shuffle = True)
                    test_loader = DataLoader(temp_test, batch_size = len(test_x), shuffle = False)
                    nn_model = NeuralNetworkC(dim = train_x.shape[1])
                    optimizer = torch.optim.Adam(nn_model.parameters(), 
                                                 lr = base_param[month]['NeuralNetwork'][i]['learning_rate'], 
                                                 weight_decay = base_param[month]['NeuralNetwork'][i]['weight_decay'])
                    criterion = nn.CrossEntropyLoss(
                        weight = torch.tensor([1-base_param[month]['NeuralNetwork'][i]['bad_weight'], 
                                               base_param[month]['NeuralNetwork'][i]['bad_weight']])).to('cpu')
                    network, _, _ = trainingC(nn_model, train_loader, train_loader, optimizer, criterion, epoch = 100, 
                                              early_stop = 10)
                    for X, Y in test_loader:
                        X, Y = X.float(), Y.long()
                        output = network(X)
                        _, predict_y = torch.max(output.data, 1)
                    predict = pd.DataFrame({'N': predict_y.numpy()})
                    model_predict = pd.concat([model_predict, predict], axis = 1)
                
                if 'XGBoost' in model_list:
                    clf = XGBClassifier(**base_param[month]['XGBoost'][i], use_label_encoder = False, n_jobs = -1)
                    clf.fit(train_x, train_y)
                    predict_y = clf.predict_proba(test_x)
                    predict = pd.DataFrame({'X': predict_y[:, 0]})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'LightGBM' in model_list:
                    clf = LGBMClassifier(**base_param[month]['LightGBM'][i])
                    clf.fit(train_x, train_y)
                    predict_y = clf.predict_proba(test_x)
                    predict = pd.DataFrame({'L': predict_y[:, 0]})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'CatBoost' in model_list:
                    clf = CatBoostClassifier(**base_param[month]['CatBoost'][i])
                    clf.fit(train_x, train_y)
                    predict_y = clf.predict_proba(test_x)
                    predict = pd.DataFrame({'C': predict_y[:, 0]})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'AdaBoost' in model_list:
                    tree_param = {
                        'base_estimator': DecisionTreeClassifier(
                            max_depth = base_param[month]['AdaBoost'][i]['max_depth']
                        )}
                    boost_param = dict(
                        (key, base_param[month]['AdaBoost'][i][key]) for key in ['learning_rate', 'n_estimators']
                    )
                    boost_param.update(tree_param)
                    clf = AdaBoostClassifier(**boost_param)
                    clf.fit(train_x, train_y)
                    predict_y = clf.predict_proba(test_x)
                    predict = pd.DataFrame({'A': predict_y[:, 0]})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'RandomForest' in model_list:
                    clf = RandomForestClassifier(**base_param[month]['RandomForest'][i])
                    clf.fit(train_x, train_y)
                    predict_y = clf.predict_proba(test_x)
                    predict = pd.DataFrame({'R': predict_y[:, 0]})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'ExtraTrees' in model_list:
                    clf = ExtraTreesClassifier(**base_param[month]['ExtraTrees'][i])
                    clf.fit(train_x, train_y)
                    predict_y = clf.predict_proba(test_x)
                    predict = pd.DataFrame({'E': predict_y[:, 0]})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

            elif mode == 'R':

                if 'XGBoost' in model_list:
                    reg = XGBRegressor(**base_param[month]['XGBoost'][i], n_jobs = -1)
                    reg.fit(train_x, train_y)
                    predict_y = reg.predict(test_x)
                    predict = pd.DataFrame({'X': predict_y})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'LightGBM' in model_list:
                    reg = LGBMRegressor(**base_param[month]['LightGBM'][i])
                    reg.fit(train_x, train_y)
                    predict_y = reg.predict(test_x)
                    predict = pd.DataFrame({'L': predict_y})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'CatBoost' in model_list:
                    reg = CatBoostRegressor(**base_param[month]['CatBoost'][i])
                    reg.fit(train_x, train_y)
                    predict_y = reg.predict(test_x)
                    predict = pd.DataFrame({'C': predict_y})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'AdaBoost' in model_list:
                    tree_param = {
                        'base_estimator': DecisionTreeRegressor(
                            max_depth = base_param[month]['AdaBoost'][i]['max_depth']
                        )}
                    boost_param = dict(
                        (key, base_param[month]['AdaBoost'][i][key]) for key in ['learning_rate', 'n_estimators']
                    )
                    boost_param.update(tree_param)
                    reg = AdaBoostRegressor(**boost_param)
                    reg.fit(train_x, train_y)
                    predict_y = reg.predict(test_x)
                    predict = pd.DataFrame({'A': predict_y})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'RandomForest' in model_list:
                    reg = RandomForestRegressor(**base_param[month]['RandomForest'][i])
                    reg.fit(train_x, train_y)
                    predict_y = reg.predict(test_x)
                    predict = pd.DataFrame({'R': predict_y})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

                if 'ExtraTrees' in model_list:
                    reg = ExtraTreesRegressor(**base_param[month]['ExtraTrees'][i])
                    reg.fit(train_x, train_y)
                    predict_y = reg.predict(test_x)
                    predict = pd.DataFrame({'E': predict_y})
                    model_predict = pd.concat([model_predict, predict], axis = 1)

            month_test = pd.concat([month_test, model_predict], axis = 1)
        month_done = pd.concat([month_test, test_y], axis = 1)
        test_dict[i] = month_done
        
    return test_dict


# ### Meta Learner

# In[62]:


# input training data transformed by base classifiers and output the final prediction 
def LR(train_x, test_x, train_y, test_y, config):
    
    subconfig = config.copy()
    del subconfig['meta_learner']
    if config['meta_learner'] == 'LogisticRegression':
        clf = LogisticRegression(**subconfig)
        clf.fit(train_x, train_y)
        coef = clf.coef_
    elif config['meta_learner'] == 'ExtraTrees':
        clf = ExtraTreesClassifier(**subconfig)
        clf.fit(train_x, train_y)
        coef = clf.feature_importances_
    predict_y = clf.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result, coef


# input training data transformed by base regressors and output the final prediction (optional)
def RidgeR(train_x, test_x, train_y, test_y, config):
    
    reg = Ridge(**config)
    reg.fit(train_x, train_y)
    coef = reg.coef_
    predict_y = reg.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result, coef


# run all resampling datasets to the meta classifer
def runall_LR(trainset_x, testset_x, trainset_y, testset_y, config):
    
    table_set = pd.DataFrame()
    coef_set = pd.DataFrame()
    set_index = list(config.keys())
    judge = set_index[0]

    for i, j in tqdm(enumerate(set_index)):
        print('\n', f'Data{j}:')
        if isinstance(config[judge], dict) :
            best_config = config[j]
        else :
            best_config = config
        model_name = trainset_x[j].columns.to_list()

        result, coef = LR(trainset_x[j], testset_x[j], trainset_y[j], testset_y[j], 
                          best_config)
        table = cf_matrix(result, trainset_y[j])
        coef_df = pd.DataFrame({str(j): coef.flatten()})
        table_set = pd.concat([table_set, table]).rename(index = {0: f'data{j}'})
        coef_set = pd.concat([coef_set, coef_df], axis = 1)
    coef_set.index = model_name
    
    return table_set, coef_set


# run all resampling datasets to the meta regressor (optional)
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


# ### Feature Importance

# In[198]:


def correlation_plot(target_data):
    
    correlation = target_data.iloc[:, :-1].corr()
    plot = sn.heatmap(correlation, annot = True, cmap = 'magma')
    plot.set_title('Correlation Coefficient of Base Learner Outputs')
    
    return correlation 
    
    
def vif(target_data):
    
    corr = target_data.corr()
    vif = round(corr / (1 - corr), 2)   
    return vif


def forest_importance(target_data, mode = 'C'):
    
    colname = target_data.columns.to_list()[:-1]
    X, Y = label_divide(target_data, None, 'GB', train_only = True)
    if mode == 'C':
        clf = RandomForestClassifier(n_estimators = 500)
    elif mode == 'R':
        clf = RandomForestRegressor(n_estimators = 500)
    clf.fit(X, Y)
    importance = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis = 0)
    importances = pd.DataFrame(dict(importance = importance, std = std), index = colname)
    importances = importances.sort_values('importance', ascending = True)
    
    plt.figure()
    plt.barh(importances.index, importances['importance'].values, color = 'darkgreen', 
             xerr = importances['std'].values, ecolor = 'limegreen')
    plt.title('Feature Importance by RandomForest Node Split')
    plt.xlabel('importance')
    plt.ylabel('model')
    
    return importances
    
    
def xgb_permutation(target_data, mode = 'C'):
    
    colnames = target_data.columns.to_list()[:-1]
    X, Y = label_divide(target_data, None, 'GB', train_only = True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    if mode == 'C':
        clf = XGBClassifier(n_estimators = 500)
    elif mode == 'R':
        clf = XGBRegressor(n_estimators = 500)
    clf.fit(X_train, y_train)
    
    importance = permutation_importance(clf, X_test, y_test, n_repeats = 10, n_jobs = 2)
    importances = pd.DataFrame(dict(importance = importance.importances_mean, std = importance.importances_std), 
                               index = colnames)
    importances = importances.sort_values('importance', ascending = True)
    
    plt.figure()
    plt.barh(importances.index, importances['importance'].values, color = 'firebrick', 
             xerr = importances['std'].values, ecolor = 'coral')
    plt.title('Feature Importance by LightGBM Permutation')
    plt.xlabel('importance')
    plt.ylabel('model')
    
    return importances


#####!!!!! forest_shap can only run for regressor !!!!!#####
def lgbm_shap(target_data, mode = 'C'):
    
    colnames = target_data.columns.to_list()[:-1]
    X, Y = label_divide(target_data, None, 'GB', train_only = True)
    if mode == 'C':
        d_train = lightgbm.Dataset(X, label = Y)
        clf = lightgbm.train(dict(n_estimators = 300, objective = 'binary'), d_train, 500, verbose_eval = -1)
    elif mode == 'R':
        clf = LGBMRegressor(n_estimators = 500)
        clf.fit(X, Y)
    
    explainer = shap.Explainer(clf)
    shap_values = explainer.shap_values(X)
    values = abs(shap_values[1]).mean(axis = 0)
    shap_df = pd.DataFrame(dict(value = values), index = colnames).sort_values('value', ascending = True)
    
    plt.figure()
    shap.summary_plot(shap_values, X)
    
    return shap_df


def GLM_coefficient(target_data, mode = 'C'):
    
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
def rank_importance(target_data, mode = 'C'):
    
    print(vif(target_data))
    correlation = correlation_plot(target_data)
    coefficient = GLM_coefficient(target_data, mode = mode).GLM
    forest = forest_importance(target_data, mode = mode).importance
    permutation = xgb_permutation(target_data, mode = mode).importance
    shapvalue = lgbm_shap(target_data).value
    rank_df = pd.DataFrame()
    rank_df['GLM'] = coefficient.rank(ascending = False)
    rank_df['forest'] = forest.rank(ascending = False)
    rank_df['permutation'] = permutation.rank(ascending = False)
    rank_df['SHAP'] = shapvalue.rank(ascending = False)
    rank_df['total_rank'] = rank_df.apply(sum, axis = 1).rank()
    rank_df = rank_df.sort_values('total_rank', ascending = True)
    
    return rank_df


# ### Optuna

# In[159]:


# creator of optuna study for all 3 schemes of stackingCV
def stackingCV_creator(train_data, mode, learner = 'ExtraTrees', num_valid = 5, label = 'GB'):
    
    if learner not in ['LogisticRegression', 'ExtraTrees']:
        raise(f'{learner} is not implemented !')
    
    def objective(trial):
        
        if mode == 'C':
            meta_learner = {
                'meta_learner': trial.suggest_categorical('meta_learner', [learner])
            }
            
            if meta_learner['meta_learner'] == 'LogisticRegression':      
                param = {
                    'solver': 'lbfgs',
                    'C': trial.suggest_categorical('C', [100, 10 ,1 ,0.1, 0.01]),
                    'penalty': trial.suggest_categorical('penalty', ['none', 'l2']),
                    'n_jobs': -1
                }

            elif meta_learner['meta_learner'] == 'ExtraTrees':
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500, step = 200),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 24, step = 2),
                    'max_depth': trial.suggest_int('max_depth', 2, 4, step = 1),
                    'n_jobs': -1
                }
            
            param.update(meta_learner)

        elif mode == 'R':
            meta_learner = 'RidgeCV'
            
            if meta_learner == 'RidgeCV':
                param = {
                    'alpha': trial.suggest_float('alpha', 0, 1, step = 0.1)
                }
            
            elif meta_learner == 'Extra Trees':
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500, step = 100),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 32, step = 5),
                    'max_depth': trial.suggest_int('max_depth', 3, 21, step = 3),
                    'n_jobs': -1
                }
        
        # objective function
        result_list = []
        for i in range(num_valid):

            train_good = train_data[train_data.GB == 0]
            train_bad = train_data[train_data.GB == 1]
            train_good_x, train_good_y = label_divide(train_good, None, label, train_only = True)
            train_bad_x, train_bad_y = label_divide(train_bad, None, label, train_only = True)
            train_g_x, valid_g_x, train_g_y, valid_g_y = train_test_split(train_good_x, train_good_y, test_size = 0.25)
            train_b_x, valid_b_x, train_b_y, valid_b_y = train_test_split(train_bad_x, train_bad_y, test_size = 0.25)
            train_x = pd.concat([train_g_x, train_b_x], axis = 0)
            train_y = pd.concat([train_g_y, train_b_y], axis = 0)
            valid_x = pd.concat([valid_g_x, valid_b_x], axis = 0)
            valid_y = pd.concat([valid_g_y, valid_b_y], axis = 0)

            if mode == 'C':
                result, _ = LR(train_x, valid_x, train_y, valid_y, param)
                table = cf_matrix(result, valid_y)
                recall = table['Recall']
                precision = table['Precision']
                beta = 1
                if recall.values > 0:
                    fscore = ((1+beta**2)*recall*precision) / (recall+(beta**2)*precision)
                else:
                    fscore = 0
                result_list.append(fscore)

            elif mode == 'R':
                result, _ = RidgeR(train_x, valid_x, train_y, valid_y, param)
                pr_matrix = PR_matrix(result, valid_y)
                auc = AUC(pr_matrix['Recall'], pr_matrix['Aging Rate'])
                result_list.append((-1)*auc)

        return np.mean(result_list)
    
    return objective


# ## 
'''
# ### Load Data

# In[65]:


### training data ### 
training_month = range(2, 5)

data_dict, trainset_x, trainset_y = multiple_month(training_month, num_set = 10, filename = 'dataset')

print('\nCombined training data:\n')
run_train = multiple_set(num_set = 10)
run_train_x, run_train_y = train_set(run_train)

### testing data ###
run_test = pd.read_csv('test_runhist.csv').iloc[:, 2:]
run_test_x, run_test_y = label_divide(run_test, None, 'GB', train_only = True)
print('\n', 'Dimension of testing data:', run_test.shape)


# ## Base Learner

# ### Hyperparameters for Training Data Transformation

# In[66]:


target_month = range(2, 5)
target_model = ['LightGBM', 'XGBoost', "NeuralNetwork"]
target_iter = {'LightGBM': 25, 'XGBoost': 25, 'NeuralNetwork': 10}


# In[67]:


##### datasets of each month are optimized by by optuna ##### 
base_param_monthC = optimize_base(train_data = data_dict, 
                                  mode = 'C', 
                                  TPE_multi = False, 
                                  base_list = target_model,
                                  iter_dict = target_iter,
                                  filename = 'runhist_array_m2m4_m5_3criteria')


# In[14]:


##### or load hyperparmeters of base learner from stackingCV scheme 2 #####
base_param_monthC = month_param(date = '20220315', 
                                month_list = list(target_month), 
                                model_list = target_model, 
                                iter_dict = target_iter, 
                                filename = 'runhist_array_m2m4_m5_3criteria', 
                                mode = 'C', 
                                TPE_multi = False)


# ### Hyperparameters for Testing Data Transformation

# In[68]:


##### datasets of whole are optimized by by optuna ##### 
base_param_allC = optimize_base(train_data = {'all': run_train}, 
                                mode = 'C', 
                                TPE_multi = False, 
                                base_list = target_model, 
                                iter_dict = target_iter,
                                filename = 'runhist_array_m2m4_m5_3criteria')


# In[115]:


##### or load hyperparmeters of base learner from stackingCV scheme 1 #####
base_param_allC = all_param(date = '20220308', 
                            model_list = target_model, 
                            iter_dict = target_iter, 
                            filename = 'runhist_array_m2m4_m5_3criteria', 
                            mode = 'C', 
                            TPE_multi = False)


# 
# ### Data Transform

# In[71]:


print('Transform Training Data:')
train_firstC = transform_train(data_dict, 
                               mode = 'C', 
                               base_param = base_param_monthC, 
                               cv = 5)
print('\nTransform Testing Data:')
test_firstC = transform_test(run_train, 
                             run_test, 
                             mode = 'C', 
                             base_param = base_param_allC)
train_firstC_x, train_firstC_y = train_set(train_firstC)
test_firstC_x, test_firstC_y = train_set(test_firstC) 

# ignore
train_firstC['set0'] = {}


# ## Meta Learner

# ### Search for The Best Hyperparameters

# In[130]:


best_paramC, all_scoreC = all_optuna(all_data = train_firstC, 
                                     mode = 'C', 
                                     TPE_multi = False, 
                                     n_iter = 10,
                                     filename = 'runhist_array_m2m4_m5_3criteria_StackingCV3',
                                     creator = stackingCV_creator)


# In[131]:


##### optimization history plot #####
optuna_history(best_paramC, all_scoreC, num_row = 3, num_col = 3, model = 'StackingCV Scheme3 Classifier')
            
##### best hyperparameter table #####
param_table = pd.DataFrame(best_paramC).T
param_table


# ### Feature Importance of Meta Learner

# In[199]:


target_set = 5
rank_importance(train_firstC[f'set{target_set}'], mode = 'C')


# ### Classifier

# In[132]:


table_setC, coefC = runall_LR(train_firstC_x, test_firstC_x, train_firstC_y, test_firstC_y, best_paramC)
line_chart(table_setC, title = 'StackingCV Classifier (Scheme 3)')


# In[133]:


table_setC


# ### Export

# In[17]:


savedate = '20220315'
TPE_multi = False

table_setC['sampler'] = 'multivariate-TPE' if TPE_multi else 'univariate-TPE'
table_setC['model'] = 'StackingCV3'
with pd.ExcelWriter(f'{savedate}_Classifier.xlsx', mode = 'a') as writer:
    table_setC.to_excel(writer, sheet_name = 'StackingCV3')
'''
