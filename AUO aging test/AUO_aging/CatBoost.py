#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import itertools
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import catboost
from catboost import CatBoostClassifier, CatBoostRegressor
import optuna
from sklearn.model_selection import train_test_split, KFold

from Dataset_Construction import Balance_Ratio 
from Sampling import label_divide
from AdaClassifier import train_set, multiple_set, print_badC, bad_plot, line_chart, cf_matrix
from AdaRegressor import AUC, PR_curve, multiple_curve, PR_matrix, best_threshold 
from Aging_Score import score1
from XGBoost import optuna_history
'''
os.chdir('C:/Users/user/Desktop/Darui_R08621110') 
os.getcwd()
'''

# ### Boosting model

# In[2]:


def CatBoostC(train_x, test_x, train_y, test_y, config, cat_feature):
    
    clf = CatBoostClassifier(**config, verbose = 0)
    clf.fit(train_x, train_y, cat_features = cat_feature)
    predict_y = clf.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result


def CatBoostR(train_x, test_x, train_y, test_y, config, cat_feature):
    
    reg = CatBoostRegressor(**config, verbose = 0)
    reg.fit(train_x, train_y, cat_features = cat_feature)
    predict_y = reg.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})

    return result


# ### Run all dataset

# In[3]:


def runall_CatBoostC(num_set, trainset_x, test_x, trainset_y, test_y, config, cat_feature, record_bad = True):
    
    table_set = pd.DataFrame()
    bad_set = pd.DataFrame()
    judge = list(config.keys())[0]

    for i in range(num_set):
        print('\n', f'Dataset {i}:')
        
        if isinstance(config[judge], dict) :
            best_config = config[f'set{i}']
        else :
            best_config = config
        
        result = CatBoostC(trainset_x[f'set{i}'], test_x, trainset_y[f'set{i}'], test_y, best_config, cat_feature)
        table = cf_matrix(result, trainset_y[f'set{i}'])
        table_set = pd.concat([table_set, table]).rename(index = {0: f'dataset {i}'})
        
        if record_bad:
            bad_table = print_badC(result, test_x, Bad_Types) 
            bad_set = pd.concat([bad_set, bad_table]).rename(index = {0: f'dataset {i}'})

    if record_bad:
        return table_set, bad_set
    else:
        return table_set
    
    
def runall_CatBoostR(num_set, trainset_x, test_x, trainset_y, test_y, config, cat_feature, thres_target = 'Recall', 
                     threshold = 0.8, record_bad = True):
    
    table_set = pd.DataFrame()
    bad_set = pd.DataFrame()
    pr_dict = {}
    judge = list(config.keys())[0]

    for i in range(num_set):
        print('\n', f'Dataset {i}:')
        
        if isinstance(config[judge], dict) :
            best_config = config[f'set{i}']
        else :
            best_config = config

        predict = CatBoostR(trainset_x[f'set{i}'], test_x, trainset_y[f'set{i}'], test_y, best_config, cat_feature)
        pr_matrix = PR_matrix(predict, trainset_y[f'set{i}'])
        pr_dict[f'set{i}'] = pr_matrix
        
        best_data, best_thres = best_threshold(pr_matrix, target = thres_target, threshold = threshold)
        table_set = pd.concat([table_set, best_data]).rename(index = {best_data.index.values[0]: f'dataset {i}'})
        
        if record_bad:
            bad_table = print_badC(predict, test_x, Bad_Types, threshold = best_thres)
            bad_set = pd.concat([bad_set, bad_table]).rename(index = {0: f'dataset {i}'})
    
    if record_bad:
        return pr_dict, table_set, bad_set
    else:
        return pr_dict, table_set

'''
# ## Data Processing

# In[4]:


###bad types###
bad = pd.read_csv('event/Bad_Types.csv').iloc[:, 1:]
Bad_Types = {bad.cb[i]:i for i in range (len(bad))}
print('Total bad types:', len(bad))

###single dataset###
test = pd.read_csv('event/TestingSet_0.csv').iloc[:, 2:]
train = pd.read_csv('event/TrainingSet_new.csv').iloc[:, 2:]
print('\ntraining data:', train.shape, '\nBalance Ratio:', Balance_Ratio(train))
print('\ntesting data:', test.shape, '\nBalance Ratio:', Balance_Ratio(test))

train_x, train_y, test_x, test_y = label_divide(train, test, 'GB')

###multiple dataset###
data_dict = multiple_set(num_set = 10)
trainset_x, trainset_y = train_set(data_dict, num_set = 10, label = 'GB')
test_x, test_y = label_divide(test, None, 'GB', train_only = True)


#####for runhist dataset#####
# bad = pd.read_csv('run_bad_types.csv').iloc[:, 1:]
# Bad_Types = {bad.cb[i]:i for i in range (len(bad))}
# print('Total bad types:', len(bad))

run_test = pd.read_csv('test_runhist.csv').iloc[:, 2:]
run_test_x, run_test_y = label_divide(run_test, None, 'GB', train_only = True)
print('\n', 'Dimension of run test:', run_test.shape)


# ### Classifier

# In[ ]:


#table_set1, bad_set1 = runall_CatBoostC(9, trainset_x, test_x, trainset_y, test_y, cat_feature = [], event_clf_param)
table_set1 = runall_CatBoostC(10, trainset_x, run_test_x, trainset_y, run_test_y, best_paramC, cat_feature = [],
                              record_bad = False)
line_chart(table_set1, title = 'CatBoost Classifier')
#bad_plot(bad_set1)


# In[ ]:


table_set1


# ### Regressor

# In[ ]:


# pr_dict, table_set2, bad_set2 = runall_CatBoostR(9, trainset_x, test_x, trainset_y, test_y, event_reg_param, 
#                                                  thres_target = 'Recall', threshold = 0.8)
pr_dict, table_set2 = runall_CatBoostR(10, trainset_x, run_test_x, trainset_y, run_test_y, best_paramR, cat_feature = [], 
                                       thres_target = 'Recall', threshold = 0.8, record_bad = False)
line_chart(table_set2, title = 'CatBoost Regressor')
#bad_plot(bad_set2)


# In[ ]:


multiple_curve(3, 3, pr_dict, table_set2, target = 'Aging Rate')
multiple_curve(3, 3, pr_dict, table_set2, target = 'Precision')
table_set2


# ## Optimization

# ### Optuna

# In[5]:


def objective_creator(train_data, mode, cat_feature, num_valid = 3, label = 'GB') :

    def objective(trial) :
    
        param_1 = {
            #'objective': 'CrossEntropy',
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'iterations': trial.suggest_int('iterations', 100, 300, step = 50),
            'depth': trial.suggest_int('depth', 2, 12, step = 2),
            'learning_rate': trial.suggest_float('learning_rate', 0.025, 0.325, step = 0.05),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9, step = 0.2),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-4, 10) # lambda
        }

        if param_1['grow_policy'] == 'Depthwise' :
            param_2 = {
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 3, 30, step = 3)
            }
            param = {**param_1, **param_2}
        
        elif param_1['grow_policy'] == 'Lossguide' :
            param_3 = {
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 3, 30, step = 3),
                'max_leaves': trial.suggest_int('max_leaves', 15, 50, step = 5)
            }
            param = {**param_1, **param_3}
        
        else :
            param = param_1

        result_list = []
        for i in range(num_valid):

            train_x, train_y = label_divide(train_data, None, label, train_only = True)
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.25)

            if mode == 'C':
                result = CatBoostC(train_x, valid_x, train_y, valid_y, param, cat_feature)
                table = cf_matrix(result, valid_y)
                recall = table['Recall']
                aging = table['Aging Rate']
                effi = table['Efficiency']
            
                #result_list.append(effi)
                result_list.append(recall - 0.1*aging)

            elif mode == 'R':
                result = CatBoostR(train_x, valid_x, train_y, valid_y, param, cat_feature)
                pr_matrix = PR_matrix(result, valid_y)

                #best_data, _ = best_threshold(pr_matrix, target = 'Recall', threshold = 0.8)
                #aging = best_data['Aging Rate']
                #result_list.append((-1)*aging)

                auc = AUC(pr_matrix['Recall'], pr_matrix['Aging Rate'])
                result_list.append((-1)*auc)

        return np.mean(result_list)

    return objective


def all_optuna(num_set, all_data, mode, cat_feature, TPE_multi, n_iter, num_valid = 3, return_addition = True) :

    best_param = {}
    #all_study = {}
    all_score = {}
    for i in tqdm(range(num_set)) :
        
        ##### define objective function and change optimized target dataset in each loop #####
        objective = objective_creator(train_data = data_dict[f'set{i}'], mode = mode, cat_feature = cat_feature,
                                      num_valid = num_valid)
        
        ##### optimize one dataset in each loop #####
        print(f'Dataset{i} :')
        
        study = optuna.create_study(sampler = optuna.samplers.TPESampler(multivariate = TPE_multi), 
                                       direction = 'maximize')
        study.optimize(objective, n_trials = n_iter, show_progress_bar = True, gc_after_trial = True)
        #n_trials or timeout
        best_param[f'set{i}'] = study.best_trial.params
        
        ##### return score and entire params for score plot or feature importance #####
        if return_addition :
            collect_score = []
            [collect_score.append(x.values) for x in study.trials]
            #all_study[f'set{i}'] = study
            all_score[f'set{i}'] = collect_score 
        
        print(f"Sampler is {study.sampler.__class__.__name__}")
    
    ##### store the best hyperparameters #####
    multi_mode = 'multivariate' if TPE_multi else 'univariate'
    with open(f'runhist_array_m2m5_CatBoost{mode}_{multi_mode}-TPE_{n_iter}.data', 'wb') as f:
        pickle.dump(best_param, f)
    
    if return_addition :
        return best_param, all_score, #all_study
    else :
        return best_param


# In[ ]:


best_paramC, all_scoreC = all_optuna(num_set = 10, all_data = data_dict, mode = 'C', cat_feature = [],
                                     TPE_multi = False, n_iter = 250)
best_paramR, all_scoreR = all_optuna(num_set = 10, all_data = data_dict, mode = 'R', cat_feature = [],
                                     TPE_multi = False, n_iter = 250)


# In[ ]:


##### optimization history plot #####
optuna_history(best_paramC, all_scoreC, model = 'CatBoost Classifier')
            
##### best hyperparameter table #####
param_table = pd.DataFrame(best_paramC).T
param_table
'''
