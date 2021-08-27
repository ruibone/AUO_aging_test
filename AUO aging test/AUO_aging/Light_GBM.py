#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import itertools
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm
import plotly
import matplotlib.pyplot as plt

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
import optuna
from sklearn.model_selection import train_test_split, KFold
import skopt
import skopt.plots

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

# In[ ]:


def LightGBMC(train_x, test_x, train_y, test_y, config):
    
    clf = LGBMClassifier(**config)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result


def LightGBMR(train_x, test_x, train_y, test_y, config):
    
    reg = LGBMRegressor(**config)
    reg.fit(train_x, train_y)
    predict_y = reg.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result


# ### Run all dataset

# In[ ]:


def runall_LightGBMC(num_set, trainset_x, test_x, trainset_y, test_y, config, record_bad = True):
    
    table_set = pd.DataFrame()
    bad_set = pd.DataFrame()
    judge = list(config.keys())[0]

    for i in range(num_set):
        print('\n', f'Dataset {i}:')
        
        if isinstance(config[judge], dict) :
            best_config = config[f'set{i}']
        else :
            best_config = config
        
        result = LightGBMC(trainset_x[f'set{i}'], test_x, trainset_y[f'set{i}'], test_y, best_config)
        table = cf_matrix(result, trainset_y[f'set{i}'])
        table_set = pd.concat([table_set, table]).rename(index = {0: f'dataset {i}'})
        
        if record_bad:
            bad_table = print_badC(result, test_x, Bad_Types) 
            bad_set = pd.concat([bad_set, bad_table]).rename(index = {0: f'dataset {i}'})

    if record_bad:
        return table_set, bad_set
    else:
        return table_set
    
    
def runall_LightGBMR(num_set, trainset_x, test_x, trainset_y, test_y, config, thres_target = 'Recall', threshold = 0.8, 
                     record_bad = True):
    
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

        predict = LightGBMR(trainset_x[f'set{i}'], test_x, trainset_y[f'set{i}'], test_y, best_config)
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

# In[ ]:


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


# run_clf_param = {
#         'objective': 'binary',
#         'metric': 'binary_logloss',
#         'boosting_type': 'goss',
#         'num_iterations': 100,
#         'subsample': 0.7,
#         'num_leaves': 20,
#         'min_child_samples': 3,
#         'max_depth': 7,
#         'learning_rate': 0.125,
#         'lambda_l1': 0.0006,

#         'lambda_l2': 0.003
# } #tpe/recall-0.1*aging/set6

#table_set1, bad_set1 = runall_LightGBMC(9, trainset_x, test_x, trainset_y, test_y, event_clf_param)
table_set1 = runall_LightGBMC(10, trainset_x, run_test_x, trainset_y, run_test_y, best_paramC, record_bad = False)
line_chart(table_set1, title = 'LightGBM Classifier')
#bad_plot(bad_set1)


# In[ ]:


table_set1


# ### Regressor

# In[ ]:


# run_reg_param = {
#         'objective': 'binary',
#         'metric': 'binary_logloss',
#         'boosting_type': 'gbdt',
#         'num_iterations': 150,
#         'subsample': 0.9,
#         'num_leaves': 20,
#         'min_child_samples': 9,
#         'max_depth': 5,
#         'learning_rate': 0.475,
#         'lambda_l1': 0.0003,
#         'lambda_l2': 0.0006
# }

# pr_dict, table_set2, bad_set2 = runall_LightGBMR(9, trainset_x, test_x, trainset_y, test_y, event_reg_param, 
#                                                  thres_target = 'Recall', threshold = 0.8)
pr_dict, table_set2 = runall_LightGBMR(10, trainset_x, run_test_x, trainset_y, run_test_y, best_paramR, 
                                       thres_target = 'Recall', threshold = 0.8, record_bad = False)
line_chart(table_set2, title = 'LightGBM Regressor')
#bad_plot(bad_set2)


# In[ ]:


multiple_curve(3, 3, pr_dict, table_set2, target = 'Aging Rate')
multiple_curve(3, 3, pr_dict, table_set2, target = 'Precision')
table_set2


# ## Optimization

# ### Optuna

# In[ ]:


def objective_creator(train_data, mode, num_valid = 3, label = 'GB') :

    def objective(trial) :
    
        param = {
            'objective': trial.suggest_categorical('objective', ['binary', 'cross_entropy']),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss']),
            'num_iterations': trial.suggest_int('num_iterations', 100, 300, step = 50),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9, step = 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 5, 40, step = 5),
            'min_child_samples': trial.suggest_int('min_child_samples', 3, 24, step = 3),
            'max_depth': trial.suggest_int('max_depth', 3, 15, step = 2),
            'learning_rate': trial.suggest_float('learning_rate', 0.025, 0.425, step = 0.05),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 5), # alpha
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 5) # lambda
        }

        result_list = []
        for i in range(num_valid):

            train_x, train_y = label_divide(train_data, None, label, train_only = True)
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.25)

            if mode == 'C':
                result = LightGBMC(train_x, valid_x, train_y, valid_y, param)
                table = cf_matrix(result, valid_y)
                recall = table['Recall']
                aging = table['Aging Rate']
                effi = table['Efficiency']
            
                #result_list.append(effi)
                result_list.append(recall - 0.1*aging)

            elif mode == 'R':
                result = LightGBMR(train_x, valid_x, train_y, valid_y, param)
                pr_matrix = PR_matrix(result, valid_y)

                #best_data, _ = best_threshold(pr_matrix, target = 'Recall', threshold = 0.8)
                #aging = best_data['Aging Rate']
                #result_list.append((-1)*aging)

                auc = AUC(pr_matrix['Recall'], pr_matrix['Aging Rate'])
                result_list.append((-1)*auc)

        return np.mean(result_list)

    return objective


def all_optuna(num_set, all_data, mode, TPE_multi, n_iter, num_valid = 3, return_addition = True) :

    best_param = {}
    #all_study = {}
    all_score = {}
    for i in tqdm(range(num_set)) :
        
        ##### define objective function and change optimized target dataset in each loop #####
        objective = objective_creator(train_data = data_dict[f'set{i}'], mode = mode, num_valid = num_valid)
        
        ##### optimize one dataset in each loop #####
        print(f'Dataset{i} :')
        
        study = optuna.create_study(sampler = optuna.samplers.TPESampler(multivariate = TPE_multi), 
                                       direction = 'maximize')
        study.optimize(objective, n_trials = n_iter, show_progress_bar = True, gc_after_trial = True)
        #n_trials or timeout
        best_param[f'set{i}'] = study.best_trial.params
        
        ##### return score and entire params for score plot or feature importance
        if return_addition :
            collect_score = []
            [collect_score.append(x.values) for x in study.trials]
            #all_study[f'set{i}'] = study
            all_score[f'set{i}'] = collect_score 
        
        print(f"Sampler is {study.sampler.__class__.__name__}")
    
    ##### store the best hyperparameters #####
    multi_mode = 'multivariate' if TPE_multi else 'univariate'
    with open(f'runhist_array_m2m5_LightGBM{mode}_{multi_mode}-TPE_{n_iter}.data', 'wb') as f:
        pickle.dump(best_param, f)
    
    if return_addition :
        return best_param, all_score#, all_study
    else :
        return best_param


# In[ ]:


best_paramC, all_scoreC = all_optuna(num_set = 10, all_data = data_dict, mode = 'C', TPE_multi = False, n_iter = 250)
best_paramR, all_scoreR = all_optuna(num_set = 10, all_data = data_dict, mode = 'R', TPE_multi = False, n_iter = 250)


# In[ ]:


##### optimization history plot #####
optuna_history(best_paramC, all_scoreC, model = 'XGBoost Classifier')
            
##### best hyperparameter table #####
param_table = pd.DataFrame(best_paramC).T
param_table


# ### Scikit Optimize

# In[ ]:


def skopt_creator(train_data, mode, num_valid = 3, label = 'GB') :
    
    def skopt_objective(param) :

        param_dict = {
            'objective': 'binary',
            'metric': 'binary_loss',
            'boosting_type': param[0],
            'num_iterations': param[1],
            'subsample': param[2],
            'num_leaves': param[3],
            'min_child_samples': param[4],
            'max_depth': param[5],
            'learning_rate': param[6],
            'reg_alpha': param[7],
            'reg_lambda': param[8]
        }
        
        result_list = []
        for i in range(num_valid):

            train_x, train_y = label_divide(train_data, None, label, train_only = True)
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.25)

            if mode == 'C':
                result = LightGBMC(train_x, valid_x, train_y, valid_y, param_dict)
                table = cf_matrix(result, valid_y)
                recall = table['Recall']
                aging = table['Aging Rate']

                result_list.append(0.1*aging - recall)

            elif mode == 'R':
                result = LightGBMR(train_x, valid_x, train_y, valid_y, param_dict)
                pr_matrix = PR_matrix(result, valid_y)
                auc = AUC(pr_matrix['Recall'], pr_matrix['Aging Rate'])

                result_list.append(auc)

        return np.mean(result_list)
        ##### minimize
    
    return skopt_objective


def all_skopt(num_set, all_data, mode, n_iter, optimizer, sel_func, num_valid = 3, record_addition = True) :
    
    skopt_param = [
        skopt.space.space.Categorical(name = 'boosting_type', categories = ['gbdt', 'goss']),
        skopt.space.space.Categorical(name = 'num_iterations', categories = [100, 150, 200, 250, 300]),
        skopt.space.space.Categorical(name = 'subsample', categories = [0.5, 0.7, 0.9]),
        skopt.space.space.Integer(name = 'num_leaves', low = 5, high = 40),
        skopt.space.space.Integer(name = 'min_child_samples', low = 3, high = 24),
        skopt.space.space.Integer(name = 'max_depth', low = 3, high = 15),
        skopt.space.space.Real(name = 'learning_rate', low = 0.01, high = 0.4, prior = 'uniform'),
        skopt.space.space.Real(name = 'reg_alpha', low = 1e-4, high = 5, prior = 'log-uniform'), # alpha
        skopt.space.space.Real(name = 'reg_lambda', low = 1e-4, high = 5, prior = 'log-uniform') # lambda
    ]
    
    opt_list = ['GauP', 'forest', 'gbrt']
    
    best_param = {}
    record_studies = {}
    for i in tqdm(range(num_set)) :
        
        skopt_objective = skopt_creator(all_data[f'set{i}'], mode = mode)
        if optimizer == opt_list[0] :
            result = skopt.gp_minimize(skopt_objective, skopt_param, n_calls = n_iter, acq_func = sel_func, n_jobs = -1)
        elif optimizer == opt_list[1] :
            result = skopt.forest_minimize(skopt_objective, skopt_param, n_calls = n_iter, acq_func = sel_func, 
                                           n_jobs = -1)
        elif optimizer == opt_list[2] :
            result = skopt.gbrt_minimize(skopt_objective, skopt_param, n_calls = n_iter, acq_func = sel_func, 
                                           n_jobs = -1)
        
        # return to dictionary
        record_param = result.x
        dict_param = {
            'objective': 'binary',
            'metric': 'binary_loss',
            'boosting_type': record_param[0],
            'num_iterations': record_param[1],
            'subsample': record_param[2],
            'num_leaves': record_param[3],
            'min_child_samples': record_param[4],
            'max_depth': record_param[5],
            'learning_rate': record_param[6],
            'reg_alpha': record_param[7],
            'reg_lambda': record_param[8]
        }
        
        best_param[f'set{i}'] = dict_param
        if record_addition :
            record_studies[f'set{i}'] = result
        
    # save the hyperparameter dictionary
    with open(f'runhist_array_label_LightGBM{mode}_{optimizer}_{n_iter}.data', 'wb') as f :
        pickle.dump(best_param, f)

    return best_param, record_studies


# In[ ]:


best_paramC, all_studiesC = all_skopt(num_set = 9, all_data = data_dict, mode = 'C', n_iter = 500, 
                                      optimizer = 'GauP', sel_func = 'EI', num_valid = 3)
best_paramR, all_studiesR = all_skopt(num_set = 9, all_data = data_dict, mode = 'R', n_iter = 250, 
                                      optimizer = 'GauP', sel_func = 'EI', num_valid = 3)


# In[ ]:


##### convergence plot #####
all_studies  = all_studiesC

plt.figure(figsize = (12, 8))
convergence = skopt.plots.plot_convergence(
    ('dataset 0', all_studies['set0']),
    ('dataset 1', all_studies['set1']),
    ('dataset 2', all_studies['set2']),
    ('dataset 3', all_studies['set3']),
    ('dataset 4', all_studies['set4']),
    ('dataset 5', all_studies['set5']),
    ('dataset 6', all_studies['set6']),
    ('dataset 7', all_studies['set7']),
    ('dataset 8', all_studies['set8'])
)
convergence.legend(loc = "upper right", prop = {'size': 8})
convergence.set_title('Convergence Plot of LightGBM Classifier (gradient boost)')
convergence.set_xlabel('Iterations')
convergence.set_ylabel('Objective Values')
'''
