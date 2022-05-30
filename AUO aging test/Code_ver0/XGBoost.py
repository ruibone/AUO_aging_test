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
import plotly
import matplotlib.pyplot as plt

import xgboost as xgb
import optuna
from sklearn.model_selection import GridSearchCV, train_test_split

from Dataset_Construction import Balance_Ratio 
from Sampling import label_divide
from AdaClassifier import train_set, multiple_set, print_badC, bad_plot, line_chart, cf_matrix
from AdaRegressor import AUC, PR_curve, multiple_curve, PR_matrix, best_threshold 
from Aging_Score import score1
'''
os.chdir('C:/Users/user/Desktop/Darui_R08621110') 
os.getcwd()
'''

# ### Boosting model

# In[2]:


def XGBoostC(train_x, test_x, train_y, test_y, config):
    
    clf = xgb.XGBClassifier(**config, n_jobs = -1)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result
    
    
def XGBoostR(train_x, test_x, train_y, test_y, config):
    
    reg = xgb.XGBRegressor(**config, n_jobs = -1)
    reg.fit(train_x, train_y)
    predict_y = reg.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})

    return result


# ### Run all dataset

# In[3]:


def runall_XGBoostC(num_set, trainset_x, test_x, trainset_y, test_y, config, record_bad = True):
    
    table_set = pd.DataFrame()
    bad_set = pd.DataFrame()
    judge = list(config.keys())[0]

    for i in range(num_set):
        print('\n', f'Dataset {i}:')

        if isinstance(config[judge], dict) :
            best_config = config[f'set{i}']
        else :
            best_config = config
            
        result = XGBoostC(trainset_x[f'set{i}'], test_x, trainset_y[f'set{i}'], test_y, best_config)    
        table = cf_matrix(result, trainset_y[f'set{i}'])
        table_set = pd.concat([table_set, table]).rename(index = {0: f'dataset {i}'})
        
        if record_bad:
            bad_table = print_badC(result, test_x, Bad_Types) 
            bad_set = pd.concat([bad_set, bad_table]).rename(index = {0: f'dataset {i}'})

    if record_bad:
        return table_set, bad_set
    else:
        return table_set
    
    
def runall_XGBoostR(num_set, trainset_x, test_x, trainset_y, test_y, config, thres_target = 'Recall', threshold = 0.8, 
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

        predict = XGBoostR(trainset_x[f'set{i}'], test_x, trainset_y[f'set{i}'], test_y, best_config)     
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
print('\ntesting data:', test.shape, '\nBalance Ratio:', Balance_Ratio(test), '\n')

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
#         'objective': 'binary:logistic',
#         'n_estimators': 200,
#         'subsample': 0.5,
#         'min_child_weight': 3,
#         'max_depth': 7,
#         'learning_rate': 0.425,
#         'reg_alpha': 0.001,
#         'reg_lambda': 0.0005,
# } ###tpe/recall-0.1*aging/set6

#table_set1, bad_set1 = runall_XGBoostC(10, trainset_x, test_x, trainset_y, test_y, event_clf_param)
table_set1 = runall_XGBoostC(10, trainset_x, run_test_x, trainset_y, run_test_y, best_paramC, record_bad = False)
line_chart(table_set1, title = 'XGBoost Classfifer')
#bad_plot(bad_set1)


# In[ ]:


table_set1


# ### Regression

# In[ ]:


# run_reg_param = {
#         'objective': 'binary:logistic',
#         'n_estimators': 150,
#         'subsample': 0.7,
#         'min_child_weight': 9,
#         'max_depth': 7,
#         'learning_rate': 0.325,
#         'reg_alpha': 0.25,
#         'reg_lambda': 0.06
# } #tpe/auc/set6

# pr_dict, table_set2, bad_set2 = runall_XGBoostR(10, trainset_x, test_x, trainset_y, test_y, event_reg_param, 
#                                                 thres_target = 'Recall', threshold = 0.8)
pr_dict, table_set2 = runall_XGBoostR(10, trainset_x, run_test_x, trainset_y, run_test_y, best_paramR, 
                                       thres_target = 'Recall', threshold = 0.7, record_bad = False)
line_chart(table_set2, title = 'XGBoost Regressor')
#bad_plot(bad_set2)


# In[ ]:


multiple_curve(3, 3, pr_dict, table_set2, target = 'Aging Rate')
multiple_curve(3, 3, pr_dict, table_set2, target = 'Precision')
table_set2
'''

# ## Optimization

# ### Optuna

# In[5]:


def objective_creator(train_data, mode, num_valid = 3) :
    
    def objective(trial) :

        param = {
            'objective': 'binary:logistic',
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step = 50),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9, step = 0.2),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 24, step = 3),
            'max_depth': trial.suggest_int('max_depth', 3, 13, step = 2),
            'learning_rate': trial.suggest_float('learning_rate', 0.025, 0.425, step = 0.05),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 2), # alpha
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 2) # lambda
        }

        result_list = []
        for i in range(num_valid):

            train_x, train_y = label_divide(train_data, None, 'GB', train_only = True)
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.25)

            if mode == 'C':
                result = XGBoostC(train_x, valid_x, train_y, valid_y, param)
                table = cf_matrix(result, valid_y)
                recall = table['Recall']
                aging = table['Aging Rate']
                effi = table['Efficiency']

                #result_list.append(effi)
                result_list.append(recall - 0.1*aging)

            elif mode == 'R':
                result = XGBoostR(train_x, valid_x, train_y, valid_y, param)
                pr_matrix = PR_matrix(result, valid_y)

                #best_data, _ = best_threshold(pr_matrix, target = 'Recall', threshold = 0.8)
                #aging = best_data['Aging Rate']
                #result_list.append((-1)*aging)

                auc = AUC(pr_matrix['Recall'], pr_matrix['Aging Rate'])
                result_list.append((-1)*auc)

        return np.mean(result_list)
    
    return objective


def all_optuna(num_set, all_data, mode, TPE_multi, n_iter, filename, creator, num_valid = 3, return_addition = True) :

    best_param = {}
    #all_study = {}
    all_score = {}
    for i in tqdm(range(num_set)) :
        
        ##### define objective function and change optimized target dataset in each loop #####
        objective = creator(train_data = all_data[f'set{i}'], mode = mode, num_valid = num_valid)
        
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
    multi_mode = 'multivariate-TPE' if TPE_multi else 'univariate-TPE'
    with open(f'{filename}{mode}_{multi_mode}_{n_iter}.data', 'wb') as f:
        pickle.dump(best_param, f)
    
    if return_addition :
        return best_param, all_score#, all_study
    else :
        return best_param
    

def optuna_history(best_param, all_score, num_row, num_col, model = 'XGBoost Classifier') :

    fig, axs = plt.subplots(num_row, num_col, figsize = (num_row*10, num_col*5))
    plt.suptitle(f'Optimization History of {model}', y = 0.94, fontsize = 25)    
    for row in range(num_row):
        for col in range(num_col):
            index = num_col*row + col
            
            if index < len(best_param) :
                axs[row, col].plot(range(len(all_score[f'set{index}'])), all_score[f'set{index}'], 'r-', linewidth = 1)
                axs[row, col].set_title(f'Dataset {index}')
                axs[row, col].set_xlabel('Iterations')
                axs[row, col].set_ylabel('Values')

'''
# In[ ]:


#####for single dataset#####
study = optuna.create_study(sampler = optuna.samplers.TPESampler(multivariate = False), direction = 'maximize') 
#TPE, Random, Grid, CmaEs#
objective = objective_creator(train_data = data_dict['set6'], mode = 'C', num_valid = 3)
study.optimize(objective, n_trials = 5, show_progress_bar = True, gc_after_trial = True) #n_trials or timeout

#####store the result#####
with open(f'one_study.data', 'wb') as f:
    pickle.dump(study, f)
    
# with open(f'one_study.data', 'rb') as f:
#     load_study = pickle.load(study, f)
    
##### hyperparameter importance #####
#importances = optuna.importance.get_param_importances(study)
#importances.optuna.importance.get_param_importances(study, evaluator = optuna.importance.FanovaImportanceEvaluator())
importance_fig = optuna.visualization.plot_param_importances(study)
slice_fig = optuna.visualization.plot_slice(study)
importance_fig.show()
slice_fig.show()

##### top 20 hyper-parameters#####
all_value = []
[all_value.append(x.values) for x in study.trials]
val = np.array(all_value)
best_val = np.flip(val.argsort(axis = 0))[0:20]
val_table = pd.DataFrame()
for i in best_val:
    temp_table = pd.DataFrame(pd.Series(study.trials[i[0]].params)).T
    temp_table['value'] = study.trials[i[0]].value
    val_table = pd.concat([val_table, temp_table])

val_table = val_table.reset_index(drop = True)

##### value loss plot #####
val_tpe = val
#val_mtpe = val

fig = plt.figure(figsize = (15,8))
plt.plot(val_tpe, 'b--', linewidth = 0.2, label = 'TPE')
#plt.plot(val_mtpe, 'r--', linewidth = 0.2, label = 'MTPE')
plt.title('Optimized Values of XGBoost Regressor (aging rate)')
plt.xlabel('Iterations')
plt.ylabel('Values')
#plt.ylim((0.94, 0.97))
plt.legend(loc = 'lower right', frameon = False)


# In[ ]:


best_paramC, all_scoreC = all_optuna(num_set = 10, all_data = data_dict, mode = 'C', TPE_multi = False, n_iter = 250)
best_paramR, all_scoreR = all_optuna(num_set = 10, all_data = data_dict, mode = 'R', TPE_multi = False, n_iter = 250)


# In[ ]:


##### optimization history plot #####
optuna_history(best_paramC, all_scoreC, model = 'XGBoost Classifier')
            
##### best hyperparameter table #####
param_table = pd.DataFrame(best_paramC).T
param_table


# ### Grid Search

# In[ ]:


clf = xgb.XGBClassifier(n_estimators = 50, 
                        learning_rate = 0.1, 
                        min_child_weight = 4, 
                        subsample = 0.7, 
                        max_depth = 9, 
                        reg_lambda = 0.2
                       )

reg = xgb.XGBRegressor(n_estimators = 200, 
                        learning_rate = 0.1, 
                        min_child_weight = 4, 
                        subsample = 0.7, 
                        max_depth = 7, 
                        reg_lambda = 0.2
                       )

param_dict = {
            'n_estimators': [100, 150,200],
            'learning_rate': [0.1, 0.2],
            'min_child_weight': [4, 5, 6, 7, 8],
            'subsample': [0.7],
            'max_depth': [3, 5, 7, 9],
            'reg_lambda':np.array([0.2])
            }

fit_params = {'early_stopping_rounds': 10}

grid_search = GridSearchCV(clf, param_grid = param_dict, scoring = 'precision', cv = 3, verbose = 10, n_jobs = -1)

train_x, train_y = label_divide(data_dict['set5'], None, train_only = True)
result = grid_search.fit(train_x, train_y)
'''
