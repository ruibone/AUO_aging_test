#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm.auto import tqdm

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import optuna
from sklearn.model_selection import train_test_split

from Dataset_Construction import Balance_Ratio 
from Sampling import label_divide
from AdaClassifier import train_set, multiple_set, print_badC, bad_plot, line_chart, cf_matrix
from AdaRegressor import AUC, PR_curve, multiple_curve, PR_matrix, best_threshold 
from Aging_Score import score1
from XGBoost import optuna_history, all_optuna
'''
os.chdir('C:/Users/user/Desktop/Darui_R08621110') 
os.getcwd()
'''

# ### Bagging model

# In[ ]:


def RandomForestC(train_x, test_x, train_y, test_y, config):
    
    clf = RandomForestClassifier(**config, n_jobs = -1)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result


def RandomForestR(train_x, test_x, train_y, test_y, config):
    
    clf = RandomForestRegressor(**config, n_jobs = -1)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result


# ### Run all dataset

# In[ ]:


def runall_ForestC(num_set, trainset_x, test_x, trainset_y, test_y, config, record_bad = True):
    
    table_set = pd.DataFrame()
    bad_set = pd.DataFrame()
    judge = list(config.keys())[0]

    for i in range(num_set):
        print('\n', f'Dataset {i}:')
        
        if isinstance(config[judge], dict) :
            best_config = config[f'set{i}']
        else :
            best_config = config
        
        result = RandomForestC(trainset_x[f'set{i}'], test_x, trainset_y[f'set{i}'], test_y, best_config)
        table = cf_matrix(result, trainset_y[f'set{i}'])
        table_set = pd.concat([table_set, table]).rename(index = {0: f'dataset {i}'})
        
        if record_bad:
            bad_table = print_badC(result, test_x, Bad_Types) 
            bad_set = pd.concat([bad_set, bad_table]).rename(index = {0: f'dataset {i}'})

    if record_bad:
        return table_set, bad_set
    else:
        return table_set
    
    
def runall_ForestR(num_set, trainset_x, test_x, trainset_y, test_y, config, thres_target = 'Recall', threshold = 0.8, 
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

        predict = RandomForestR(trainset_x[f'set{i}'], test_x, trainset_y[f'set{i}'], test_y, best_config)
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


#table_set1, bad_set1 = runall_ForestC(9, trainset_x, test_x, trainset_y, test_y)
table_set1 = runall_ForestC(10, trainset_x, run_test_x, trainset_y, run_test_y, best_paramC, record_bad = False)
line_chart(table_set1, title = 'Random Forest Classifier')
#bad_plot(bad_set1)


# In[ ]:


table_set1


# ### Regressor

# In[ ]:


# pr_dict, table_set2, bad_set2 = runall_ForestR(9, trainset_x, test_x, trainset_y, test_y, thres_target = 'Recall', 
#                                                threshold = 0.8)
pr_dict, table_set2 = runall_ForestR(10, trainset_x, run_test_x, trainset_y, run_test_y, best_paramR, 
                                     thres_target = 'Recall', threshold = 0.7, record_bad = False)
line_chart(table_set2, title = 'Random Forest Regressor')
#bad_plot(bad_set2)


# In[ ]:


multiple_curve(3, 3, pr_dict, table_set2, target = 'Aging Rate')
multiple_curve(3, 3, pr_dict, table_set2, target = 'Precision')
table_set2


# ## Optimization

# ### Optuna

# In[ ]:


def objective_creator(train_data, mode, num_valid = 3) :
    
    def objective(trial) :

        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800, step = 100),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 32, step = 5),
            'max_depth': trial.suggest_int('max_depth', 3, 21, step = 3),
            #'learning_rate': trial.suggest_float('learning_rate', 0.025, 0.425, step = 0.05),
            #'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 2), # alpha
            #'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 2) # lambda
        }

        result_list = []
        for i in range(num_valid):

            train_x, train_y = label_divide(train_data, None, 'GB', train_only = True)
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.25)

            if mode == 'C':
                result = RandomForestC(train_x, valid_x, train_y, valid_y, param)
                table = cf_matrix(result, valid_y)
                recall = table['Recall']
                aging = table['Aging Rate']
                effi = table['Efficiency']

                #result_list.append(effi)
                result_list.append(recall - 0.1*aging)

            elif mode == 'R':
                result = RandomForestR(train_x, valid_x, train_y, valid_y, param)
                pr_matrix = PR_matrix(result, valid_y)

                #best_data, _ = best_threshold(pr_matrix, target = 'Recall', threshold = 0.8)
                #aging = best_data['Aging Rate']
                #result_list.append((-1)*aging)

                auc = AUC(pr_matrix['Recall'], pr_matrix['Aging Rate'])
                result_list.append((-1)*auc)

        return np.mean(result_list)
    
    return objective


# In[ ]:


best_paramC, all_scoreC = all_optuna(num_set = 10, 
                                     all_data = data_dict, 
                                     mode = 'C', 
                                     TPE_multi = True, 
                                     n_iter = 50, 
                                     filename = 'runhist_array_m2m5_4selection_RandomForest', 
                                     creator = objective_creator
                                    )


# In[ ]:


best_paramR, all_scoreR = all_optuna(num_set = 10, 
                                     all_data = data_dict, 
                                     mode = 'R', 
                                     TPE_multi = True, 
                                     n_iter = 50,
                                     filename = 'runhist_array_m2m5_4selection_RandomForest', 
                                     creator = objective_creator
                                    )


# In[ ]:


##### optimization history plot #####
optuna_history(best_paramC, all_scoreC, model = 'RandomForest Classifier')
            
##### best hyperparameter table #####
param_table = pd.DataFrame(best_paramC).T
param_table
'''
