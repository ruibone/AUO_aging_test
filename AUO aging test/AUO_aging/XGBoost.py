#!/usr/bin/env python
# coding: utf-8

# In[19]:


import os
import time
import itertools
import numpy as np
import pandas as pd
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
os.chdir('C:/Users/Darui Yen/OneDrive/桌面/data_after_mid') 
os.getcwd()
'''

# ### Boosting model

# In[9]:


def XGBoostC(train_x, test_x, train_y, test_y, config):
    
    clf = xgb.XGBClassifier(**config)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result
    
    
def XGBoostR(train_x, test_x, train_y, test_y, config):
    
    reg = xgb.XGBRegressor(**config)
    reg.fit(train_x, train_y)
    predict_y = reg.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})

    return result


# ### Run all dataset

# In[10]:


def runall_XGBoostC(num_set, trainset_x, test_x, trainset_y, test_y, config, record_bad = True):
    
    table_set = pd.DataFrame()
    bad_set = pd.DataFrame()

    for i in range(num_set):
        print('\n', f'Dataset {i}:')
        
        result = XGBoostC(trainset_x[f'set{i}'], test_x, trainset_y[f'set{i}'], test_y, config)
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

    for i in range(num_set):
        print('\n', f'Dataset {i}:')

        predict = XGBoostR(trainset_x[f'set{i}'], test_x, trainset_y[f'set{i}'], test_y, config)
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

# In[8]:


###bad types###
bad = pd.read_csv('original_data/Bad_Types.csv').iloc[:, 1:]
Bad_Types = {bad.cb[i]:i for i in range (len(bad))}
print('Total bad types:', len(bad))

###single dataset###
test = pd.read_csv('original_data/TestingSet_0.csv').iloc[:, 2:]
train = pd.read_csv('original_data/TrainingSet_new.csv').iloc[:, 2:]
print('\ntraining data:', train.shape, '\nBalance Ratio:', Balance_Ratio(train))
print('\ntesting data:', test.shape, '\nBalance Ratio:', Balance_Ratio(test))

train_x, train_y, test_x, test_y = label_divide(train, test, 'GB')

###multiple dataset###
data_dict = multiple_set(num_set = 9)
trainset_x, trainset_y = train_set(data_dict, num_set = 9, label = 'GB')
test_x, test_y = label_divide(test, None, 'GB', train_only = True)


#####for runhist dataset#####
# bad = pd.read_csv('run_bad_types.csv').iloc[:, 1:]
# Bad_Types = {bad.cb[i]:i for i in range (len(bad))}
# print('Total bad types:', len(bad))

run_test = pd.read_csv('test_runhist.csv').iloc[:, 2:]
run_test_x, run_test_y = label_divide(run_test, None, 'GB', train_only = True)
print('\n', 'Dimension of run test:', run_test.shape)


# ### Classifier

# In[15]:


start = time.time()

run_clf_param = {
        #'objective': 'binary',
        #'metric': 'binary_logloss',
        'n_estimators': 50,
        'subsample': 0.6,
        'min_child_weight': 4,
        'max_depth': 9,
        'learning_rate': 0.1,
        'reg_lambda': 0.004,
}

#table_set1, bad_set1 = runall_XGBoostC(9, trainset_x, test_x, trainset_y, test_y, event_clf_param)
table_set1 = runall_XGBoostC(9, trainset_x, run_test_x, trainset_y, run_test_y, run_clf_param, record_bad = False)
line_chart(table_set1, title = 'XGBoost Classfifer')
#bad_plot(bad_set1)

end = time.time()
print("\nRun Time：%f seconds" % (end - start))


# In[ ]:


table_set1


# ### Optuna

# In[21]:


def objective(trial, train_data = data_dict['set6'], mode = 'C', num_valid = 3):
    
    param = {
        #'objective': 'binary',
        #'metric': 'binary_logloss',
        #'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'goss']),
        'n_estimators': trial.suggest_int('num_iterations', 100, 300, step = 50),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9, step = 0.1),
        #'num_leaves': trial.suggest_int('num_leaves', 5, 45, step = 5),
        'min_child_weight': trial.suggest_int('min_child_samples', 3, 30, step = 3),
        'max_depth': trial.suggest_int('max_depth', 3, 15, step = 2),
        'learning_rate': trial.suggest_float('learning_rate', 0.025, 0.475, step = 0.05),
        'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-4, 10) # lambda
    }

    result_list = []
    for i in range(num_valid):
        
        train_x, train_y = label_divide(train_data, None, 'GB', train_only = True)
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.25)

        if mode == 'C':
            result = XGBoostC(train_x, valid_x, train_y, valid_y, param)
            table = cf_matrix(result, valid_y)
            recall = table['Recall']
            precision = table['Precision']
            aging = table['Aging Rate']
            result_list.append(recall + precision + 0.1*aging)

        elif mode == 'R':
            result = XGBoostR(train_x, valid_x, train_y, valid_y, param)
            pr_matrix = PR_matrix(result, valid_y)
            
            best_data, _ = best_threshold(pr_matrix, target = 'Recall', threshold = 0.8)
            aging = best_data['Aging Rate']
            result_list.append((-1)*aging)
            
            #auc = AUC(pr_matrix.Recall, pr_matrix.Precision)
            #result_list.append((-1)*auc)

            
    return np.mean(result_list)


# In[22]:


#####Optimization#####
start = time.time()

study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 10, show_progress_bar = True, gc_after_trial = True) #n_trials or timeout
 
print(f"Sampler is {study.sampler.__class__.__name__}")
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

end = time.time()
print("\nRun Time：%f seconds" % (end - start))


# In[ ]:


##### top 50 hyper-parameters & value loss plot#####
all_value = []
[all_value.append(x.values) for x in study.trials]
val = np.array(all_value)
best_val = np.flip(val.argsort(axis = 0))[0:50]

value_fig = 

val_table = pd.DataFrame()
for i in best_val:
    temp_table = pd.DataFrame(pd.Series(study.trials[i[0]].params)).T
    temp_table['value'] = study.trials[i[0]].value
    val_table = pd.concat([val_table, temp_table])

val_table.reset_index(drop = True)
val_table


# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:


start = time.time()

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

end = time.time()
print("\nRun Time：%f seconds" % (end - start))


# ### Regression

# In[84]:


start = time.time()

pr_dict, table_set2, bad_set2 = runall_XGBoostR(9, trainset_x, test_x, trainset_y, test_y, thres_target = 'Recall', 
                                                threshold = 0.8)
line_chart(table_set2, title = 'XGBoost Regressor')
bad_plot(bad_set2)

end = time.time()
print("\nRun Time：%f seconds" % (end - start))


# In[83]:


multiple_curve(3, 3, pr_dict, table_set2, target = 'Aging Rate')
multiple_curve(3, 3, pr_dict, table_set2, target = 'Precision')
table_set2
'''
