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

from sklearn.ensemble import AdaBoostRegressor
import optuna
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor 

from Dataset_Construction import Balance_Ratio 
from Sampling import label_divide
from AdaClassifier import train_set, multiple_set, print_badC, bad_plot, line_chart
from Aging_Score import score1
'''
os.chdir('C:/Users/user/Desktop/Darui_R08621110')  
os.getcwd()
'''

# ### Boosting Model

# In[ ]:


def AdaBoostR(train_x, test_x, train_y, test_y, config) :
    
    clf = AdaBoostRegressor(**config)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result


# ### Recall & Precision for Regressor

# In[ ]:


def PR_matrix(predict, train_y, prob = 0.5):
    
    Y_new = predict.sort_values(['predict', 'truth'], ascending = [False, True]).reset_index(drop = True)
    Y_new.loc[Y_new['truth'] != 1, 'truth'] = 0
    
    matrix = pd.DataFrame(Y_new.groupby('predict').sum()).rename(columns = {'truth': 'Bad_Count'})
    matrix = matrix.sort_index(ascending = False)
    matrix['All_Count'] = Y_new.groupby('predict').count()
    matrix['Class_Prob'] = matrix.index
    
    matrix['train_OK'] = sum(train_y < 0.5)
    matrix['train_NG'] = len(train_y) - matrix['train_OK'].values[0]
    matrix['Balance Ratio'] = matrix['train_OK'] / matrix['train_NG']
    
    matrix['TP'] = matrix['Bad_Count'].cumsum()
    matrix['FP'] = matrix['All_Count'].cumsum() - matrix['TP']
    matrix['FN'] = matrix['TP'].values[-1] - matrix['TP']
    matrix['TN'] = matrix['FP'].values[-1] - matrix['FP']
    
    matrix['Precision'] = matrix['TP'] / (matrix['TP'] + matrix['FP'])
    matrix['Recall'] = matrix['TP'] / (matrix['TP'] + matrix['FN'])
    matrix['Aging Rate'] = (matrix['TP'] + matrix['FP']) / (matrix['TP'] + matrix['FP'] + matrix['FN'] + matrix['TN'])
    matrix['Efficiency'] = matrix['Recall'] / matrix['Aging Rate']
    matrix['Score'] = score1(matrix['Recall'], matrix['Aging Rate'])
              
    matrix = matrix.drop(columns = ['Bad_Count', 'All_Count']).reset_index(drop = True)
    
    return matrix


def best_threshold(pr_matrix, target, threshold = False):
    
    # input threshold, or find maximum
    if threshold:
        index = pr_matrix[pr_matrix[target] >= threshold].head(1).index.values[0]
    else:
        index = pr_matrix[target].idxmax()
        
    best_data = pr_matrix.loc[index]
    best_thres = best_data['Class_Prob']
    best_data = pd.DataFrame(best_data).T
    print('Best Threshold:', best_thres, '\n')
    print('Recall:', best_data['Recall'].values, ',   Precision:', best_data['Precision'].values,           ',   Aging Rate:', best_data['Aging Rate'].values)

    return best_data, best_thres


# ### Run all dataset

# In[ ]:


def runall_AdaBoostR(num_set, trainset_x, test_x, trainset_y, test_y, config, thres_target = 'Recall', threshold = False, 
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
            
        # seperate the decision tree hyperparameter and adaboost hyperparameter
        tree_param = {'base_estimator': DecisionTreeRegressor(max_depth = best_config['max_depth'])}
        boost_param = dict((key, best_config[key]) for key in ['learning_rate', 'n_estimators'] if key in best_config)
        boost_param.update(tree_param)

        predict = AdaBoostR(trainset_x[f'set{i}'], test_x, trainset_y[f'set{i}'], test_y, boost_param)
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


# ### Plot all dataset

# In[ ]:


def AUC(x, y):
    
    area = 0
    left = x[0]*y[0]
    right = (1 - x[len(x)-1])*y[len(x)-1]
    
    for i in range(1, len(x)):
        wide = x[i] - x[i-1]
        height = (y[i-1] + y[i])/2
        area = area + wide*height
        
    area = left + area + right
    
    return area


def PR_curve(pr_matrix, best_data, title = 'PR_curve'):
    
    plt.plot(pr_matrix['Recall'], pr_matrix['Precision'], 'b-')
    plt.plot(pr_matrix['Recall'], pr_matrix['Precision'], 'r.')
    plt.plot(best_data['Recall'], best_data['Precision'], 'go', markersize = 10)
    print('Precision, Recall, Aging Rate:', best_data['Precision'].values, best_data['Recall'].values, 
          best_data['Aging Rate'].values)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title}')
    plt.show()
    auc = AUC(pr_matrix['Recall'].values, pr_matrix['Precision'].values)
    print('AUC: ', auc, '\n')
    
    
def multiple_curve(row_num, col_num, pr_dict, table_set, target = 'Aging Rate'):
    
    fig, axs = plt.subplots(row_num, col_num, sharex = False, sharey = False, figsize = (row_num*8 + 1, col_num*6))
    plt.suptitle(f'{target} & Recall Curve of Dataset 0 - {len(table_set)}', y = 0.94, fontsize = 30)
    
    for row in range(row_num):
        for col in range(col_num):
            
            index = col_num*row + col
            if index < len(table_set) :
                auc = AUC(pr_dict[f'set{index}']['Recall'].values, pr_dict[f'set{index}'][target].values).round(5)
                ar = table_set["Aging Rate"][index].round(3)
                recall = table_set["Recall"][index].round(3)
                precision = table_set["Precision"][index].round(5)

                axs[row, col].plot(pr_dict[f'set{index}']['Recall'], pr_dict[f'set{index}'][target], 'b-')
                axs[row, col].plot(pr_dict[f'set{index}']['Recall'], pr_dict[f'set{index}'][target], 'r.', markersize = 10)
                axs[row, col].plot(table_set['Recall'][index], table_set[target][index], 'go', markersize = 15)
                axs[row, col].set_xlabel('Recall')
                axs[row, col].set_ylabel(target)

                if target == 'Aging Rate':
                    axs[row, col].set_title(f'dataset {index}, AUC = {auc}, Aging Rate = {ar}, Recall = {recall}, Precision = {precision}')
                elif target == 'Precision':
                    axs[row, col].set_title(f'dataset {index}, AUC = {auc}, Aging Rate = {ar}, Recall = {recall}')

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


# ### Regressor

# In[ ]:


# pr_dict, table_set, bad_set = runall_AdaBoostR(9, trainset_x, test_x, trainset_y, test_y, event_reg_param,
#                                                thres_target = 'Recall', threshold = 0.7)
pr_dict, table_set = runall_AdaBoostR(10, trainset_x, run_test_x, trainset_y, run_test_y, best_paramR,
                                      thres_target = 'Recall', threshold = 0.8, record_bad = False)
line_chart(table_set, title = 'AdaBoost Regressor')
#bad_plot(bad_set)


# In[ ]:


multiple_curve(4, 3, pr_dict, table_set, target = 'Aging Rate')
multiple_curve(4, 3, pr_dict, table_set, target = 'Precision')
table_set


# ## Opitmization

# ### Optuna

# In[ ]:


def objective_creator(train_data, mode, num_valid = 3) :
    
    def objective(trial) :

        tree_param = {
            'max_depth': trial.suggest_int('max_depth', 1, 3)
        }
        
        param = {
            'base_estimator': DecisionTreeRegressor(**tree_param),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step = 50),
            'learning_rate': trial.suggest_float('learning_rate', 0.025, 0.825, step = 0.05),
        }
        
        result_list = []
        for i in range(num_valid):

            train_x, train_y = label_divide(train_data, None, 'GB', train_only = True)
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.25)

            if mode == 'R':
                result = AdaBoostR(train_x, valid_x, train_y, valid_y, param)
                pr_matrix = PR_matrix(result, valid_y)

                #best_data, _ = best_threshold(pr_matrix, target = 'Recall', threshold = 0.8)
                #aging = best_data['Aging Rate']
                #result_list.append((-1)*aging)

                auc = AUC(pr_matrix['Recall'], pr_matrix['Aging Rate'])
                result_list.append((-1)*auc)

        return np.mean(result_list)
    
    return objective


# In[ ]:


best_paramR, all_scoreR = all_optuna(num_set = 10, 
                                     all_data = data_dict, 
                                     mode = 'R', 
                                     TPE_multi = True, 
                                     n_iter = 25,
                                     filename = 'runhist_array_m2m5_4selection_AdaBoost',
                                     creator = objective_creator
                                    )


# In[ ]:


##### optimization history plot #####
optuna_history(best_paramR, all_scoreR, model = 'AdaBoost Regressor')
            
##### best hyperparameter table #####
param_table = pd.DataFrame(best_paramR).T
param_table
'''
