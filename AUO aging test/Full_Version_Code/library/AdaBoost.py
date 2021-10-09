#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pickle
import plotly

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import optuna
from sklearn.model_selection import train_test_split

from library.Data_Preprocessing import Balance_Ratio
from library.Imbalance_Sampling import label_divide
from library.Aging_Score_Contour import score1
'''
os.chdir('C:/Users/user/Desktop/Darui_R08621110')  
os.getcwd()
'''

# ## 

# ### load multiple dataset

# In[ ]:


def multiple_month(month_list, num_set, filename = 'dataset'):
    
    month_dict = {}
    for i in month_list:
        print(f'\nMonth {i}:\n')
        month_dict[f'm{i}'] = multiple_set(num_set = num_set, filename = f'm{i}_{filename}')
        trainset_x, trainset_y = train_set(month_dict[f'm{i}'], num_set = num_set)
        
    return month_dict, trainset_x, trainset_y


def multiple_set(num_set, filename = 'dataset'):
    
    data_dict = {}
    for i in range(num_set):
        data_dict[f'set{i}'] = pd.read_csv(f'{filename}_{i}.csv').iloc[:, 1:]
        print('Dimension of dataset', i, ':', data_dict[f'set{i}'].shape, ' balance ratio:',               Balance_Ratio(data_dict[f'set{i}']))
    
    print('\n', num_set, 'datasets are loaded.')
    return data_dict


def train_set(data_dict, num_set, label = 'GB'):
    
    trainset_x = {}
    trainset_y = {}
    
    for i in range(num_set):
        X, Y = label_divide(data_dict[f'set{i}'], None, label, train_only = True)
        trainset_x[f'set{i}'] = X
        trainset_y[f'set{i}'] = Y
        
    print('\nLabels of ', num_set, 'datasets are divided.')
    return trainset_x, trainset_y


# ### for classifier

# In[ ]:


def cf_matrix(predict, train_y):
    
    # confusion matrix
    mask_FP = predict['predict'] > predict['truth']
    mask_FN = predict['predict'] < predict['truth']
    mask_TP = (predict['predict'] == predict['truth']) * (predict['predict'] == 1)
    mask_TN = (predict['predict'] == predict['truth']) * (predict['predict'] == 0)
    TP = mask_TP.sum()
    FP = mask_FP.sum()
    FN = mask_FN.sum()
    TN = mask_TN.sum()
    
    #balance ratio, train OK & NG
    train_OK = sum(train_y < 0.5)
    train_NG = len(train_y) - train_OK
    br = train_OK / train_NG
    
    #precision, recall, aging rate, efficiency, score
    num_pd = TP + FP
    if num_pd != 0:
        precision = TP / num_pd
    else:
        precision = 0
    
    recall = TP / (TP + FN)
    ar = (TP + FP) / (TP + FP + FN + TN)
    eff = recall / ar
    score = score1(recall, ar)
    
    table = pd.Series({'Balance Ratio': br, 'Train_OK': train_OK, 'Train_NG': train_NG, 'TP': TP, 'FP': FP, 'FN': FN,                        'TN': TN, 'Precision': precision, 'Recall': recall, 'Aging Rate': ar, 'Efficiency': eff,                        'Score': score})
    table = pd.DataFrame(table).T
    
    print('Precision:', precision, '\nRecall:', recall, '\nAging Rate:', ar)
    return  table


# ### for regressor

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


# ### plot

# In[ ]:


def line_chart(table_set, title):
    
    plt.style.use('seaborn-dark-palette')
    
    x = list(range(len(table_set)))
    fig, ax1 = plt.subplots(figsize = (15,8))
    ax2 = ax1.twinx()
    
    plt.title(title, fontsize = 16)
    plt.xticks(range(1,13,1))
    ax1.plot(x, table_set['Aging Rate'], 'b--', linewidth = 1, label = 'Aging Rate')
    ax1.plot(x, table_set['Aging Rate'], 'b.', markersize = 15)
    ax1.plot(x, table_set['Recall'], 'r-', linewidth = 1, label = 'Recall')
    ax1.plot(x, table_set['Recall'], 'r.', markersize = 15)
    ax2.plot(x, table_set['Precision'], 'g--', linewidth = 1, label = 'Precision')
    ax2.plot(x, table_set['Precision'], 'g.', markersize = 15)
    ax1.set_xlabel('\nDataset', fontsize = 12)
    ax1.set_ylabel('Recall & Aging Rate', color = 'b')
    ax2.set_ylabel('Precision', color = 'g')
    
    ax1.legend(loc = 'upper left', frameon = False)
    ax2.legend(loc = 'upper right', frameon = False)
    
    plt.show()
    
    
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


# ### adaboost 

# In[ ]:


def AdaBoostC(train_x, test_x, train_y, test_y, config):
    
    clf = AdaBoostClassifier(**config)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result


def AdaBoostR(train_x, test_x, train_y, test_y, config) :
    
    clf = AdaBoostRegressor(**config)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result


def runall_AdaBoostC(num_set, trainset_x, test_x, trainset_y, test_y, config):
    
    table_set = pd.DataFrame()
    bad_set = pd.DataFrame()
    judge = list(config.keys())[0]

    for i in tqdm(range(num_set)):
        print('\n', f'Dataset {i}:')
        
        if isinstance(config[judge], dict) :
            best_config = config[f'set{i}']
        else :
            best_config = config
            
        # seperate the decision tree hyperparameter and adaboost hyperparameter
        tree_param = {'base_estimator': DecisionTreeClassifier(max_depth = best_config['max_depth'])}
        boost_param = dict((key, best_config[key]) for key in ['learning_rate', 'n_estimators'] if key in best_config)
        boost_param.update(tree_param)

        result = AdaBoostC(trainset_x[f'set{i}'], test_x, trainset_y[f'set{i}'], test_y, boost_param)
        table = cf_matrix(result, trainset_y[f'set{i}'])
        table_set = pd.concat([table_set, table]).rename(index = {0: f'dataset {i}'})
    
    return table_set


def runall_AdaBoostR(num_set, trainset_x, test_x, trainset_y, test_y, config, thres_target = 'Recall', threshold = False):
    
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
        
    return pr_dict, table_set


# ### optuna

# In[ ]:


def AdaBoost_creator(train_data, mode, num_valid = 3):
    
    def objective(trial) :

        tree_param = {
            'max_depth': trial.suggest_int('max_depth', 1, 3)
        }
        
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step = 50),
            'learning_rate': trial.suggest_float('learning_rate', 0.025, 0.825, step = 0.05),
        }
        if mode == 'C':
            base = {'base_estimator': DecisionTreeClassifier(**tree_param)}
        elif mode == 'R':
            base = {'base_estimator': DecisionTreeRegressor(**tree_param)}
        param.update(base)


        result_list = []
        for i in range(num_valid):

            train_x, train_y = label_divide(train_data, None, 'GB', train_only = True)
            train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size = 0.25)

            if mode == 'C':
                result = AdaBoostC(train_x, valid_x, train_y, valid_y, param)
                table = cf_matrix(result, valid_y)
                recall = table['Recall']
                aging = table['Aging Rate']
                effi = table['Efficiency']
                result_list.append(recall - 0.1*aging)
                
            elif mode == 'R':
                result = AdaBoostR(train_x, valid_x, train_y, valid_y, param)
                pr_matrix = PR_matrix(result, valid_y)
                auc = AUC(pr_matrix['Recall'], pr_matrix['Aging Rate'])
                result_list.append((-1)*auc)

        return np.mean(result_list)
    
    return objective


def all_optuna(num_set, all_data, mode, TPE_multi, n_iter, filename, creator, num_valid = 3, return_addition = True):

    best_param = {}
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
            all_score[f'set{i}'] = collect_score 
        
        print(f"Sampler is {study.sampler.__class__.__name__}")
    
    ##### store the best hyperparameters #####
    multi_mode = 'multivariate-TPE' if TPE_multi else 'univariate-TPE'
    with open(f'{filename}{mode}_{multi_mode}_{n_iter}.data', 'wb') as f:
        pickle.dump(best_param, f)
    
    if return_addition :
        return best_param, all_score
    else :
        return best_param
    

def optuna_history(best_param, all_score, num_row, num_col, model):

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


# ### search for best hyperparameter

# In[ ]:


best_paramC, all_scoreC = all_optuna(num_set = 10, 
                                     all_data = run_train, 
                                     mode = 'C', 
                                     TPE_multi = False, 
                                     n_iter = 1, 
                                     filename = 'runhist_array_m2m5_4selection_AdaBoost',
                                     creator = AdaBoost_creator
                                    )


# In[ ]:


best_paramR, all_scoreR = all_optuna(num_set = 10, 
                                     all_data = run_train, 
                                     mode = 'R', 
                                     TPE_multi = False, 
                                     n_iter = 1,
                                     filename = 'runhist_array_m2m5_4selection_AdaBoost',
                                     creator = AdaBoost_creator
                                    )


# In[ ]:


##### optimization history plot #####
optuna_history(best_paramC, all_scoreC, model = 'XGBoost Classifier')
            
##### best hyperparameter table #####
param_table = pd.DataFrame(best_paramC).T
param_table


# In[ ]:


study = optuna.create_study(sampler = optuna.samplers.TPESampler(multivariate = False), direction = 'maximize') 
#TPE, Random, Grid, CmaEs#
objective = objective_creator(train_data = data_dict['set6'], mode = 'C', num_valid = 3)
study.optimize(objective, n_trials = 5, show_progress_bar = True, gc_after_trial = True)


##### hyperparameter importance #####
#importances = optuna.importance.get_param_importances(study)
#importances.optuna.importance.get_param_importances(study, evaluator = optuna.importance.FanovaImportanceEvaluator())
importance_fig = optuna.visualization.plot_param_importances(study)
slice_fig = optuna.visualization.plot_slice(study)
importance_fig.show()
slice_fig.show()


# ## 

# ### classifier

# In[ ]:


table_setC = runall_AdaBoostC(10, run_train_x, run_test_x, run_train_y, run_test_y, best_paramC)
line_chart(table_set, title = 'AdaBoost Classifier')


# In[ ]:


table_setC


# ### regressor

# In[ ]:


pr_dict, table_setR = runall_AdaBoostR(10, run_train_x, run_test_x, run_train_y, run_test_y, best_paramR,
                                      thres_target = 'Recall', threshold = 0.8)
line_chart(table_setR, title = 'AdaBoost Regressor')


# In[ ]:


multiple_curve(4, 3, pr_dict, table_setR, target = 'Aging Rate')
multiple_curve(4, 3, pr_dict, table_setR, target = 'Precision')
table_setR
'''
