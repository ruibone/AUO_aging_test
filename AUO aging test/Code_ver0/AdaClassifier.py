#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pickle

from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier
import optuna
from sklearn.model_selection import train_test_split

from Dataset_Construction import Balance_Ratio
from Sampling import label_divide
from Aging_Score import score1
'''
os.chdir('C:/Users/user/Desktop/Darui_R08621110')  
os.getcwd()
'''

# ### Load multiple dataset

# In[ ]:


def multiple_set(num_set):
    
    data_dict = {}
    for i in range(num_set):
        data_dict[f'set{i}'] = pd.read_csv(f'dataset_{i}.csv').iloc[:, 1:]
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


# ### Boosting Model

# In[ ]:


def AdaBoostC(train_x, test_x, train_y, test_y, config):
    
    clf = AdaBoostClassifier(**config)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result


# ### Recall & Precision for Classifier

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
    
    table = pd.Series({'Balance Ratio': br, 'Train_OK': train_OK, 'Train_NG': train_NG, 'TP': TP, 'FP': FP, 'FN': FN,                        'TN': TN, 'Precision': precision, 'Recall': recall, 'Aging Rate': ar, 'Efficiency': eff, 'Score': score})
    table = pd.DataFrame(table).T
    
    print('Precision:', precision, '\nRecall:', recall, '\nAging Rate:', ar)
    return  table


def print_badC(predict, test_x, Bad_Types, threshold = 1):
    
    Bad = []
    Bad_miss = []
    TP = predict[(predict['truth'] == 1) & (predict['predict'] >= threshold)].index
    FN = predict[(predict['truth'] == 1) & (predict['predict'] < threshold)].index
    for j in range(len(TP)):
        Index = TP[j]
        Key = test_x.values[Index]
        Key = pd.DataFrame(Key).T.apply(lambda x:'_'.join(x.astype(str)), axis = 1)
        Bad.append(Bad_Types[Key[0]])
        Bad.sort()
    print('Types of Bad found:', Bad) 
    
    for j in range(len(FN)):
        Index = FN[j]
        Key = test_x.values[Index]
        Key = pd.DataFrame(Key).T.apply(lambda x:'_'.join(x.astype(str)),axis=1)
        Bad_miss.append(Bad_Types[Key[0]])
        Bad_miss.sort()
    print('Types of Bad not found:', Bad_miss)
    
    bad_table = pd.Series({'Bad_Found': set(Bad), 'Bad_Missed': set(Bad_miss)})
    bad_table = pd.DataFrame(bad_table).T
    bad_table['Detect Ratio'] = len(Bad) / (len(Bad) + len(Bad_miss))
    
    return bad_table


# ### Run all dataset

# In[ ]:


def runall_AdaBoostC(num_set, trainset_x, test_x, trainset_y, test_y, config, record_bad = True):
    
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
        
        if record_bad:
            bad_table = print_badC(result, test_x, Bad_Types) 
            bad_set = pd.concat([bad_set, bad_table]).rename(index = {0: f'dataset {i}'})

    if record_bad:
        return table_set, bad_set
    else:
        return table_set


# ### Plot all dataset

# In[ ]:


def bad_plot(bad_set):
    
    # record all bad types
    bad_list = []
    [bad_list.append(x) for x in bad_set.loc['dataset 1'][0]]
    [bad_list.append(x) for x in bad_set.loc['dataset 1'][1]]
    bad_list.sort()
    
    bad_array = np.empty([len(bad_set), len(bad_list)])
    for j in range(len(bad_set)):
        for i in range(len(bad_list)):
            if bad_list[i] in bad_set.iloc[j, 0]:
                bad_array[j, i] = 1
            else:
                bad_array[j ,i] = 0
                          
    bad_df = pd.DataFrame(bad_array)
    bad_df.columns = bad_list
    
    plt.pcolor(bad_df, cmap = 'Reds')
    plt.title("Bad Types Detection across All Datasets")
    plt.yticks(np.arange(0.5, len(bad_df.index), 1), bad_df.index)
    plt.xticks(np.arange(0.5, len(bad_df.columns), 1), bad_df.columns.astype(int))
    plt.xlabel("ID of Bad Types", size = 12)
    plt.ylabel("Dataset", size = 12)
    
    plt.savefig('Bad Types Detection across All Datasets.jpg')
    plt.show()
    
    
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
    
    #plt.savefig(f'{title}.jpg')
    plt.show()

'''
# ## Data Processing
# 

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


#table_set, bad_set = runall_AdaBoostC(9, trainset_x, test_x, trainset_y, test_y)
table_set = runall_AdaBoostC(10, trainset_x, run_test_x, trainset_y, run_test_y, best_paramC, record_bad = False)
line_chart(table_set, title = 'AdaBoost Classifier')
#bad_plot(bad_set)


# In[ ]:


table_set


# ## Optimization

# ### Optuna

# In[ ]:


def objective_creator(train_data, mode, num_valid = 3) :
    
    def objective(trial) :

        tree_param = {
            'max_depth': trial.suggest_int('max_depth', 1, 3)
        }
        
        param = {
            'base_estimator': DecisionTreeClassifier(**tree_param),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step = 50),
            'learning_rate': trial.suggest_float('learning_rate', 0.025, 0.825, step = 0.05),
        }


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

                #result_list.append(effi)
                result_list.append(recall - 0.1*aging)

        return np.mean(result_list)
    
    return objective


# In[ ]:


best_paramC, all_scoreC = all_optuna(num_set = 10, 
                                     all_data = data_dict, 
                                     mode = 'C', 
                                     TPE_multi = True, 
                                     n_iter = 25, 
                                     filename = 'runhist_array_m2m5_4selection_AdaBoost',
                                     creator = objective_creator
                                    )


# In[ ]:


##### optimization history plot #####
optuna_history(best_paramC, all_scoreC, model = 'AdaBoost Classifier')
            
##### best hyperparameter table #####
param_table = pd.DataFrame(best_paramC).T
param_table
'''
