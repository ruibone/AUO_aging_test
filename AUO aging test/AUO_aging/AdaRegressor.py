#!/usr/bin/env python
# coding: utf-8

# In[151]:


import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostRegressor

from Dataset_Construction import Balance_Ratio 
from Sampling import label_divide
from AdaClassifier import train_set, multiple_set, print_badC, bad_plot, line_chart
from Aging_Score import score1
'''
os.chdir('C:/Users/Darui Yen/OneDrive/桌面/data_after_mid') 
os.getcwd()
'''

# ### Boosting Model

# In[2]:


def AdaBoostR(train_x, test_x, train_y, test_y, n_estimator = 100, LR = 0.7):
    
    clf = AdaBoostRegressor(n_estimators = n_estimator, learning_rate = LR, random_state = None)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result


# ### Recall & Precision for Regressor

# In[252]:


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
    
    matrix['Recall'] = matrix['TP'] / (matrix['TP'] + matrix['FN'])
    matrix['Precision'] = matrix['TP'] / (matrix['TP'] + matrix['FP'])
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

# In[250]:


def runall_AdaBoostR(num_set, trainset_x, test_x, trainset_y, test_y, thres_target = 'Recall', threshold = False, 
                     record_bad = True):
    
    table_set = pd.DataFrame()
    bad_set = pd.DataFrame()
    pr_dict = {}

    for i in range(num_set):
        print('\n', f'Dataset {i}:')

        predict = AdaBoostR(trainset_x[f'set{i}'], test_x, trainset_y[f'set{i}'], test_y)
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

# In[377]:


def AUC(x, y):
    
    area=0
    left=x[0]*y[0]
    right=(1-x[len(x)-1])*y[len(x)-1]
    
    for i in range(1,len(x)):
        wide=x[i]-x[i-1]
        area=area+wide*((y[i-1]+y[i])/2)
    area=left+area+right  
    
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
    plt.suptitle(f'{target} & Recall Curve of Dataset 0 - {len()}', y = 0.94, fontsize = 30)
    
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
                    axs[row, col].set_title(f'dataset {index},   AUC = {auc},   Aging Rate = {ar},   Recall = {recall}, \
                                            Precision = {precision}')
                elif target == 'Precision':
                    axs[row, col].set_title(f'dataset {index},       AUC = {auc},       Aging Rate = {ar},       Recall = {recall}')
'''
# ## Data Processing

# In[5]:


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


# In[254]:


start = time.time()

pr_dict, table_set, bad_set = runall_AdaBoostR(9, trainset_x, test_x, trainset_y, test_y, 
                                               thres_target = 'Recall', threshold = 0.8)

bad_plot(bad_set)

end = time.time()
print("\nRun Time：%f seconds" % (end - start))


# In[255]:


line_chart(table_set, title = 'AdaBoost Regressor')


# In[336]:


table_set


# In[337]:


bad_set


# In[378]:


multiple_curve(3, 3, pr_dict, table_set)
'''
