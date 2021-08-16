#!/usr/bin/env python
# coding: utf-8

# In[45]:


import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.ensemble import AdaBoostClassifier

from Dataset_Construction import Balance_Ratio
from Sampling import label_divide
from Aging_Score import score1

#os.chdir('C:/Users/Darui Yen/OneDrive/桌面/data_after_mid') 
#os.getcwd()


# ### Load multiple dataset

# In[46]:


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

# In[68]:


def AdaBoostC(train_x, test_x, train_y, test_y, n_estimator = 100, LR = 0.7):
    
    clf = AdaBoostClassifier(n_estimators = n_estimator, learning_rate = LR)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    result = pd.DataFrame({'truth': test_y, 'predict': predict_y})
    
    return result


# ### Recall & Precision for Classifier

# In[62]:


def cf_matrix(predict, train_y):
    
    #confusion matrix
#     cf = np.zeros([2,2])
#     for i in range(len(predict)):
#         if predict['predict'].values[i] == 0:
#             row_index = 1
#         else:
#             row_index = 0
#         if predict['truth'].values[i] == 0:
#             col_index = 1
#         else:
#             col_index = 0
#         cf[row_index][col_index] += 1
#     cf = cf.astype(int)
    
#     TP, FP, FN, TN = cf[0,0], cf[0,1], cf[1,0], cf[1,1]

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

# In[50]:


def runall_AdaBoostC(num_set, trainset_x, test_x, trainset_y, test_y, record_bad = True):
    
    table_set = pd.DataFrame()
    bad_set = pd.DataFrame()

    for i in tqdm(range(num_set)):
        print('\n', f'Dataset {i}:')

        result = AdaBoostC(trainset_x[f'set{i}'], test_x, trainset_y[f'set{i}'], test_y)
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

# In[58]:


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
    ax1.set_xlabel('\nDataSet', fontsize = 12)
    ax1.set_ylabel('Recall & Aging rate', color = 'b')
    ax2.set_ylabel('Precision', color = 'g')
    
    ax1.legend(loc = 'upper left', frameon = False)
    ax2.legend(loc = 'upper right', frameon = False)
    
    #plt.savefig(f'{title}.jpg')
    plt.show()

'''
# ## Data Processing
# 

# In[53]:


###bad types###
bad = pd.read_csv('original_data/Bad_Types.csv').iloc[:, 1:]
Bad_Types = {bad.cb[i]:i for i in range (len(bad))}
print('Total bad types:', len(bad))

###single dataset###
test = pd.read_csv('original_data/TestingSet_0.csv').iloc[:, 2:]
#train = pd.read_csv('data_from_newpy/Train_sample.csv').iloc[:, 1:]
#print('\ntraining data:', train.shape, '\nBalance Ratio:', Balance_Ratio(train))
print('\ntesting data:', test.shape, '\nBalance Ratio:', Balance_Ratio(test))

#train_x, train_y, test_x, test_y = label_divide(train, test, 'GB')

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


# In[67]:


start = time.time()

#table_set, bad_set = runall_AdaBoostC(9, trainset_x, test_x, trainset_y, test_y)
table_set = runall_AdaBoostC(9, trainset_x, run_test_x, trainset_y, run_test_y, record_bad = False)
line_chart(table_set, title = 'AdaBoost Classifier')

end = time.time()
print("\nRun Time：%f seconds" % (end - start))


# In[64]:


table_set


# In[65]:


bad_plot(bad_set)
bad_set
'''
