#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import pyreadr
import numpy as np
import pandas as pd

#import smote_variants as sv
from imblearn import FunctionSampler
from imblearn.under_sampling import RandomUnderSampler

from Dataset_Construction import Balance_Ratio
'''
os.chdir('C:/Users/Darui Yen/OneDrive/桌面/data_after_mid')
os.getcwd()
'''

# In[8]:


def label_divide(train, test, label = 'GB', train_only = False):
    
    train_x = train.drop(columns = label)
    train_y = train[label]
    
    if not train_only:
        test_x = test.drop(columns = label)
        test_y = test[label]    
        return train_x, train_y, test_x, test_y
    else:
        return train_x, train_y


# ### Oversampling 

# In[9]:


def before_over(dataset, label = 'GB'):
    
    colnames = dataset.columns
    Y = dataset[label]
    Y = Y.reset_index(drop = True)
    Y = np.array(Y)
    X = dataset.drop(columns = [label])
    X = X.reset_index(drop = True)
    X = X.to_numpy()
    
    return X, Y, colnames


def after_over(X, Y, colnames):
    
    colnames = colnames[:X.shape[1]]
    X = pd.DataFrame(X, columns = colnames)
    Y = pd.Series(Y)
    
    return X, Y


def over_sample(X, Y, method, proportion = 0.5, n_neighbors = 7, *args):
    
    method_list = ['NoSMOTE', 'SMOTE', 'MSMOTE', 'ROSE']
    if method not in method_list:
        raise Exception('Invalid method !')
    
    if method == method_list[0]:
        over_sampler = sv.NoSMOTE()
    elif method == method_list[1]:
        over_sampler = sv.SMOTE(proportion, n_neighbors)
    elif method == method_list[2]:
        over_sampler = sv.MSMOTE(proportion, n_neighbors)
    elif method == method_list[3]:
        over_sampler = sv.ROSE(proportion)
        
    over_X, over_Y = over_sampler.sample(X, Y)
    
    return over_X, over_Y


# ### Undersampling

# In[10]:


def before_under(dataset, label = 'GB'):
    
    Y = dataset[label]
    X = dataset.drop(columns = [label])
    
    return X, Y


def under_sample(X, Y, method, *args):
    
    method_list = [None, 'random', 'Tomek']
    if method not in method_list:
        raise Exception('Invalid method !')
    
    if method == method_list[0]:
        return X, Y
        
    elif method == method_list[1]:
        undersampler = RandomUnderSampler(sampling_strategy = 'majority', random_state = None)
        
    elif method == method_list[2]:
        undersampler = TomekLinks(sampling_strategy = 'majority')
    
    under_X, under_Y = undersampler.fit_resample(X, Y)
    return under_X, under_Y


def over_under(dataset, over_method, under_method, *args, label = 'GB'):
    
    if over_method != None:
        temp_X, temp_Y, colnames = before_over(dataset, label)
        over_X, over_Y = over_sample(temp_X, temp_Y, over_method)
        X, Y = after_over(over_X, over_Y, colnames)
    else:
        X, Y = before_under(dataset, label)
    
    if under_method != None:
        under_X, under_Y = under_sample(X, Y, under_method)
        return under_X, under_Y
    else:
        return X, Y


# ### Generate multiple dataset file

# In[6]:


def generate_set(train_data, over_method, under_method, index, label = 'GB'):
    
    train_x, train_y = over_under(train_data, over_method, under_method, label)
    train = pd.concat([train_x, train_y], axis = 1)
    train = train.rename(columns = {0: label})
    train.to_csv(f'dataset_{index}.csv')
    
    return train

'''
# ### Data processing
# 

# In[3]:


panel = pd.read_csv('original_data/TrainingSet_new.csv', index_col = 'id').iloc[:, 1:]
print('Dimension:', panel.shape, '\nBalance Ratio:', Balance_Ratio(panel))


# In[150]:


train_x, train_y = over_under(panel, 'GB', None, None)
train = pd.concat([train_x, train_y], axis = 1)
train = train.rename(columns = {0: 'GB'})
train.to_csv('Train_sample.csv')

print('Dimension:', '\ntrain x:', train_x.shape, '\ntrain y:', train_y.shape, '\nBalance Ratio:', Balance_Ratio(train))


# In[ ]:


start = time.time()

generate_set(panel, 'NoSMOTE', None, 0)

generate_set(panel, 'SMOTE', None, 1)
generate_set(panel, 'MSMOTE', None, 2)
generate_set(panel, 'ROSE', None, 3)

generate_set(panel, 'SMOTE', 'random', 4)
generate_set(panel, 'MSMOTE', 'random', 5)
generate_set(panel, 'ROSE', 'random', 6)

end = time.time()
print("\nRun Time：%f seconds" % (end - start))

'''