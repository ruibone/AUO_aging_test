#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import pandas as pd

from library.Data_Preprocessing import Balance_Ratio 
'''
os.chdir('C:/Users/user/Desktop/Darui_R08621110') 
os.getcwd()
'''

# ## 

# ### identfy all kinds of dimensional overlapping data

# In[4]:


def Corner(D, label = 'GB'):
    
    Data = D.copy()
    Data['cb'] = Data[Data.columns[1:-1]].apply(lambda x: '_'.join(x.astype(str)), axis = 1)
    Data[f'{label}_mean'] = Data.groupby('cb')[label].transform('mean')
    Data[f'{label}_count'] = Data.groupby('cb')[label].transform('count')
    Corner_overlap = Data.copy() 
    
    return Corner_overlap


def Kind(Corner_overlap, label = 'GB'):

    Data = Corner_overlap
    
    Kinds_0 = Data[Data[f'{label}_mean'] == 0].sort_values(f'{label}_count') # represent all data in the same cb are good  
    Kinds_1 = Data[Data[label] == 1].sort_values(f'{label}_count') # represent bad data
    Kinds = pd.concat([Kinds_1, Kinds_0]) # reprecent all types of 'cb'
    Kinds_after_duplicate = Kinds.drop_duplicates(subset = ['cb']).reset_index(drop = True) 
    # each 'cb' only keep one data(the first)
    
    return Kinds_after_duplicate


def Dictionary_Build(Data):
    
    Corner_overlap = Corner(Data)
    Kinds = Kind(Corner_overlap).copy()

    ## reorder cols
    cols = Kinds.columns.tolist()
    cols = cols[0:1] + cols[-4:] 
    Dictionary = Kinds[cols] #select id, GB, cb, GB_mean, GB_count
    Dictionary['G_count'] = 0 # add new column 
    Dictionary=Dictionary.reset_index(drop=True)
    for i in range(len(Dictionary)):
        Dictionary['G_count'][i] = Dictionary.GB_count[i] - Dictionary.GB[i] # represent??
    
    return Dictionary


# ### relabel overlapping data

# In[5]:


def Remove_SD(Data, count = 1, label = 'GB'):
    
    Corner_Overlap = Data
    RSD = pd.concat([Data[Data[label] == 1], Data[(Data[label] == 0) & (Data[f'{label}_count'] > count)]]) 
    # remove good data that only show once
    return RSD


def Corner_Blend(Data, ratio = 0.002, label = 'GB'):
    
    D_1 = Data[(Data[f'{label}_mean'] <= 1) & (Data[f'{label}_mean'] >= ratio)].sort_values(f'{label}_count') # not relabel case
    D_1[[label]] = 1  #if G_mean > ratio ==> relabel all data as bad
    D_2 = Data[(Data[f'{label}_mean'] < ratio)].sort_values(f'{label}_count')
    D_2[[label]] = 0  # otherwise relabel all data as good
    Training_new = pd.concat([D_1,D_2]).iloc[:,:-3]
    
    return Training_new



# ## 

# ### loading training data

# In[32]:

'''
##### training data of each month #####
training_month = [2, 3, 4]
train_runhist = {}
for i in training_month:
    train_runhist[f'm{i}'] = pd.read_csv(f'train_runhist_m{i}.csv').iloc[:, 1:]
    print(f'Dimension of month {i}:', train_runhist[f'm{i}'].shape)

##### training & testing data #####
train_runhist['all'] = pd.read_csv('train_runhist.csv').iloc[:, 1:]
test_runhist = pd.read_csv('test_runhist.csv').iloc[:, 1:]
print('\nDimension of training data:', train_runhist['all'].shape,
      '\nDimension of testing data:', test_runhist.shape)


# ### relabel the trraining data by month

# In[42]:


##### training data of each month #####
training_month = [2, 3, 4]

overlap = {}
for i in training_month:
    overlap[f'm{i}'] = Corner(train_runhist[f'm{i}'])
    new_runhist = Corner_Blend(overlap[f'm{i}'], 1/10000)
    new_runhist.to_csv(f'relabel_runhist_m{i}.csv')
    print('Month {i} after overlap relabel:', new_runhist.shape, 'balance ratio:', Balance_Ratio(new_runhist), 
          '# total bad:', sum(new_runhist.GB))
    
##### the whole training data & testing data #####
train_overlap = Corner(train_runhist['all'])
new_train = Corner_Blend(train_overlap, 1/10000)
new_train.to_csv('relabel_runhist.csv')
print('\nAll training data after overlap relabel:', new_train.shape, ', balance ratio:', Balance_Ratio(new_train), 
          ', # total bad:', sum(new_train.GB))

test_overlap = Corner(test_runhist)
new_test = Corner_Blend(test_overlap, 1/10000)
all_overlap = pd.concat([train_overlap, test_overlap], axis = 0)
new_all = pd.concat([new_train, new_test], axis = 0)

print('All testing data after overlap relabel:', new_test.shape, ', balance ratio:', Balance_Ratio(new_test), 
          ', # total bad:', sum(new_test.GB))
print('All runhist data after overlap relabel:', new_all.shape, ', balance ratio:', Balance_Ratio(new_all), 
          ', # total bad:', sum(new_all.GB))


# ### cauculate the number of combination types 

# In[43]:


##### training data of each month #####
kinds = {}
for i in training_month:
    kinds[f'm{i}'] = Kind(overlap[f'm{i}'])
    kinds[f'm{i}'].to_csv(f'kind_m{i}.csv')
    print('Number of kinds in month {i}:', len(kinds[f'm{i}']))

##### the whole training data & testing data #####
train_kinds = Kind(train_overlap)
train_kinds.to_csv('kind.csv')
print('\nNumber of kinds in all training data:', len(train_kinds))

test_kinds = Kind(test_overlap)
all_kinds = Kind(all_overlap)

print('Number of kinds in all testing data:', len(test_kinds))
print('Number of kinds in all runhist data:', len(all_kinds))
'''
