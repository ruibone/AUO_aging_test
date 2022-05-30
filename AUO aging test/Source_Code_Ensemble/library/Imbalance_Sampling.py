#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import math
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# import smote_variants as sv
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, InstanceHardnessThreshold, NearMiss
from imblearn.over_sampling import ADASYN, SMOTEN, RandomOverSampler

from library.Data_Preprocessing import Balance_Ratio, training_def
from library.Training_Data_Processing import Corner, Kind
'''
os.chdir('C:/Users/user/Desktop/Darui_R08621110')
os.getcwd()
'''

# ## 

# In[2]:


# seperate a dataset into X & Y
def label_divide(train, test, label = 'GB', train_only = False):
    
    train_x = train.drop(columns = label)
    train_y = train[label]
    
    if not train_only:
        test_x = test.drop(columns = label)
        test_y = test[label]    
        return train_x, train_y, test_x, test_y
    else:
        return train_x, train_y


# ### Self-Defined Oversampling (Modified Border)
# first writen by ChungCheng Huang, and then modified 

# In[10]:


# distance between instances
def distance_matrix(data1, data2, triangle = False):
    
    data1 = np.array(data1.iloc[:, :-1])
    data2 = np.array(data2.iloc[:, :-1])
    dis_mat = pd.DataFrame((data1[:, None, :] != data2).sum(2))
    if triangle:
        dis_mat = dis_mat.where(np.triu(np.ones(dis_mat.shape)).astype(bool))
    
    return dis_mat


# find the (row, col) given the dataframe & distance
def get_indexes(dis_mat, value):

    pos_list = []
    # Get bool dataframe with True at positions where the given value exists
    result = dis_mat.isin([value])
    # Get list of columns that contains the value
    col_target = result.any()
    colnames = list(col_target[col_target == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in colnames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            pos_list.append((row, col))
    
    return pos_list


# smote between two given bad instances
def perm(point_smote, cols_diff, num_over, farthest_generate = 3):
    
    generate_df = pd.DataFrame()
    for i in range(num_over): # synthesize a new instances in every iteration
        new_data = point_smote.copy()
        change_num = random.sample(range(1, farthest_generate+1), 1)[0]
        diff_index = cols_diff[cols_diff == True].index.tolist()
        change_index = random.sample(diff_index, change_num)
        for j in change_index: # change the index randomly selected based on central instance
            new_data[j] = 1 if point_smote[j] == 0 else 0
        new_df = pd.DataFrame(new_data).T
        generate_df = pd.concat([generate_df, new_df], axis = 0)
            
    return generate_df


# modified-border main function
def Border(data, kind, max_distance, num_over, over_ratio = 1):
    
    good_num = len(data[data.GB == 0])
    bad_num = len(data[data.GB == 1])
    bad_kind = kind[kind.GB == 1]
    full_kind = kind.iloc[:, :-1].copy()
    training_df = pd.DataFrame()
    
    bad_dis = distance_matrix(bad_kind, bad_kind) # calculate the distance between bad instances
    for dis in range(1, max_distance+1):
        print(f'Distance = {dis} ...')
        done = False
        bad_indexes = get_indexes(bad_dis, dis) # given the specific distance and find the pair of bad instances
        
        smote_df = pd.DataFrame()
        if len(bad_indexes) != 0:   
            total_num = 0
            for pair in bad_indexes:
                point_0, point_1 = pair
                point_smote = full_kind.loc[point_0].copy() # let point_0 be the initially central point of synthetic data
                cols_diff = (full_kind.loc[point_0] != full_kind.loc[point_1]) # find the different cols between two bad
                perm_df = perm(point_smote, cols_diff, int(num_over/2)) # generate new instances
                smote_df = pd.concat([smote_df, perm_df], axis = 0)
                total_num += len(perm_df)
                
                if (total_num + len(training_df) + bad_num) >= good_num*over_ratio: # synthetic bad instances are enough
                    print(f'# over: {total_num}')
                    done = True
                    break
            print(f'# over: {total_num}')
                
        training_df = pd.concat([training_df, smote_df], axis = 0)
        training_df = training_df.drop_duplicates().reset_index(drop = True)
        if done:
            break
    training_df['GB'] = 1
    
    return training_df


# ### Oversampling

# In[19]:


# oversampling preparation
def before_over(dataset, label = 'GB'):
    
    colnames = dataset.columns
    Y = dataset[label]
    Y = Y.reset_index(drop = True)
    Y = np.array(Y)
    X = dataset.drop(columns = [label])
    X = X.reset_index(drop = True)
    X = X.to_numpy()
    
    return X, Y, colnames


# processing data afer oversampling
def after_over(X, Y, colnames, back_to_category = False):
    
    colnames = colnames[:X.shape[1]]
    X = pd.DataFrame(X, columns = colnames)
    
    if back_to_category:
        for j in tqdm(range(X.shape[1])):
            colvalue = X.iloc[:, j]
            upper = np.array(colvalue[colvalue < 1])
            lower = np.array(upper[upper > 0])
            colmean = np.mean(lower)
            
            mask = colvalue >= colmean
            X.iloc[mask, j] = 1
            X.iloc[~mask, j] = 0
    
    Y = pd.Series(Y)
    
    return X, Y


# apply oversampling methods
def over_sample(X, Y, method, ratio, n_neighbors = 5, *args):
    
    method_list = ['NoSMOTE', 'SMOTE', 'MSMOTE', 'ROSE', 'SMOTEN', 'ADASYN']
    if method not in method_list:
        raise Exception('Invalid method !')
    
    if method == method_list[0]:
        over_sampler = sv.NoSMOTE()
    elif method == method_list[1]:
        over_sampler = sv.SMOTE(ratio, n_neighbors)
    elif method == method_list[2]:
        over_sampler = sv.MSMOTE(ratio, n_neighbors)
    elif method == method_list[3]:
        over_sampler = RandomOverSampler(sampling_strategy = ratio, shrinkage = 2.5)  
    elif method == method_list[4]:
        over_sampler = SMOTEN(sampling_strategy = ratio, k_neighbors = n_neighbors)
    elif method == method_list[5]:
        over_sampler = ADASYN(sampling_strategy = ratio, n_neighbors = n_neighbors)    
    
    if method in method_list[0:3]:
        over_X, over_Y = over_sampler.sample(X, Y)
    else:
        over_X, over_Y = over_sampler.fit_resample(X, Y)
    
    return over_X, over_Y


# ### Undersampling

# In[12]:


# undersampling preparation
def before_under(dataset, label = 'GB'):
    
    Y = dataset[label]
    X = dataset.drop(columns = [label])
    
    return X, Y


# apply undersampling methods
def under_sample(X, Y, method, ratio, *args):
    
    method_list = [None, 'random', 'Tomek', 'IHT', 'NM', 'one-sided', 'r-one-sided']
    if method not in method_list:
        raise Exception('Invalid method !')
    
    if method == method_list[0]:
        return X, Y
        
    elif method == method_list[1]:
        under_sampler = RandomUnderSampler(sampling_strategy = ratio)    
    elif method == method_list[2]:
        under_sampler = TomekLinks(sampling_strategy = 'majority')
    elif method == method_list[3]:
        under_sampler = InstanceHardnessThreshold(sampling_strategy = ratio, cv = 5, n_jobs = -1)
    elif method in (method_list[4] + method_list[5]):
        under_sampler = NearMiss(sampling_strategy = ratio, version = 2, n_jobs = -1)
    elif method == method_list[6]:
        under_sampler = InstanceHardnessThreshold(sampling_strategy = 1, cv = 5, n_jobs = -1)
    
    under_X, under_Y = under_sampler.fit_resample(X, Y)
    
    if method == method_list[5]:
        second_sampler = InstanceHardnessThreshold(sampling_strategy = 1, cv = 5, n_jobs = -1)
        under_X, under_Y = second_sampler.fit_resample(under_X, under_Y)
    elif method == method_list[6]:
        second_sampler = NearMiss(sampling_strategy = ratio, version = 2, n_jobs = -1)
        under_X, under_Y = second_sampler.fit_resample(under_X, under_Y)
    
    return under_X, under_Y


# ### Protocol to Generate Datasets

# In[13]:


# resampling combination (undersampling first) 
def under_over(dataset, over_method, under_method, over_ratio, under_ratio, label = 'GB'):
    
    # undersampling
    if under_method != None:
        X, Y = before_under(dataset, label)
        Y = Y.astype(int)
        print('Size before Undersampling:', len(Y))
        under_X, under_Y = under_sample(X, Y, under_method, under_ratio)
        dataset = pd.concat([under_X, under_Y], axis = 1)
        print('Size after Undersampling:', len(under_Y))
    
    # oversampling
    temp_X, temp_Y, colnames = before_over(dataset, label)
    print('Size before Oversampling:', len(temp_Y))
    over_X, over_Y = over_sample(temp_X, temp_Y, over_method, over_ratio)
    X, Y = after_over(over_X, over_Y, colnames)
    print('Size after Oversampling:', len(Y))
        
    return X, Y


# resampling combination (oversampling first)
def over_under(dataset, over_method, under_method, over_ratio, under_ratio, label = 'GB') :
    
    # oversampling
    if over_method != None :
        X, Y, colnames = before_over(dataset, label)
        print('Size before Oversampling:', len(Y))
        temp_X, temp_Y = over_sample(X, Y, over_method, over_ratio)
        over_X, over_Y = after_over(temp_X, temp_Y, colnames)
        print('Size after Oversampling:', len(over_Y))
        over_dataset = pd.concat([over_X, over_Y], axis = 1)
        dataset = over_dataset.rename(columns = {0 : label})

    # undersampling
    X, Y = before_under(dataset, label)
    Y = Y.astype(int)
    under_X, under_Y = under_sample(X, Y, under_method, under_ratio)
    print('Size after Undersampling:', len(under_Y))
    
    return under_X, under_Y
    
# main function to generating a resampling dataset
def generate_set(train_data, over_method, under_method, index, over_ratio, under_ratio, order, label = 'GB'):
    
    print('\n', f'Generating Dataset {index}')
    if order == 'under' :
        train_x, train_y = under_over(train_data, over_method, under_method, over_ratio, under_ratio, label)
    elif order == 'over' :
        train_x, train_y = over_under(train_data, over_method, under_method, over_ratio, under_ratio, label)
        
    train = pd.concat([train_x, train_y], axis = 1)
    train = train.rename(columns = {0: label})
    
    return train


# main function to generate a resampling dataset with border and undersampling technique
def border_set(train_data, kind_data, under_method, index, num_over, over_ratio, under_ratio, order):
    
    ##### oversampling first #####
    if order == 'over':
        print('Size before Border:', len(train_data))    
        OS_B = Border(train_data, kind_data, 25, num_over, over_ratio = over_ratio)
        self_runhist = pd.concat([train_data, OS_B], axis = 0).reset_index(drop = True)
        print('Size after Border:', len(self_runhist))
        
        dataset = generate_set(self_runhist, None, under_method, index, over_ratio = None, under_ratio = under_ratio, 
                               order = 'over')
        print(f'Size after Undersampling:', dataset.shape, ', Balance Ratio:', Balance_Ratio(dataset))
        
        return dataset
    
    ##### undersampling first #####
    elif order == 'under':
        print('Size before Undersampling:', len(train_data))
        self_under = generate_set(train_data, None, under_method, index, over_ratio = None, under_ratio = under_ratio, 
                                  order = 'over')
        print('Size after Undersampling:', len(self_under))
        
        corner_overlap = Corner(self_under)
        under_kind = Kind(corner_overlap).iloc[:, :-3]
        US_B = Border(self_under, under_kind, 25, num_over, over_ratio = over_ratio)
        dataset = pd.concat([self_under, US_B], axis = 0).reset_index(drop = True)
        print('Size after Border:', dataset.shape, ', Balance Ratio:', Balance_Ratio(dataset))
        
        return dataset


# ### Main Function for Generating 10 Datasets

# In[17]:


##### main function for generating all resampling datasets #####
def resampling_dataset(runhist, kinds, train_month, final_br, num_os):    
    dataset = {}
    combine_dataset = {}
    for i in range(10):
        combine_dataset[i] = pd.DataFrame()

    for i in tqdm(train_month):

        print(f'Month {i}:')
        print('# bad:', sum(runhist[f'm{i}'].GB))
        br = Balance_Ratio(runhist[f'm{i}'])
        over_br = num_os / br
        under_br = final_br / num_os

        # Border-related datasets
        dataset[2] = border_set(runhist[f'm{i}'], kinds[f'm{i}'], 'NM', 2, num_over = num_os, over_ratio = over_br, 
                                under_ratio = final_br, order = 'over')
        dataset[6] = border_set(runhist[f'm{i}'], kinds[f'm{i}'], 'NM', 6, num_over = num_os, over_ratio = final_br, 
                                under_ratio = under_br, order = 'under')
        # original dataset
        dataset[0] = runhist[f'm{i}'].copy()
        # oversampling-first datasets 
        dataset[1] = generate_set(runhist[f'm{i}'], 'ADASYN', 'NM', 1, over_ratio = over_br, under_ratio = final_br, 
                                  order = 'over')
        dataset[3] = generate_set(runhist[f'm{i}'], 'ROSE', 'NM', 3, over_ratio = over_br, under_ratio = final_br,
                                  order = 'over')
        dataset[4] = generate_set(runhist[f'm{i}'], 'SMOTEN', 'NM', 4, over_ratio = over_br, under_ratio = final_br, 
                                  order = 'over')
        # undersampling-first datasets
        dataset[5] = generate_set(runhist[f'm{i}'], 'ADASYN', 'NM', 5, over_ratio = final_br, under_ratio = under_br, 
                                  order = 'under')
        dataset[7] = generate_set(runhist[f'm{i}'], 'ROSE', 'NM', 7, over_ratio = final_br, under_ratio = under_br, 
                                  order = 'under')
        dataset[8] = generate_set(runhist[f'm{i}'], 'SMOTEN', 'NM', 8, over_ratio = final_br, under_ratio = under_br, 
                                  order = 'under')
        # only undersampling
        special = final_br if final_br < 0.1 else 0.1
        dataset[9] = generate_set(runhist[f'm{i}'], None, 'NM', 9, over_ratio = None, under_ratio = special, 
                                  order = 'over')

        ### combine all training data after sampling by each month and save data files ###
        for j in range(10):
            temp_combine = pd.concat([combine_dataset[j], dataset[j]], axis = 0).fillna(0)
            temp_cols = temp_combine.columns.to_list()
            GB_pos = temp_cols.index('GB')
            fine_cols = temp_cols[: GB_pos] + temp_cols[GB_pos+1: ] + temp_cols[GB_pos: GB_pos+1]
            combine_dataset[j] = temp_combine[fine_cols]

            dataset[j].to_csv(f'm{i}_dataset_{j}.csv')
            combine_dataset[j].to_csv(f'dataset_{j}.csv')


# ## 

# ### Loading Relabeled Training Data & Kind

# In[5]:

'''
##### training data #####
training_month = range(2, 5)

runhist = {}
for i in training_month:
    runhist[f'm{i}'] = pd.read_csv(f'relabel_runhist_m{i}.csv', index_col = 'id').iloc[:, 1:]
    print(f'Month {i}:')
    print(f'Dimension:', runhist[f'm{i}'].shape, ', # Bad:', sum(runhist[f'm{i}'].GB))
runhist['all'] = training_def(runhist, training_month)
print('All Runhist Data:\n', 'Dimension of :', runhist['all'].shape, ', # Bad:', sum(runhist['all'].GB), '\n')

##### kind data (for border only) #####
kinds = {}
for i in training_month:
    kinds[f'm{i}'] = pd.read_csv(f'kind_m{i}.csv').iloc[:, 2:-3]
    print(f'Month {i}:')
    print(f'# kinds:', len(kinds[f'm{i}']))


# ### Oversampling & Undersampling Combination

# In[20]:


resampling_dataset(train_month = range(2, 5), final_br = 1, num_os = 10)
'''
