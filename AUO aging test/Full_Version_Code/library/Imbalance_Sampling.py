#!/usr/bin/env python
# coding: utf-8

# In[16]:


import os
import math
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# import smote_variants as sv
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, InstanceHardnessThreshold, NearMiss
from imblearn.over_sampling import ADASYN, SMOTEN

from library.Data_Preprocessing import Balance_Ratio
from library.Training_Data_Processing import Corner, Kind
'''
os.chdir('C:/Users/user/Desktop/Darui_R08621110')
os.getcwd()
'''

# ## 

# In[10]:


def label_divide(train, test, label = 'GB', train_only = False):
    
    train_x = train.drop(columns = label)
    train_y = train[label]
    
    if not train_only:
        test_x = test.drop(columns = label)
        test_y = test[label]    
        return train_x, train_y, test_x, test_y
    else:
        return train_x, train_y


# ### self-defined oversampling (border)
# first writen by chungcheng, and then modified 

# In[11]:


'''DEF 1'''
## 計算距離 data1 及 data2 的距離
## Output : 資料點之間的距離
def Distance(data1, data2):
    
    data1 = data1.iloc[:,:-1].values
    data2 = data2.iloc[:,:-1].values

    df=pd.DataFrame()
    for i in tqdm(range(len(data1))):
        
        hamming_set=[]
        for j in range(len(data2)):
            
            hamming = abs(data1[i] - data2[j]).sum()
            hamming_set=np.append(hamming_set,hamming)
            
        hamming_set=pd.DataFrame(hamming_set).T
        df=pd.concat([df,hamming_set])
        
    dis_df = df.reset_index().iloc[:,1:]
    
    return dis_df


'''DEF 2'''
## 給定 df 和 value 找出其 row 和 col
def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    
    return listOfPos


'''DEF 3'''
## Input : X+Y(level)
#給定一個 資料集 和 值 後, 找出相對應的位置
def ID_Given_Distance_2(data1, data2, d):
    
    D_Matrix = Distance(data1, data2)
    ##數量不同 無法使用上三角
    D_Matrix = D_Matrix.where(np.triu(np.ones(D_Matrix.shape)).astype(np.bool))
    combine = getIndexes(D_Matrix, d)
    
    return combine


'''DEF 4'''
# 2個以上不同 的排列組合
def perm(cols):
    
    s=set()
    num=cols-1 
    random_list = []
    for i in range(3) :
        TF_list_1 = [True]*(cols-1-i) + [False]*(i+1)
        TF_list_2 = [False]*(cols-1-i) + [True]*(i+1)
        shuffle_time = [1, 3, 6]
        
        for j in range(shuffle_time[i]) :
            temp1 = TF_list_1.copy()
            temp2 = TF_list_2.copy()
            random.shuffle(temp1)
            random.shuffle(temp2)
            random_list.append(temp1)
            random_list.append(temp2)
            
    return(random_list)
    

'''DEF 7'''
#找到各組間隔為 rank 的座標
def cumu_conbine(data, rank, level = 'GB'):
    
    data1=data[data[level]==1]
    data2=data[data[level]==0]
    combine=[]
    for i in range(rank):
        combine=combine+ID_Given_Distance_2(data1,data2,i+1)
        
    return combine


'''Main Function 1'''
def Border(data, Near_Minor = 3, Major_Ratio_max = 0.5, n_major_corner = 20, level = 'GB'):
    
    data1=data[data[level]==1]
    data2=data[data[level]==0]
    d=data.iloc[:,:-1].copy()
    training_df=pd.DataFrame()
    
    for a in tqdm(range(2,Near_Minor+1)):

        combine=ID_Given_Distance_2(data1, data1, a)
        smote_df=pd.DataFrame()
        if len(combine)!=0:

            for b in tqdm(range(len(combine))):

                ##選定一組數字,一個當中心點
                a_pair=combine[b]
                point_0=a_pair[0]
                point_1=a_pair[1]

                # d_smote 的初始值和中心點一樣
                d_smote=d.loc[point_0].copy()

                # 找出有差異處的 cols
                d_X=d.loc[point_0]-d.loc[point_1]
                cols=d_X[d_X!=0].index

                ## L為距離 2 的點之間的排列組合
                L=perm(len(cols))

                '''創造DATA_SMOTE(給定中心點:d_smote 和 L)'''
                s_df=pd.DataFrame()

                for i in tqdm(range(len(L))):

                    cb=L[i]
                    s=pd.DataFrame([d_smote]).copy()

                    for j in range(len(cb)):
                        if cb[j]==True:
                            s[cols[j]]=1-d_smote[cols[j]]
                        elif cb[j]==False:
                            s[cols[j]]=d_smote[cols[j]]
                    s_df=pd.concat([s_df,s])
                s_df=s_df.reset_index(drop=True)

                smote_df = pd.concat([smote_df, s_df]) #new added
                smote_df['GB'] = 1 #new_added

        smote_df=smote_df.drop_duplicates().reset_index(drop=True)
        training_df=pd.concat([training_df,smote_df])

    training_df=training_df.drop_duplicates().reset_index(drop=True)
    
    return training_df


# ### oversampling

# In[12]:


def before_over(dataset, label = 'GB'):
    
    colnames = dataset.columns
    Y = dataset[label]
    Y = Y.reset_index(drop = True)
    Y = np.array(Y)
    X = dataset.drop(columns = [label])
    X = X.reset_index(drop = True)
    X = X.to_numpy()
    
    return X, Y, colnames


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
        over_sampler = sv.ROSE(ratio)   
    elif method == method_list[4]:
        over_sampler = SMOTEN(sampling_strategy = ratio, k_neighbors = n_neighbors)
    elif method == method_list[5]:
        over_sampler = ADASYN(sampling_strategy = ratio, n_neighbors = n_neighbors)    
    
    if method in method_list[0:4]:
        over_X, over_Y = over_sampler.sample(X, Y)
    else:
        over_X, over_Y = over_sampler.fit_resample(X, Y)
    
    return over_X, over_Y


# ### undersampling

# In[13]:


def before_under(dataset, label = 'GB'):
    
    Y = dataset[label]
    X = dataset.drop(columns = [label])
    
    return X, Y


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


# ### protocol to generate datasets

# In[39]:


def under_over(dataset, over_method, under_method, over_ratio, under_ratio, label = 'GB'):
    
    #undersampling
    if under_method != None:
        X, Y = before_under(dataset, label)
        Y = Y.astype(int)
        print('Size before Undersampling:', len(Y))
        under_X, under_Y = under_sample(X, Y, under_method, under_ratio)
        dataset = pd.concat([under_X, under_Y], axis = 1)
        print('Size after Undersampling:', len(under_Y))
    
    #oversampling
    temp_X, temp_Y, colnames = before_over(dataset, label)
    print('Size before Oversampling:', len(temp_Y))
    over_X, over_Y = over_sample(temp_X, temp_Y, over_method, over_ratio)
    X, Y = after_over(over_X, over_Y, colnames)
    print('Size after Oversampling:', len(Y))
        
    return X, Y


def over_under(dataset, over_method, under_method, over_ratio, under_ratio, label = 'GB') :
    
    #oversampling
    if over_method != None :
        X, Y, colnames = before_over(dataset, label)
        print('Size before Oversampling:', len(Y))
        temp_X, temp_Y = over_sample(X, Y, over_method, over_ratio)
        over_X, over_Y = after_over(temp_X, temp_Y, colnames)
        print('Size after Oversampling:', len(over_Y))
        over_dataset = pd.concat([over_X, over_Y], axis = 1)
        dataset = over_dataset.rename(columns = {0 : label})

    #undersampling
    X, Y = before_under(dataset, label)
    Y = Y.astype(int)
    under_X, under_Y = under_sample(X, Y, under_method, under_ratio)
    print('Size after Undersampling:', len(under_Y))
    
    return under_X, under_Y
    
    
def generate_set(train_data, over_method, under_method, index, over_ratio, under_ratio, order, label = 'GB'):
    
    print('\n', f'Generating Dataset {index}')
    
    if order == 'under' :
        train_x, train_y = under_over(train_data, over_method, under_method, over_ratio, under_ratio, label)
    elif order == 'over' :
        train_x, train_y = over_under(train_data, over_method, under_method, over_ratio, under_ratio, label)
        
    train = pd.concat([train_x, train_y], axis = 1)
    train = train.rename(columns = {0: label})
    
    return train


def border_set(train_data, kind_data, under_method, index, min_over, under_ratio, order):
    
    ##### oversampling first #####
    if order == 'over':
        print('Size before Border:', len(train_data))
        redo = True
        distance = 12
        while redo:
            OS_B = Border(kind_data, Near_Minor = distance)
            self_runhist = pd.concat([train_data, OS_B], axis = 0).reset_index(drop = True)
            if (len(self_runhist) - len(train_data)) < sum(train_data.GB)*min_over:
                distance += 1
            else:
                redo = False
        print('Size after Border:', len(self_runhist))
        
        dataset = generate_set(self_runhist, None, under_method, index, over_ratio = None, under_ratio = under_ratio, 
                               order = 'over')
        print(f'Size after Undersampling:', dataset.shape, ', Balance Ratio:', Balance_Ratio(dataset),               ', distance:', distance)
        
        return dataset
    
    ##### undersampling first #####
    elif order == 'under':
        print('Size before Undersampling:', len(train_data))
        self_under = generate_set(train_data, None, under_method, index, over_ratio = None, under_ratio = under_ratio, 
                                  order = 'over')
        print('Size after Undersampling:', len(self_under))
        
        corner_overlap = Corner(self_under)
        under_kind = Kind(corner_overlap).iloc[:, :-3]
        
        redo = True
        distance = 12
        while redo:
            US_B = Border(under_kind, Near_Minor = distance)
            dataset = pd.concat([self_under, US_B], axis = 0).reset_index(drop = True)
            if (len(dataset) - len(self_under)) < len(self_under)*min_over*0.1:
                distance += 1
            else:
                redo = False
        print('Size after Border:', dataset.shape, ', Balance Ratio:', Balance_Ratio(dataset), ', distance:', distance)
        
        return dataset


# ## 

# ### loading training data & kind

# In[19]:
'''

##### training data #####
training_month = [2, 3, 4]

runhist = {}
for i in training_month:
    runhist[f'm{i}'] = pd.read_csv(f'relabel_runhist_m{i}.csv', index_col = 'id').iloc[:, 1:]
    print(f'Dimension of month {i}:', runhist[f'm{i}'].shape, ', # Bad Instance:', sum(runhist[f'm{i}'].GB))
runhist['all'] = pd.read_csv('relabel_runhist.csv', index_col = 'id').iloc[:, 1:]
print('Dimension of all runhist:', runhist['all'].shape, ', # Bad Instance:', sum(runhist['all'].GB))

##### kind data (for border) #####
kinds = {}
for i in training_month:
    kinds[f'm{i}'] = pd.read_csv(f'kind_m{i}.csv').iloc[:, 2:-3]
    print(f'Number of kinds in month {i}:', len(kinds[f'm{i}']))
kinds['all'] = pd.read_csv('kind.csv').iloc[:, 2:-3]
print('Number of kinds in all runhist:', len(kinds['all']))


# ### oversampling by self-defined method

# In[40]:


for i in tqdm(training_month):
    dataset_2 = border_set(runhist[f'm{i}'], kinds[f'm{i}'], 'NM', 2, min_over = 9, under_ratio = 1, order = 'over')
    dataset_6 = border_set(runhist[f'm{i}'], kinds[f'm{i}'], 'NM', 6, min_over = 9, under_ratio = 0.1, order = 'under')
    dataset_2.to_csv(f'm{i}_dataset_2.csv')
    dataset_6.to_csv(f'm{i}_dataset_6.csv')


# ### oversampling & undersampling

# In[50]:


##### generate datasets #####
dataset = {}
combine_dataset = {}
for i in range(10):
    combine_dataset[i] = pd.DataFrame()

for i in tqdm(training_month):
    
    dataset[2] = border_set(runhist[f'm{i}'], kinds[f'm{i}'], 'NM', 2, min_over = 9, under_ratio = 1, order = 'over')
    dataset[6] = border_set(runhist[f'm{i}'], kinds[f'm{i}'], 'NM', 6, min_over = 9, under_ratio = 0.1, order = 'under')
    
    dataset[0] = generate_set(runhist[f'm{i}'], 'NoSMOTE', None, 0, over_ratio = None, under_ratio = None, order = 'over')

    dataset[1] = generate_set(runhist[f'm{i}'], 'ADASYN', 'NM', 1, over_ratio = 0.015, under_ratio = 1, order = 'over')
    dataset[3] = generate_set(runhist[f'm{i}'], 'ROSE', 'NM', 3, over_ratio = 0.015, under_ratio = 1, order = 'over')
    dataset[4] = generate_set(runhist[f'm{i}'], 'SMOTEN', 'NM', 4, over_ratio = 0.015, under_ratio = 1, order = 'over')

    dataset[5] = generate_set(runhist[f'm{i}'], 'ADASYN', 'NM', 5, over_ratio = 1, under_ratio = 0.1, order = 'under')
    dataset[7] = generate_set(runhist[f'm{i}'], 'ROSE', 'NM', 7, over_ratio = 1, under_ratio = 0.1, order = 'under')
    dataset[8] = generate_set(runhist[f'm{i}'], 'SMOTEN', 'NM', 8, over_ratio = 1, under_ratio = 0.1, order = 'under')

    dataset[9] = generate_set(runhist[f'm{i}'], None, 'NM', 9, over_ratio = None, under_ratio = 0.1, order = 'over')
    
    ### combine all training data after sampling by each month ###
    for j in range(10):
        temp_combine = pd.concat([combine_dataset[j], dataset[j]], axis = 0).fillna(0)
        temp_cols = temp_combine.columns.to_list()
        GB_pos = temp_cols.index('GB')
        fine_cols = temp_cols[: GB_pos] + temp_cols[GB_pos+1: ] + temp_cols[GB_pos: GB_pos+1]
        combine_dataset[j] = temp_combine[fine_cols]
        
        dataset[j].to_csv(f'm{i}_dataset_{j}.csv')
        combine_dataset[j].to_csv(f'dataset_{j}.csv')

'''
