#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
'''
os.chdir("C:/Users/Darui Yen/OneDrive/桌面/data_before_mid")
os.getcwd()
'''

# ### Score

# In[224]:


def score1(r, ag):
    eff = r/ag
    eff_one = eff > 1
    score = r*np.log(1+eff)
    score = score*eff_one
    return score

def score2(r, ag):
    eff = r/ag
    eff_one = eff > 1
    score = (r-0.5)*np.log(1+eff)
    score = score*(score > 0)
    score = score*eff_one
    return score

def score3(r, ag):
    eff = r/ag
    eff_one = eff > 1
    score = (r-0.5)*np.log(2+eff)
    score = score * (score > 0)
    score = score*eff_one
    return score

def contourplot(x,y,z):
    fig = plt.figure()
    plt.contourf(x, y, z, levels = 50, cmap = "gist_rainbow", alpha = 0.8)
    plt.colorbar()
    plt.plot([0, 1], [0, 1], 'k-', lw = 2)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("Recall")
    plt.ylabel("Aging Rate")
    fig.set_size_inches(9, 5)
    
    return fig

'''
# In[222]:


x = np.linspace(0.01, 1, 100)
y = np.linspace(0.01, 1, 100)
X, Y = np.meshgrid(x, y)

Z1 = score1(X, Y)
Z2 = score2(X, Y)
Z3 = score3(X, Y)


# In[231]:


###score3###
fig3 = contourplot(X, Y, Z3)
plt.title("Score 3")
plt.plot([0.2,1], [0,0.8], "w-.", lw = 1)
fig3.savefig("score3.png", transparent = True)


# In[205]:


###score2###
fig2 = contourplot(X, Y, Z2)
plt.title("Score 2")
fig2.savefig("score2.png", transparent = True)


# In[206]:


###score1###
fig1 = contourplot(X, Y, Z1)
plt.title("Score 1")
fig1.savefig("score1.png", transparent = True)
'''
