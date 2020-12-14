#!/usr/bin/env python
# coding: utf-8

# In[6]:


cn=np.array([[ 959,    0,    2,    1,    0,    4,   10,    2,    2,    0],
       [   0, 1119,    2,    2,    0,    1,    4,    2,    5,    0],
       [   3,    7,  938,    7,   16,    3,   11,   11,   32,    4],
       [   1,    0,   24,  926,    2,   20,    3,    7,   20,    7],
       [   1,    2,    2,    1,  938,    0,   10,    3,    7,   18],
       [   8,    5,    4,   30,   10,  775,   20,    4,   32,    4],
       [   7,    3,    4,    1,    7,    7,  925,    0,    4,    0],
       [   1,    9,   22,    3,    9,    0,    0,  949,    2,   33],
       [   3,   10,    5,   16,    8,   19,    9,    9,  889,    6],
       [   8,    7,    2,   10,   35,    7,    0,   20,   10,  910]])


# In[5]:


import numpy as np


# In[7]:


cn


# In[8]:


import seaborn as sn


# In[9]:


import matplotlib.pyplot as plt


# In[15]:


plt.figure(figsize=(10,7))
sn.heatmap(cn,annot=True,fmt='d')
plt.xlabel('pre')
plt.ylabel('real')


# In[ ]:


# evaluate model by confusion matrix

