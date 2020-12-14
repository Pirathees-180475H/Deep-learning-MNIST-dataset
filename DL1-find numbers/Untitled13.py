#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[2]:


(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()


# In[3]:


x_train.shape


# In[5]:


plt.matshow(x_train[15])  # to draw 2 dimentional arrays


# In[6]:


x_test.shape


# In[7]:


plt.matshow(x_test[15])  # to draw 2 dimentional arrays


# In[8]:


x_train_a=x_train/255  # scaling for accuracy


# In[9]:


X_test_a=x_test/255


# In[16]:


X_train_a_f=x_train_a.reshape(len(x_train),28*28) # reshape to 1 dim array


# In[17]:


X_test_a_f=X_test_a.reshape(len(x_test),28*28) # reshape to 1 dim array


# In[10]:


x_trainf=x_train.reshape(len(x_train),28*28) # reshape to 1 dim array


# In[11]:


x_testf=x_test.reshape(len(x_test),28*28) # reshape to 1 dim array


# In[12]:


# lets create simple nural network#


# In[13]:


model=keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation ="sigmoid")    #10 is out put shape
    
                                                                    #adam for efficient
] )

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy']
             )


# In[14]:


# finsihed 


# In[15]:


model.fit(x_trainf,y_train,epochs=10)


# In[22]:


model.fit(X_train_a_f,y_train ,epochs=4) # by scaling can get better accuracy --divid by 255


# In[23]:


# let do for test dataset


# In[24]:


model.fit(X_test_a_f,y_test )


# In[25]:


# lets do predict


# In[ ]:





# In[30]:


y_pre=model.predict(X_test_a_f) # model prideic for all test cases


# In[37]:


plt.matshow(x_test[0])


# In[32]:


plt.plot(y_pre[0])


# In[33]:


# max value is 7


# In[34]:


#or use numpy


# In[35]:


np.argmax(y_pre[0])


# In[39]:


# enother example
plt.matshow(x_test[45])


# In[40]:


plt.plot(y_pre[45]) 


# In[41]:


np.argmax(y_pre[45])


# In[42]:


y_pre[45]


# In[43]:


y_test[0]


# In[44]:


# prediction value is in array we need to convert expected values 


# In[58]:


y_predicted_values=[np.argmax(i) for i in y_pre] # convert to actual numbers


# In[56]:


y_predicted_values[:4]


# In[ ]:





# In[60]:


confusion_matrix=tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_values)


# In[61]:


confusion_matrix


# In[62]:


# lets visualize it


# In[63]:


import seaborn as sn


# In[ ]:


#in anoconda root 

