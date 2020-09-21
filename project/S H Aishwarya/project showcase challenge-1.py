#!/usr/bin/env python
# coding: utf-8

# In[97]:


#NumPy is a general-purpose array-processing package
import numpy as np

#Pandas is an open-source Python Library providing 
#high-performance data manipulation and analysis tool using its powerful data structures.
import pandas as pd

#Matplotlib is a visualization library in Python for 2D plots of arrays
import matplotlib as mpl
import matplotlib.pyplot as plt

# The sklearn library contains a lot of efficient tools for machine learning and statistical 
#modeling including classification, regression, clustering and dimensionality reduction.
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# IPython offers an enhanced read-eval-print loop (REPL) 
#environment particularly well adapted to scientific computing
from IPython.display import HTML


# In[98]:


#loading the dataset
boston=load_boston()
#characteristics
print(boston.DESCR)


# In[99]:


#converting to data frame
features=pd.DataFrame(boston.data,columns=boston.feature_names)
features


# In[100]:


#converting to a dataframe
target=pd.DataFrame(boston.target,columns=['target'])


# In[101]:


#concatenating the dataframes features and target
df=pd.concat([features,target],axis=1)


# In[102]:


#The describe() method is used for calculating some statistical data like percentile,
#mean and std of the numerical values of the Series or DataFrame
df.describe().round(decimals=2)


# In[103]:


#Correlation is a statistical technique that can show whether and how strongly pairs of variables are related.
corr=df.corr('pearson')
corrs=[abs(corr[attr]['target']) for attr in  list(features)]

l=list(zip(corrs,list(features)))
l.sort(key=lambda x:x[0] ,reverse=True)
corrs,labels=list(zip((*l)))

index=np.arange(len(labels))
#plot the values
plt.figure(figsize=(15,5))
plt.bar(index,corrs,width=0.5)
plt.xlabel('Attributes')
plt.ylabel('Correlatoin with the target variable')
plt.xticks(index,labels)
plt.show()   
           


# In[104]:


#LSTAT is more correlated with target
X=df['LSTAT'].values
Y=df['target'].values


# In[105]:


print(Y[:5])


# In[106]:


#MinMaxScaler Transform features by scaling each feature to a given range.
#This estimator scales and translates each feature individually such that it is in the given range on the training set
x_scaler=MinMaxScaler()
X=x_scaler.fit_transform(X.reshape(-1,1))
X=X[:,-1]
y_scaler=MinMaxScaler()
Y=y_scaler.fit_transform(Y.reshape(-1,1))
Y=Y[:,-1]


# In[107]:


print(Y[:5])


# In[111]:


#mean square error
def error(m,x,c,t):
    N=x.size
    e=sum(((m*x+c)-t)**2)
    return e*(1/(2*N))


# In[112]:


xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)


# In[113]:


def update(m,x,c,t,learning_rate):
    grad_m=sum(2*((m*x+c)-t)*x)
    grad_c=sum(2*((m*x+c)-t))
    m=m-grad_m*learning_rate
    c=c-grad_c*learning_rate
    return m,c


# In[114]:


"""Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient.
In machine learning, we use gradient descent to update the parameters of our model. Parameters refer to coefficients in Linear Regression"""

def gradient_descent(init_m,init_c,x,t,learning_rate,iterations,error_threshold):
    m=init_m
    c=init_c
    error_values=list()
    mc_values=list()
    for i in range(iterations):
        e=error(m,x,c,t)
        if e<error_threshold:
            print('Error less than the threshold. Stopping gradient descent')
            break
        error_values.append(e)
        m,c=update(m,x,c,t,learning_rate)
        mc_values.append((m,c))
    return m,c,error_values,mc_values    


# In[ ]:





# In[115]:


#%%time prints the wall time for the entire cell

init_m=0.9
init_c=0
learning_rate=0.001
iterations=250
error_threshold=0.001
m,c,error_values,mc_values=gradient_descent(init_m,init_c,xtrain,ytrain,learning_rate,iterations,error_threshold)


# In[116]:


#y=mx+c
predicted=m*xtest+c


# In[117]:


#error in the prediction
mean_squared_error(ytest,predicted)


# In[118]:


#plotting the values
plt.scatter(xtest,ytest,color='b')
plt.plot(xtest,predicted,color='r')


# In[119]:


#prediction values
predicted=predicted.reshape(-1,1)
xtest=xtest.reshape(-1,1)
ytest=ytest.reshape(-1,1)
#inverse transformation
xtest_scaled=x_scaler.inverse_transform(xtest)
ytest_scaled=y_scaler.inverse_transform(ytest)
predicted_scaled=y_scaler.inverse_transform(predicted)

xtest_scaled=xtest_scaled[:,-1]
ytest_scaled=ytest_scaled[:,-1]
predicted_scaled=predicted_scaled[:,-1]
#printing sample values for the output
p=pd.DataFrame(list(zip(xtest_scaled,ytest_scaled,predicted_scaled)),columns=['x','target_y','predicted_y'])
p=p.round(decimals=2)
p.head()


# In[ ]:





# In[ ]:




