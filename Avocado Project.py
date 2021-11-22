#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Read in the Avocado Prices csv file as a DataFrame called df


# In[3]:


df = pd.read_csv(r"C:\Users\sndpy\Downloads\avocado.csv")


# In[4]:


#Lets check our data head:
    
df.head()


# In[5]:


#The Feature "Unnamed:0" is just a representation of the indexes, so it's useless to keep it, lets remove it !

df.drop('Unnamed: 0',axis=1,inplace=True)


# In[6]:


#Lets check our data head again to make sure that the Feature Unnamed:0 is removed

df.head()


# In[7]:


#use the info() methode to get an a general idea about our data:

df.info()


# In[8]:


#we dont have any missing values (18249 complete data) and 13 columns. Now let's do some Feature Engineering on the Date Feature so we can be able to use the day and the month columns in building our machine learning model later. 


# In[9]:


df['Date']=pd.to_datetime(df['Date'])
df['Month']=df['Date'].apply(lambda x:x.month)
df['Day']=df['Date'].apply(lambda x:x.day)


# In[ ]:


#Lets check the head to see what we have done:


# In[10]:


df.head()


# In[11]:


byDate=df.groupby('Date').mean()
plt.figure(figsize=(12,8))
byDate['AveragePrice'].plot()
plt.title('Average Price')


# In[12]:


plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)


# As we can from the heatmap above, all the Features are not corroleted with the Average Price column, instead most of them are correlated with each other. So now I am bit worried because that will not help us get a good model. Lets try and see. First we have to do some Feature Engineering on the categorical Features : region and type

# In[13]:


df['region'].nunique()


# In[14]:


df['type'].nunique()


# as we can see we have 54 regions and 2 unique types, so it's going to be easy to to transform the type feature to dummies, but for the region its going to be a bit complexe so I decided to drop the entire column. I will drop the Date Feature as well because I already have 3 other columns for the Year, Month and Day.
# 

# In[15]:


df_final=pd.get_dummies(df.drop(['region','Date'],axis=1),drop_first=True)


# In[16]:


df_final.head()


# In[17]:


df_final.tail()


# Now our data are ready! lets apply our model which is going to be the Linear Regression because our Target variable 'AveragePrice'is continuous. Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable

# In[18]:


X=df_final.iloc[:,1:14]
y=df_final['AveragePrice']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# Creating and Training the Model

# In[19]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)


# In[20]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# The RMSE is low so we can say that we do have a good model, but lets check to be more sure. Lets plot the y_test vs the predictions

# In[21]:


plt.scatter(x=y_test,y=pred)


# As we can see that we dont have a straigt line so I am not sure that this is the best model we can apply on our data
# 
# Lets try working with the DecisionTree Regressor model

# In[22]:


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
pred=dtr.predict(X_test)


# In[23]:


plt.scatter(x=y_test,y=pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# Here we can see that we nearly have a straigt line, in other words its better than the Linear regression model, and to be more sure lets check the RMSE.

# In[24]:


print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# our RMSE is lower than the previous one we got with Linear Regression. ok now I am going to try one last model to see if I can improve my predictions for this data which is the RandomForestRegressor

# In[25]:


from sklearn.ensemble import RandomForestRegressor
rdr = RandomForestRegressor()
rdr.fit(X_train,y_train)
pred=rdr.predict(X_test)


# In[26]:


print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# Well as we can see the RMSE is lower than the two previous models, so the RandomForest Regressor is the best model in this case.

# In[27]:


sns.distplot((y_test-pred),bins=50)


# Notice here that our residuals looked to be normally distributed and that's really a good sign which means that our model was a correct choice for the data.

# In[28]:


data = pd.DataFrame({'Y Test':y_test , 'Pred':pred},columns=['Y Test','Pred'])
sns.lmplot(x='Y Test',y='Pred',data=data,palette='rainbow')
data.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




