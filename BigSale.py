#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Import the require lib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Load the train and test dataset in pandas DataFrame
train= pd.read_csv("bigmart_train.csv")
test=pd.read_csv("bigmart_test.csv")


# In[4]:


#check numb of row and col in train dataset
train.shape


# In[6]:


#print the name of train data set
train.columns


# In[7]:


#check the row and col in test data set
test.shape


# In[8]:


test.head()


# In[9]:


#print the column of test data set
test.columns


# In[10]:


#combine test and train into one file to  perform EDA
train["source"]="train"
test["source"]="test"
data = pd.concat([train,test],ignore_index=True)
print(data.shape)


# In[11]:


data.head()


# In[12]:


#Describe func for numerical data summary
data.describe()


# In[13]:


#check of missing value
data.isnull().sum()


# In[14]:


#print the unique values in the Item_Fat_content columnll,where there are only two unique type of content in item:low fat and regular
data["Item_Fat_Content"].unique()


# In[15]:


#Print the outlet_Establiashment_Year col, Share the data ranges from 1985 to 2009
data["Outlet_Establishment_Year"].unique()


# In[16]:


#Calculate the outlate age
data["Outlet_Age"]=2018-data["Outlet_Establishment_Year"]
data.head(2)


# In[17]:


# Unique values in Outlet_Size
data["Outlet_Size"].unique()


# Note::There are also missing values in this column

# In[18]:


#Printing the col value of item_fat_content column
data["Item_Fat_Content"].value_counts()


# We can see that low fat product are the most abundent

# In[19]:


#Print th ecount value of Outlet_size
data["Outlet_Size"].value_counts()


# we can see thet the majority of outlets are medium and small-scale outlet

# In[20]:


#uSE the function to find out the most common value in Outlet_size
data["Outlet_Size"].mode()[0]


# The Output Shows that "medium" is the commonly accuring value

# In[21]:


#Two variable with missing values =Item_Weight and Outlet_SizeReplacing missing values in Outlet_Size with the value "medium"

data["Outlet_Size"]=data["Outlet_Size"].fillna(data["Outlet_Size"].mode()[0])


# In[22]:


#Replacing missing value in item_weight with the mean weight
data["Item_Weight"]=data["Item_Weight"].fillna(data["Item_Weight"].mode()[0])


# In[23]:


#Plot a histogram to reveal the distribution of item_Visibility column
data["Item_Visibility"].hist(bins=20)


# Detecting the Outlier:
# An Outlier is a data point that lies outside the overall pattern in a distribution. A commonly used rule states that a data point is an outlier if it is more than 1.5*IQR above the quartile or below the first quartlie.
# 
# Using this,One can remove the outlier and output the resulting data in fill data variable.
# 

# In[24]:


#calculate the first Quantile for item _Visibility
Q1=data["Item_Visibility"].quantile(0.25)


# In[25]:


#Calculte the Second Quantile
Q3=data["Item_Visibility"].quantile(0.75)


# In[26]:


#Calculte the interquartile(IQR)
IQR=  Q3 - Q1


# Now that the IQR range is known,remove the outlier from the data.The Resulting data is stored in fill_data variable

# In[27]:


fill_data=data.query("(@Q1 - 1.5 * @IQR) <= Item_Visibility <= (@Q3 +1.5 *@IQR)")


# In[28]:


#Display the Data

fill_data.head(2)


# In[29]:


#Check the shape of the resulting dataset without outliers

fill_data.shape


# In[30]:


#Shape of the original dataset is forteen thousand two hundred and four(14204) rows and fourteen column with outliners

data.shape


# In[31]:


#Assign full_data dataset to data DataFrame

data=fill_data


# In[32]:


data.shape


# In[33]:


#Modify Item_Visibility by conerting the numberical values into the categories low visibility,visibility and high visibility
data["Item_Visibility_bins"]=pd.cut(data["Item_Visibility"],[0.000,0.065,0.13,0.2],labels=["Low viz","Viz","High Viz"])


# In[34]:


#print the count Of Item_Visibility_bins
data["Item_Visibility_bins"].value_counts()


# In[35]:


#Replace null values with low Visibility

data["Item_Visibility_bins"]=data["Item_Visibility_bins"].replace(np.nan,'Low Viz',regex=True)


# We found types and diff in representation in categories of item_Fat_Content variable.This can be correctd using the code on screen.

# In[36]:


#Replace all other representations of Low fat with Low fat

data["Item_Fat_Content"]=data["Item_Fat_Content"].replace(["Low fat","LF"],"Low fat")


# In[37]:


#Replace all representation of reg with regular

data["Item_Fat_Content"]=data["Item_Fat_Content"].replace("reg",'Regular')


# In[38]:


#Print Unique fat Count Values

data["Item_Fat_Content"].unique()


# Code all categorical variable as numerical using "LabelEncoder" from Sklearn's preprocessing model 

# In[39]:


#Initialize th elable Encoder

le=LabelEncoder()


# In[40]:


#Transform Item_Fat_Content

data["Item_Fat_Content"]=le.fit_transform(data["Item_Fat_Content"])


# In[41]:


#Transform Item_Visibility_bins

data["Item_Visbility_bins"]=le.fit_transform(data["Item_Visibility_bins"])


# In[42]:


#Transform Outlet_Size
data["Outlet_Size"]=le.fit_transform(data["Outlet_Size"])


# In[43]:


#Transform Outlet_Location_Type
data["Outle_Location_Type"]=le.fit_transform(data["Outlet_Location_Type"])


# In[44]:


#Print the Unique values of Outlet_Type

data["Outlet_Type"].unique()


# In[45]:


#create dummies for outlet_Type

dummy=pd.get_dummies(data["Outlet_Type"])
dummy.head()


# In[46]:


#Explore the column Item_Identifier

data["Item_Identifier"]


# In[47]:


#As there are multiple values of Food , nonconsumable items and drinks with different numbers,combines the item type

data["Item_Identifier"].value_counts()


# In[48]:


#As multiple categories are present in Item_Identifier, reduce this by mapping
data["Item_Type_Combined"]=data["Item_Identifier"].apply(lambda x: x [0:2])
data["Item_Type_Combined"]=data["Item_Type_Combined"].map({'FD':"Food",
                                                        "NC":"Non-Consumable",
                                                        "DR":"Drinks"})


# In[49]:


#Only three categories are present in an Item_Type_combined Column

data["Item_Type_Combined"].value_counts()


# In[50]:


data.shape


# In[51]:


#Perform on hot encoding for all columns as the model work an numerical values and not an categorical a values

data=pd.get_dummies(data,columns=["Item_Fat_Content","Outlet_Location_Type","Outlet_Size","Outlet_Type","Item_Type_Combined"])


# In[78]:


data.dtypes


# In[84]:


import warnings
warnings.filterwarnings('ignore')

#Drop the column which have been converted to different types.

data.drop(["Item_Type","Outlet_Establishment_Year"], axis=1, inplace=True)

#Divide the dataset created earlier into train and test datasets.

train= data.loc[data["source"]=="train"]
test= data.loc[data["source"]=="test"]

#Drop unnecessary columns. Export Modified version of the file.

test.drop(["Item_Outlet_Sales","source"],axis=1,inplace=True)
train.drop(["source"],axis=1,inplace=True)

#Read the Train_Modified.csv dataset

train.to_csv("train_modified.csv",index=False)
test.to_csv("test_modified.csv",index=False)


# In[60]:


train2 = pd.read_csv("train.modified.csv")
test = pd.read_csv("test.modified.csv")


# In[55]:


#print the data types train2 column

train2.dtypes


# In[153]:


# Drop the irreievent veriable from train2 dataset


# In[58]:


#Create the independent variable X_train and dependent Variable Y_train

x_train=train2.drop(["Item_Outlet_Sales","Outlet_Identifier"],axis=1)
y_train=train2.Item_Outlet_Sales


# In[63]:


#Drop the irrelevent variable from text2 data set
x_test=test2.drop(["Outlet_Identifier","Item_Identifier"],axis=1)


# In[64]:


x_test


# In[65]:


x_train.head(2)


# In[66]:


y_train.head(2)


# In[67]:


#Import Sklearn libraries from model selection

from sklearn import model_selection
from sklearn.linear_model import LinearRegression


# In[68]:


#create a train and spllit

xtrain,ytrain,xtest,ytest=model_selection.train_test_split(X_train,y_train,test_size=0,random_state=42)


# In[69]:


#Fit Linear regression to the trainning dataset
lin = LinearRegression()


# In[70]:


lin.fit(xtrain,ytrain)


# In[71]:


#Find the Co-efficient and interception of the line
#use xtrain and y train for linear regression
print(lin.coef_)
lin.intercept_


# In[72]:


#predict the test set result of training data
predictions = lin.predict(xtest)
predictions


# In[73]:


import math


# In[74]:


#Find the RMSE for the model
print(math.sqrt(mean_squared(ytest,predictions)))


# In[75]:


#A goof RMSE for this problem is 1130. Here we can improve the RMSE by using Algorithm like Decision Tree,random forest, and xGboost\
#Next we will predict the sales of each product at a particular store in test data


# In[77]:


#predict the column Item_Outlet_Sales of test dataset
y_Sales_pred=lin.predict(x_test)
y_Sales_pred


# In[82]:


test_predictions=pd.DataFrame({
    "Item_Identifier":test2['Item_Identifier'],
    'Outlet_Identifier': test2 ['Outlet_Identifier'],
    'Item_Outlet_Sales' : y_sales_pred
} , columns=['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales'])


# In[83]:


test_predictions


# In[ ]:




