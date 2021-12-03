#!/usr/bin/env python
# coding: utf-8

# ### Customer Segmentation: 
#         To segment different customers on the basis of their purchases using kmeans clustering

# ### Importing libraries & Dataset

# In[3]:


# Libraries
import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans


# In[4]:


# Dataset
data=pd.read_csv('E:/Python/Python - Decodr/DecodR_Class/October - 2021/Projects/Customer Segmentation/data.csv')
data.head()


# ### Checking missing values

# In[5]:


data.isnull().sum() #no missing values


# ### Bivariate Analysis
# a) Age v/s Spending Score

# In[6]:


plt.scatter(data['Age'],data['Spending Score (1-100)'])
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.grid(linestyle = '--')
plt.show()


# In[7]:


# Taking only 'Age' and 'Spending Score (1-100)' columns
x=data[['Age','Spending Score (1-100)']]
x


# ### Initializing centroids and finding out the best no. of groups

# In[8]:


# Creating empty list, wcss, to store the sum of squared distance within a cluster.
wcss=[]
for i in range(1,10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# ### Visualizing to find the best no. of cluster (Elbow method)

# In[9]:


no_of_clusters=range(1,10)
plt.plot(no_of_clusters,wcss)
plt.grid(linestyle = '--')
plt.show()


# In[10]:


# Taking k = 4 and fitting the data
kmeans = KMeans(4)


# ### Fitting and predicting the data clusters

# In[11]:


identified_clusters=kmeans.fit_predict(x)
identified_clusters


# ### Merging the original dataset with number of clusters

# In[12]:


table_with_clusters=data.copy()
table_with_clusters['Clusters']=identified_clusters
table_with_clusters


# ### Visualizing the number of clusters in the form of scatterplot

# In[13]:


plt.scatter(table_with_clusters['Age'],table_with_clusters['Spending Score (1-100)'],c=table_with_clusters['Clusters'],cmap='rainbow')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('Plot with clusters')
plt.grid(linestyle = '--')
plt.show()


# b) Gender v/s Spending Score

# In[14]:


plt.figure(1 , figsize = (10 , 5))
ax=sns.countplot(x = 'Gender' , data = data)
for p in ax.patches:
    ax.annotate('{:}'.format(p.get_height()), (p.get_x()+0.3, p.get_height()+1))

plt.show()  


# ### Converting into numeric categorical feature (One Hot encoding)

# In[15]:


Gender=pd.get_dummies(data['Gender'],drop_first=True)
Gender.head()


# In[16]:


# Merging with original dataset and dropping the 'Gender' column
data=pd.concat([data,Gender],axis=1)
data.head()


# In[17]:


data.drop(['Gender'],axis=1,inplace=True)
data.head()


# In[18]:


x=data[['Male','Spending Score (1-100)']]
x


# ### Again finding the best number of clusters w.r.t gender

# In[19]:



wcss=[]
for i in range(1,10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# ### Elbow method

# In[20]:


no_of_clusters=range(1,10)
plt.plot(no_of_clusters,wcss)
plt.grid(linestyle = '--')


# In[21]:


# Taking K = 3
kmeans = KMeans(3)


# In[22]:


identified_clusters=kmeans.fit_predict(x)
identified_clusters 


# ### Merging the original dataset with number of clusters

# In[23]:


table_with_clusters=data.copy()
table_with_clusters['Clusters']=identified_clusters
table_with_clusters


# ### Visualizing clusters

# In[24]:


plt.scatter(table_with_clusters['Spending Score (1-100)'],table_with_clusters['Male'],c=table_with_clusters['Clusters'],cmap='rainbow')
plt.xlabel('Male')
plt.ylabel('Spending Score (1-100)')
plt.title('Plot with clusters')
plt.grid(linestyle = '--')
plt.show()


# c) Annual Income (k$) v/s Spending Score (1-100)

# In[27]:


data=pd.read_csv('E:/Python/Python - Decodr/DecodR_Class/October - 2021/Projects/Customer Segmentation/data.csv')
data.head()


# In[28]:


plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.grid(linestyle = '--')
plt.show()


# In[29]:


x=data[['Annual Income (k$)','Spending Score (1-100)']]
x


# ### Finding the best cluster based on Annual Income

# In[30]:



wcss=[]
for i in range(1,10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# ### Elbow method

# In[31]:


no_of_clusters=range(1,10)
plt.plot(no_of_clusters,wcss)
plt.grid(linestyle = '--')
plt.show()


# In[32]:


# Taking k=5
kmeans = KMeans(5)


# In[33]:


identified_clusters=kmeans.fit_predict(x)
identified_clusters 


# In[34]:


table_with_clusters=data.copy()
table_with_clusters['Clusters']=identified_clusters
table_with_clusters


# ### Visualizing number of clusters

# In[59]:


plt.scatter(table_with_clusters['Annual Income (k$)'],table_with_clusters['Spending Score (1-100)'],c=table_with_clusters['Clusters'],cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Plot with clusters')
plt.grid(linestyle = '--')
plt.show()


# In[ ]:




