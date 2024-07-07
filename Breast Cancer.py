import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df=pd.read_csv(r"C:\Users\user\Desktop\Breast_cancer_data.csv")


df.head()


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


df['diagnosis'].value_counts().plot(kind='bar')
plt.xticks(rotation=0)
plt.title('Distributiion of Diagnosis')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# In[11]:


df.hist(bins=20,figsize=(20,15),layout=(3,2))
plt.tight_layout()
plt.show()


# In[12]:


df.plot(kind='box',figsize=(20,15),subplots=True,layout=(3,2))
plt.tight_layout()

plt.show()


# In[13]:


data=df.corr()
data


# In[14]:


# Plot heatmap of the correlation matrix
plt.figure(figsize=(12, 6))
sns.heatmap(data, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()


# In[16]:


import warnings
warnings.filterwarnings('ignore')
sns.pairplot(df,hue='diagnosis')
plt.show()


# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[25]:


x = df[['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']]
y = df['diagnosis']


# y.sample(4)
# 

# In[27]:


y.sample(4)

x.sample(5)


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[29]:


x_train.shape , y_train.shape , x_test.shape , y_test.shape


# In[30]:


# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[31]:


Model = LogisticRegression()
Model.fit(x_train, y_train)


# In[33]:


y_pred = Model.predict(x_test)


# In[34]:


# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred)*100,'%')


# In[ ]:




