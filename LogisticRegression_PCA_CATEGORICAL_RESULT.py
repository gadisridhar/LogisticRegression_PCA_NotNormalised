#!/usr/bin/env python
# coding: utf-8

# For Binary Classification

# In[8]:


############################ For regression: f_regression, mutual_info_regression
############################ For classification: chi2, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import seaborn as sns
import category_encoders as ce
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, f_classif, mutual_info_classif, mutual_info_regression
from time import time


# In[9]:


from sklearn.linear_model import LogisticRegression


# In[10]:


df = pd.read_csv('cancer.csv')


# In[11]:


df = df.drop(['id'], axis='columns')


# In[12]:


df.head(10)


# In[13]:



# FEATURES FROM RIDGECV + SELECTFROMMODEL METHOD
#df = df[['compactness_mean','concave points_mean','smoothness_se','concavity_se','concave points_se','fractal_dimension_se','fractal_dimension_worst','diagnosis']]    
# FEATURES FROM RIDGECV METHOD
#df = df[['compactness_mean','smoothness_se','concave points_se','fractal_dimension_se','fractal_dimension_worst','concavity_se','diagnosis']]


# In[14]:


np.array(df.columns)


# In[15]:


sns.heatmap(df.corr())


# In[16]:


enc = ce.OneHotEncoder(use_cat_names=True)
df_enc = enc.fit_transform((df['diagnosis']))


# In[17]:


df_enc
df = df.join(df_enc)


# In[18]:


#df['diagnosis'].replace(['M','B'], [1,0], inplace = True)
df.head(10)


# In[19]:


df = df.drop(['diagnosis'], axis='columns')


# In[21]:


df.tail(10)


# In[22]:


#Y = df.iloc[:, 1].values
#Y =  df["diagnosis"].values
Y = df.iloc[:, 31].values
print(Y.shape)
print(type(Y))
print(Y)


# In[23]:


X = df.iloc[:, 0:31].values
print(X.shape)
print(type(X))
print(X)


# In[235]:


# IMP = pd.DataFrame(importance) 
# #print(IMP)
# FEATNAMES =  pd.DataFrame(feature_names)
# #print(FEATNAMES)
# frames = [IMP, FEATNAMES]
# result = pd.concat([IMP, FEATNAMES], axis=1)
# (result)

# CONSIDERING FEATURES FROM RIDGECV
# - compactness_mean
# - smoothness_se
# - concave points_se
# - fractal_dimension_se
# - fractal_dimension_worst
# - concavity_se


# In[236]:


# sm = SMOTE(random_state=42)
# X_res, Y_res = sm.fit_resample(X, Y)


# In[237]:


# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X_res,Y_res, test_size=0.2, random_state=42)


# In[24]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


# In[25]:


X_train.shape


# In[26]:


X_test.shape


# In[27]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print(type(X_train))
print(type(Y_train))


# In[28]:


scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)


# In[29]:


pd.DataFrame(X_train)


# In[30]:


pd.DataFrame(X_test)


# In[31]:


pca = PCA(n_components=15)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[32]:


pd.DataFrame(X_train)


# In[33]:


pd.DataFrame(X_test)


# In[34]:


X_train.shape


# In[35]:


X_test.shape


# In[36]:


from sklearn.linear_model import LogisticRegression
lrRegression =  LogisticRegression(max_iter=500)


# In[37]:


lrRegression.fit(X_train,Y_train)


# In[38]:


Y_Pred_TEST = lrRegression.predict(X_test)
Y_Pred_TRAIN = lrRegression.predict(X_train)

# print(" ACCURACY SCORE FOR TEST DATA : ",lrRegression.score(Y_test, Y_Pred_TEST))
# print(" ACCURACY SCORE FOR TRAIN DATA : ",lrRegression.score(Y_train, Y_Pred_TRAIN))
#Y_Pred = lrRegression.predict(X_test[0].reshape(1,-1))

# for one value prediction
#print(lrRegression.predict(X_test[0].reshape(1,-1)))

# for multiple value prediction
#print(lrRegression.predict(X_test[0:10]))


# In[41]:


print(Y_Pred_TEST.shape)
print(Y_Pred_TRAIN.shape)


# In[42]:


acc =  accuracy_score(Y_test, Y_Pred_TEST)
acc2 = accuracy_score(Y_train, Y_Pred_TRAIN)
print("TEST DATA PRedict : ", acc)
print("TRAIN DATA PRedict : ",acc2)


# # WITH STANDARD SCALER APPROACH AND PCA  COMPONENTS = 15
# TEST DATA PRedict :  1.0
# TRAIN DATA PRedict :  0.9978021978021978
# 
# # WITH STANDARD SCALER APPROACH NON PCA
# #FROM STEP SELECTKBEST WHEN K = 7 TEST DATA 
# TEST DATA PRedict :  0.9824561403508771
# TRAIN DATA PRedict :  0.9340659340659341

# # WITHOUTSTANDAARD SCALER APPROACH
# #FROM STEP SELECTKBEST WHEN K = 7
# TEST DATA PRedict :  0.9824561403508771
# TRAIN DATA PRedict :  0.9340659340659341

# In[43]:


from sklearn.metrics import confusion_matrix, r2_score, accuracy_score
#TEST CONFUSION MATRIX
cm = confusion_matrix(Y_test, Y_Pred_TEST)
cm


# In[44]:


#TRAIN CONFUSION MATRIX
cm = confusion_matrix(Y_train, Y_Pred_TRAIN)
cm


# In[45]:


# PREVIOUS REPORT WITH STANDARD SCALER
classreport=  classification_report(Y_test, Y_Pred_TEST)
print(classreport)


# # PREVIOUS REPORT WITHOUT STANDARD SCALER
#  precision    recall  f1-score   support
# 
#            0       0.97      1.00      0.99        71
#            1       1.00      0.95      0.98        43
# 
#     accuracy                           0.98       114
#    macro avg       0.99      0.98      0.98       114
# weighted avg       0.98      0.98      0.98       114
# 
# # PREVIOUS REPORT FROM SELECTKBEST 
#  precision    recall  f1-score   support
# 
#            0       0.97      1.00      0.99        71
#            1       1.00      0.95      0.98        43
# 
#     accuracy                           0.98       114
#    macro avg       0.99      0.98      0.98       114
# weighted avg       0.98      0.98      0.98       114
# 

# In[46]:


classreport2 =  classification_report(Y_train, Y_Pred_TRAIN)
print(classreport2)


# PREVIOUS REPORT FROM SELECTKBEST
#  precision    recall  f1-score   support
# 
#            0       0.94      0.96      0.95       286
#            1       0.93      0.89      0.91       169
# 
#     accuracy                           0.93       455
#    macro avg       0.93      0.93      0.93       455
# weighted avg       0.93      0.93      0.93       455

# In[47]:


score = lrRegression.score(X_test, Y_test)
score


# prev 1 accuracy
# 0.9824561403508771

# In[48]:


plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, cmap='Blues_r', square=True)
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
all_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_title,  size=20)


# In[49]:


lrRegression.coef_


# In[50]:


lrRegression.intercept_


# In[ ]:




