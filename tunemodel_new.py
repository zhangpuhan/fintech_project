
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from datetime import datetime
import seaborn as sns


# In[15]:


df_2015 = pd.read_csv("cleaneddata12042018.csv", index_col=0)


# In[42]:


features = pd.read_csv("determinedfeatures.csv", index_col=0)


# In[44]:


features = features.values.reshape(-1).tolist()


# In[47]:


df_train = df_2015.query("train_flg == 1")
df_test =df_2015.query("train_flg == 0" )


# In[54]:


features.remove("train_flg")


# In[58]:


print("train and test sizes:")
print(df_train.shape, df_test.shape)


# In[56]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(df_train[features], df_train.loanstatus, 
                                                      test_size=0.3, random_state=2016, stratify = df_train.loanstatus )


# In[59]:


print("X_train, X_valid, y_train, y_valid sizes:")
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)


# In[60]:


X_test, y_test = df_test[features], df_test.loanstatus


# In[62]:


print("X_test, y_test sizes:")
print(X_test.shape, y_test.shape)


# In[63]:


dtrain = xgb.DMatrix(X_train, y_train, missing = np.NAN)
dvalid = xgb.DMatrix(X_valid, y_valid, missing = np.NAN)
dtest = xgb.DMatrix(X_test, y_test, missing = np.NAN)


# ## Bayesian optimization

# In[66]:


from bayes_opt import BayesianOptimization

xgtrain = xgb.DMatrix(df_train[features], df_train.loanstatus, missing = np.NAN)

def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma):
    params = dict()
    params['objective'] = 'binary:logistic'
    params['eta'] = 0.05
    params['max_depth'] = int(max_depth )   
    params['min_child_weight'] = int(min_child_weight)
    params['colsample_bytree'] = colsample_bytree
    params['subsample'] = subsample
    params['gamma'] = gamma
    params['verbose_eval'] = False 



    cv_result = xgb.cv(params, xgtrain,
                       num_boost_round=100000,
                       nfold=3,
                       metrics={'auc'},
                       seed=2018,
                       callbacks=[xgb.callback.early_stop(50)])
    print(cv_result)

    return cv_result['test-auc-mean'].max()


# In[ ]:


xgb_BO = BayesianOptimization(xgb_evaluate, 
                             {'max_depth': (4, 8),
                              'min_child_weight': (0, 20),
                              'colsample_bytree': (0.2, 0.8),
                              'subsample': (0.5, 1),
                              'gamma': (0, 2)
                             }
                            )

xgb_BO.maximize(init_points=5, n_iter=40)


# In[ ]:


# Tuning results
xgb_BO_scores = pd.DataFrame(xgb_BO.res['all']['params'])
xgb_BO_scores['score'] = pd.DataFrame(xgb_BO.res['all']['values'])
xgb_BO_scores = xgb_BO_scores.sort_values(by='score',ascending=False)
print(xgb_BO_scores.head(3))


# In[ ]:


xgb_BO_scores.to_csv("tuned_parameters.csv")

