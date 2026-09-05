#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv('train.csv',index_col='id')
data = data.map(lambda s: s.lower() if isinstance(s,str)else s)
data.head()
X_test = pd.read_csv('test.csv',index_col='id')


# In[3]:


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
print(data['gender'].unique())
print(data['stress_level'].unique())
data['academic_work_impact'].unique()


stress_lvl = ['low','medium','high',]
academic_impact = ['yes','no','nan']

preprocessor = ColumnTransformer(
    transformers=[
        ('nominal',OneHotEncoder(drop='if_binary',handle_unknown='ignore',sparse_output=False),['gender']),

        ('ordinal',OrdinalEncoder(categories=[stress_lvl,academic_impact],
                                  handle_unknown='use_encoded_value',unknown_value=np.nan,encoded_missing_value=np.nan),['stress_level','academic_work_impact'])


    ]
)


transformed_data = preprocessor.fit_transform(data)
X_data_test = preprocessor.transform(X_test)
transformed_data


# In[4]:


feature_names = preprocessor.get_feature_names_out()
transformed_df = pd.DataFrame(transformed_data,columns=feature_names,index=data.index)
X_data = pd.DataFrame(X_data_test,columns=feature_names,index=X_test.index)

remaining_cols = data.drop(columns=['gender','stress_level','academic_work_impact'])
df = pd.concat([remaining_cols,transformed_df],axis=1)
X_df = pd.concat([remaining_cols,X_data],axis=1)
df


# In[ ]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import  IterativeImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

mice = IterativeImputer(
    estimator=GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42),
    max_iter=5,
    random_state=42,
    n_nearest_features=10,
    verbose=2,
)
imputed_data = mice.fit_transform(df)
imputed_data_test = mice.transform(X_df)

data_b = pd.DataFrame(imputed_data,columns=df.columns)
X_d_test = pd.DataFrame(imputed_data_test,columns=X_df.columns)

data_b
data_b.columns


# In[6]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats

data_b['sleep_hours'] = data_b['sleep_hours'].astype(int)

r_pb, p_val_pb = stats.pointbiserialr(data_b['age'],data_b['addicted_label'])
print(f"Point-Biserial Correlation: {r_pb} ('p_value': {p_val_pb})")

addicted_age = data_b[data_b['addicted_label']==1]['sleep_hours']
not_addicted_age = data_b[data_b['addicted_label']==0]['sleep_hours']

print(f'Mean Addicted: {addicted_age.mean():.1f}')
print(f'Mean Not Addicted: {not_addicted_age.mean():.2f}')

t_stat, p_val_t = stats.ttest_ind(addicted_age,not_addicted_age,equal_var=False)
print(f"T_Test p value: {p_val_t}")

model = smf.logit('addicted_label ~ sleep_hours',data=data_b).fit()
print(model.summary())


# In[7]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import shap
import pandas as pd

# 1. Features & Data Setup
features = [
    'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
    'work_study_hours', 'sleep_hours', 'notifications_per_day',
    'app_opens_per_day', 'weekend_screen_time',
    'nominal__gender_female', 'nominal__gender_male',
    'nominal__gender_other', 'nominal__gender_nan', 'ordinal__stress_level',
    'ordinal__academic_work_impact'
]

X = data_b[features]
y = data_b.addicted_label

# 2. Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


my_model = RandomForestClassifier(
    n_estimators=50, 
    max_depth=10, 
    max_samples=0.5, 
    n_jobs=-1, 
    random_state=42,
    class_weight = 'balanced'
)

import time
print("Training single model...")
start = time.time()
my_model.fit(X_train, y_train)
print(f"Training finished in {time.time() - start:.2f} seconds!")

# 2. Get your classification report immediately
y_pred = my_model.predict(X_val)
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_val, y_pred))




# In[49]:


predicted_series = pd.Series(y_pred,index=X_val.index,name='predicted_addicted_label').sort_index(ascending=True)
print(predicted_series.head(10))
print(X_val.loc[9])


# In[17]:


import shap

explainer = shap.TreeExplainer(my_model)

X_val_sample = shap.sample(X_val, 100)
shap_values = explainer(X_val_sample,approximate=True)


# In[18]:


shap.initjs()


# In[29]:


exp = shap.Explanation(values=shap_values.values,
 base_values=shap_values.base_values,
 data=X_val_sample, feature_names=X_val_sample.columns if hasattr(X_val_sample, 'columns') else None)
shap.plots.beeswarm(exp[:,:,1])


# In[52]:


background_summary = shap.kmeans(X_val, 50)
k_explainer = shap.KernelExplainer(my_model.predict_proba, background_summary)
X_subset = X_val_sample.iloc[:5] if hasattr(X_val_sample, 'iloc') else X_val_sample[:5]

k_shap_values = k_explainer.shap_values(X_subset, nsamples=100)

vals = k_shap_values[1] if isinstance(k_shap_values, list) else k_shap_values

shap.summary_plot(vals, X_subset, plot_type='bar')

