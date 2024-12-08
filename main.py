# %% DATA IMPORT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from business import ExpectationMaximization

df = pd.read_csv("diem_thi_thpt_2024_cleaned.csv")
data = df[df['region'] == 1]['ngoai_ngu']

# %% EM
em = ExpectationMaximization(array=data, 
                             K=2,
                             epsilon=0.1)

em.print_basic_info()

em.run()

em.param_sets

# Result assigned data
df_divided = em.assign_group()
df_divided['group_name'] = df_divided['group'].map({
    0: 'lower',
    1: 'higher'
    })

# %% VISUALIZATION
fig, ax = plt.subplots(figsize=(9, 6))
sns.kdeplot(data=data)
sns.kdeplot(data=df_divided, x='score', hue='group_name')
plt.show()

fig, ax = plt.subplots(figsize=(9, 6))
sns.histplot(data=data)
sns.histplot(data=df_divided, x='score', hue='group_name')
plt.show()













