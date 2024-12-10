# %% DATA IMPORT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from business import ExpectationMaximization, WelchTTest

df = pd.read_csv("diem_thi_thpt_2024_cleaned.csv")
data = df[df['region'] == 1]['ngoai_ngu']

# %% EM
em = ExpectationMaximization(array=data, 
                             K=2,
                             epsilon=1e-3)

em.print_basic_info()

em.run()

em.param_sets

# Result assigned data
df_divided = em.assign_group()
df_divided['group_name'] = df_divided['group'].map({
    0: 'higher',
    1: 'lower'
    })

df_divided.drop(columns='group', inplace=True)

# %% VISUALIZATION
fig, ax = plt.subplots(figsize=(9, 6))
sns.kdeplot(data=data)
sns.kdeplot(data=df_divided, x='score', hue='group_name')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=data, 
             bins=150, 
             fill=None,
             edgecolor='green',
             linewidth=1)
sns.histplot(data=df_divided, x='score', 
             # hue='group_name', 
             bins=150,
             linewidth=0.5,
             # palette=['#FC2947', '#0079FF']
             )
plt.title("Phân bố điểm Tiếng Anh tốt nghiệp THPT của HS tại Hà Nội")
plt.xlabel("Điểm Tiếng Anh")
plt.ylabel("Số lượng")

plt.show()

# %% DATA ANALYSIS

df_hn = pd.read_csv('student_number_HN.csv')
urbun_proportion = df_hn.groupby('Nội thành')['Chỉ tiêu 2024'].sum()

fig, ax = plt.subplots(figsize=(6, 6))
plt.pie(urbun_proportion, 
        labels=['ngoai_thanh', 'noi_thanh'],
        autopct='%1.1f%%',
        colors=['#77CDFF', '#86D293'],
        wedgeprops={'edgecolor': 'black', 'linewidth': 1}
        )
plt.show()

# %% Welch T Test
array1 = df_divided[df_divided['group_name'] == 'higher']['score']

array2 = df_divided[df_divided['group_name'] == 'lower']['score']

wtt = WelchTTest(array1, array2)

wtt.n1
wtt.n2

wtt.n1 = len(wtt.array1)
wtt.mu1 = wtt.array1.mean()
wtt.sd1 = wtt.array1.std()

wtt.n2 = len(wtt.array2)
wtt.mu2 = wtt.array2.mean()
wtt.sd2 = wtt.array2.std()



degree_freedom = wtt.get_df()
T = wtt.get_T()







