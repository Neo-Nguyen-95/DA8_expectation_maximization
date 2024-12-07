# %% DATA IMPORT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from business import ExpectationMaximization

df = pd.read_csv("cleaned_score_data.csv")
data = df['ngoai_ngu']

# %% EM
em = ExpectationMaximization(array=data, 
                             K=2,
                             epsilon=0.1)

em.print_basic_info()

em.run()

df_divided = em.assign_group()

em.param_sets

# %% VISUALIZATION
fig, ax = plt.subplots(figsize=(9, 6))
sns.kdeplot(data=data)
sns.kdeplot(data=df_divided, x='score', hue='group')
plt.show()











