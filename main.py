# %% DATA IMPORT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from business import ExpectationMaximization

df = pd.read_csv("cleaned_score_data.csv")
# df = df[df['region']==1]
x = np.array(df['ngoai_ngu'])

# %% TEST SITE
em = ExpectationMaximization(array=df['ngoai_ngu'], 
                             K=2,
                             epsilon=1)

em.run()

em.param_sets
em.overall_LL

sns.lineplot(data = em.overall_LL)
plt.show()











# %% STAT & VISUALIZATION

n = len(df)
mean = df['ngoai_ngu'].mean()
var = sum( (x - mean) **2 for x in df['ngoai_ngu']) / n

print(f"Number of student: {n}")
print(f"Mean: {mean:.2f}")
print(f"Variance: {var:.2f}")

fig, ax = plt.subplots(figsize=(9, 6))
sns.histplot(data=df, x='ngoai_ngu')
plt.show()





# %% EM
"""
Assumption: the distribution of score come from 2 Gaussian distribution,
I wanna find out the distributions' parameters.
"""

# theta = [µ, σ2]
def likelihood(x: np.ndarray, 
               theta: list
               ) -> np.ndarray:
    
    l = (1 / np.sqrt(2 * np.pi * theta[1]) * 
         np.exp(- (x - theta[0])**2 / (2 * theta[1]))
         )
    
    return l

def weighted_likelihood(pi_k: float, 
                        x: np.ndarray, 
                        theta: list
                        ) -> np.ndarray:
    
    r = pi_k * likelihood(x, theta)
    
    return r

def sum_weighted_likelihood(pi_list: tuple, 
                            x: np.ndarray, 
                            theta_list: tuple
                            ) -> np.ndarray:
    
    total = np.zeros_like(x)
    for pi_k, theta_k in zip(pi_list, theta_list):
        total += weighted_likelihood(pi_k, x, theta_k)
        
    return total

def overall_log_likelihood(pi_list: tuple,
                       x: np.ndarray,
                       theta_list: tuple()
                       ) -> float:
    return sum(np.log(sum_weighted_likelihood(pi_list, x, theta_list)))

    # %% Calculation

theta1 = [4, 1]  # µ, σ2
theta2 = [9, 1]  # µ, σ2
pi1 = 0.6
pi2 = 0.4
n = len(x)
overall_LL = overall_log_likelihood((pi1, pi2), x, (theta1, theta2))

delta_LL = 1
delta_LL_list = []
iteration = 0

while delta_LL > 1e-6:
    # calculate responsibles
    r_k1 = (weighted_likelihood(pi1, x, theta1) / 
            sum_weighted_likelihood((pi1, pi2), x, (theta1, theta2))
            )
    
    r_k2 = (weighted_likelihood(pi2, x, theta2) / 
            sum_weighted_likelihood((pi1, pi2), x, (theta1, theta2))
            )
    
    # update values
    mu1 = sum(r_k1 * x) / sum(r_k1)
    var1 = sum(r_k1 * ((x - mu1) ** 2)) / sum(r_k1)
    theta1 = [mu1, var1]
    pi1 = sum(r_k1) / n
    
    mu2 = sum(r_k2 * x) / sum(r_k2)
    var2 = sum(r_k2 * ((x - mu2) ** 2)) / sum(r_k2)
    theta2 = [mu2, var2]
    pi2 = sum(r_k2) / n
    
    overall_LL_updated = overall_log_likelihood((pi1, pi2), x, (theta1, theta2))
    delta_LL =  overall_LL_updated - overall_LL
    delta_LL_list.append(delta_LL)
    
    overall_LL = overall_LL_updated
    
    iteration += 1


r_total = r_k1 + r_k2
r_k1_normalized = r_k1 / r_total
r_k2_normalized = r_k2 / r_total
r_k_stack = np.stack((r_k1_normalized, r_k2_normalized), axis=1)

group = ['lower', 'higher']
assignment = []

for r_pair in r_k_stack:
    assign = np.random.choice(group, p = r_pair)
    assignment.append(assign)
    
df_divided = pd.DataFrame({
    'score': x,
    'group': assignment
    })

sns.histplot(data=df_divided, x='score')
sns.histplot(data=df_divided, x='score', hue='group')
plt.show()

# %% TEST

dividing_way_EM = df_divided['group'] == 'higher'
dividing_way_region = df['economic'] == 3
compare = dividing_way_EM == dividing_way_region


