import numpy as np
import pandas as pd

# %% EXPECTATION MAXIMIZATION
class ExpectationMaximization:
    """
    Goal:
    ---
        Input: an array & number of distribution
        Output: a dataframe of group of distribution
        
    """
    
    def __init__(self, array, K, epsilon):
        # Input random variables
        # [
        # [x1, x2, ..., xn]
        # ]
        self.x = np.array(array).reshape(1, -1)
        self.n = len(self.x[0])
        
        # Converging condition
        self.epsilon = epsilon
        
        # Number of groups
        self.K = K
        
        # Initiate holders for [µ, σ2, π] in K distribution (K rows)
        # [µ, σ2, π]1
        # [µ, σ2, π]2
        # ...
        # [µ, σ2, π]K
        self.param_sets = np.zeros([K, 3])
        self.activate_initial_values()
        
        # Initiate holders for responsible
        # [r1, r2, ..., rn]1
        # [r1, r2, ..., rn]2
        # ...
        # [r1, r2, ..., rn]K
        self.res_sets = np.zeros([K, self.n])
        
        # Initiate holder for overall log-likelihood
        self.overall_LL = []
        self.delta_overall_LL = 11
        
    def print_basic_info(self):
        mean = np.mean(self.x[0])
        var = np.var(self.x[0])
        
        print(f"Number of student: {self.n}")
        print(f"Mean: {mean:.2f}")
        print(f"Variance: {var:.2f}")
        
        
        
    def activate_initial_values(self):
        """
        Guess initial values for [µ, σ2, π] for each set
        """
        
        part_len = int(self.n / self.K)
        
        for k in range(self.K):
            x_part = self.x[0, k * part_len : (k+1) * part_len]
            
            mu = np.mean(x_part)
            var = np.var(x_part) / self.K
            pi = 1 / self.K
                      
            self.param_sets[k] = [mu, var, pi]
            
    def update_res(self):
        mu = self.param_sets[:, 0].reshape(-1, 1)  # mu = [[µ1], [µ2]]
        var = self.param_sets[:, 1].reshape(-1, 1)
        pi = self.param_sets[:, 2].reshape(-1, 1)
        
        
        L = (1 / np.sqrt(2 * np.pi * var) * 
             np.exp(- (self.x - mu)**2 / (2 * var))
             )
        
        weighted_L = pi * L  # res_numerator
        
        res_denomerator = np.sum(weighted_L, axis=0)
        
        # Update responsibility
        self.res_sets = weighted_L / res_denomerator
        
        # Add current overall log-likelihood        
        self.overall_LL.append(sum(np.log(res_denomerator)))
        
            
    def update_param_sets(self):
        self.param_sets[:, 0] = (
            np.sum(self.res_sets * self.x, axis = 1 ) /
            np.sum(self.res_sets, axis=1 )
            )
        
        self.param_sets[:, 1] = (
            np.sum(self.res_sets * ((self.x - self.param_sets[:, 0].reshape(-1, 1))) ** 2, 
                   axis=1) / 
            np.sum(self.res_sets, 
                   axis=1)
            )
        
        self.param_sets[:, 2] = (
            np.sum(self.res_sets, axis=1) / self.n
            )
        
    
    def update_delta_overall_LL(self):
        if len(self.overall_LL) >= 2:
            self.delta_overall_LL = self.overall_LL[-1] - self.overall_LL[-2]

    def run(self):
        
        iteration = 0

        while self.delta_overall_LL >= self.epsilon:
            self.update_res()
            self.update_param_sets()
            self.update_delta_overall_LL()
            
            iteration += 1
            
            if iteration > 200:
                break
            
    def assign_group(self):
        group = np.array(range(self.K))
        assignment = []
        
        # normalize responsibility
        res_total = np.sum(self.res_sets, axis=0)
        res_normalized = self.res_sets / res_total
        
        for i in range(self.n):
            assign = np.random.choice(group, p = res_normalized[:, i])
            assignment.append(assign)
            
        df_divided = pd.DataFrame({
            'score': self.x[0],
            'group': assignment
            })
        
        return df_divided
        
# %% WELCH'S T TEST
class WelchTTest:
    def __init__(self, array1, array2):
        self.array1 = np.array(array1)
        self.array2 = np.array(array2)
        
        self.n1 = len(self.array1)
        self.mu1 = self.array1.mean()
        self.sd1 = self.array1.std(ddof=1)
        
        self.n2 = len(self.array2)
        self.mu2 = self.array2.mean()
        self.sd2 = self.array2.std(ddof=1)
        
    def get_df(self):
        degree_freedom = (
            ((self.sd1 ** 2 / self.n1) + (self.sd2 ** 2 / self.n2)) ** 2 /
            ((self.sd1 ** 2 / self.n1)**2/(self.n1-1) + (self.sd2 ** 2 / self.n2)**2/(self.n2-1))
            )
        
        return degree_freedom
    
    def get_T(self):
        T = ((self.mu1 - self.mu2) / 
             np.sqrt(self.sd1 ** 2 / self.n1 + self.sd2 ** 2 / self.n2)
             )
        
        return T
        
         
