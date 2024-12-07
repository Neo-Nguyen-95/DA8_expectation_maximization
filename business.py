import numpy as np

# %% CLASS
class ExpectationMaximization:
    """
    Goal:
    ---
        Input: an array & number of distribution
        Output: a dataframe of group of distribution
        
    """
    
    def __init__(self, array, K, epsilon):
        # Input random variables
        self.x = np.array(array)
        self.n = len(self.x)
        
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
        self.delta_overall_LL = 1
        
        
    def activate_initial_values(self):
        """
        Guess initial values for [µ, σ2, π] for each set
        """
        
        part_len = int(self.n / self.K)
        
        for k in range(self.K):
            x_part = self.x[k * part_len : (k+1) * part_len]
            
            mu = np.mean(x_part)
            var = np.var(x_part) / self.K
            pi = 1 / self.K
                      
            self.param_sets[k] = [mu, var, pi]
            
    def update_res(self):
        res_numerator_sets = np.zeros([self.K, self.n])
        res_denomerator = np.zeros(self.n)
        
        for k in range(self.K):
            mu = self.param_sets[k][0]
            var = self.param_sets[k][1]
            pi = self.param_sets[k][2]
            
            L = (1 / np.sqrt(2 * np.pi * var) * 
                 np.exp(- (self.x - mu)**2 / (2 * var))
                 )
            
            weighted_L = pi * L
            
            res_numerator_sets[k] = weighted_L
            
            res_denomerator += weighted_L
            
        self.res_sets = res_numerator_sets / res_denomerator
        self.overall_LL.append(sum(np.log(res_denomerator)))
            
    def update_param_sets(self):
        for k in range(self.K):
            # Update mu
            self.param_sets[k][0] = (
                sum(self.res_sets[k] * self.x) /
                sum(self.res_sets[k])
                )
            
            # Update var
            self.param_sets[k][1] = (
                sum(self.res_sets[k] * ((self.x - self.param_sets[k][0])) ** 2) / 
                sum(self.res_sets[k])
                )
            
            # Update pi
            self.param_sets[k][2] = (
                sum(self.res_sets[k]) /
                self.n
                )
    
    def update_delta_overall_LL(self):
        if len(self.overall_LL) >= 2:
            self.delta_overall_LL = self.overall_LL[-1] - self.overall_LL[-2]

    def run(self):

        while self.delta_overall_LL >= self.epsilon:
            self.update_res()
            self.update_param_sets()
            self.update_delta_overall_LL()
            
            if len(self.overall_LL) > 200:
                break
            
        
            
        





        