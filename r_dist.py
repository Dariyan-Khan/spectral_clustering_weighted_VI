import numpy as np
from scipy.stats import norm
from sigma_inv import sigma_inv_approx, jensen_approx
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
from copy import deepcopy

class R():

    def __init__(self, d, alpha=None, beta=None):

        if alpha is None:
            self.alpha = np.random.uniform(2, 5) # 2
        
        else:
            self.alpha = alpha
        
        if beta is None:
            self.beta = np.random.uniform(2, 5) # 2
        
        else:
            self.beta = beta

        self.d = d
        self.norm_const = self.compute_Id(order=self.d) #normalising constant for distribution
        self.first_moment = np.exp(np.log(self.compute_Id(order=self.d+1)) - np.log(self.norm_const))
        self.second_moment = np.exp(np.log(self.compute_Id(order=self.d+2)) - np.log(self.norm_const))

        # self.log_norm_const = self.compute_log_Id(order=self.d) # normalising constant for distribution
        # self.first_moment = np.exp(self.compute_log_Id(order=self.d+1) - self.log_norm_const)
        # self.second_moment = np.exp(self.compute_log_Id(order=self.d+2) - self.log_norm_const)

        self.pdf = lambda x: ((x**(self.d)) * np.exp(-self.alpha * (x - self.beta)**2)) /  self.norm_const

    
    def compute_Id(self, order):
        # Compute I_0 and I_1 which are needed for starting the recursion
        I_0 = np.sqrt(np.pi / self.alpha) * norm.cdf(self.beta * np.sqrt(2 * self.alpha))
        I_1 = self.beta * I_0 + np.exp(-self.alpha * self.beta**2) / (2 * self.alpha)
        
        # If d is 0 or 1, we can return I_0 or I_1 directly
        if order == 0:
            return I_0
        if order == 1:
            return I_1
        
        # Initialize previous two values for the recursion
        previous = deepcopy(I_0)
        current = deepcopy(I_1)
        
        # Recursively compute I_d using only the last two values
        for i in range(2, order + 1):
            next_value = (self.beta * current + previous * (i - 1) / (2 * self.alpha))
            previous, current = current, next_value
        
        # The current variable now holds the value of I_d
        return current
    
    def compute_log_Id(self, order):
        # Compute log I_0 and log I_1

        # I_0 = np.sqrt(np.pi / self.alpha) * norm.cdf(self.beta * np.sqrt(2 * self.alpha))
        # I_1 = self.beta * I_0 + np.exp(-self.alpha * self.beta**2) / (2 * self.alpha)
        eps = 1e-4

        log_I_0 = 0.5 * np.log(np.pi / self.alpha) + norm.logcdf(self.beta * np.sqrt(2 * self.alpha))
        log_I_1 =  np.log(self.beta * np.exp(log_I_0) + (np.exp(-self.alpha * self.beta**2) / (2 * self.alpha)) + eps)  #np.log(self.beta) + log_I_0 + np.log1p(np.exp(-self.alpha * self.beta**2 - np.log(2 * self.alpha) - log_I_0))

        # If order is 0 or 1, return log I_0 or log I_1 directly
        if order == 0:
            return log_I_0
        if order == 1:
            return log_I_1

        # Initialize previous two values for the recursion
        log_previous = deepcopy(log_I_0)
        log_current = deepcopy(log_I_1)

        # Recursively compute log I_d using only the last two values
        for i in range(2, order + 1):
            #log_term1 = np.log(self.beta * np.exp(log_current))
            # log_term2 = log_previous + np.log((i - 1) / (2 * self.alpha) + eps)
            # # log_next_value = logsumexp([log_term1, log_term2])
            # new_inner_log = self.beta * np.exp(log_current) + np.exp(log_term2) + eps

            # if new_inner_log<0:
            #     return NotImplementedError

            #log_next_value = np.log(self.beta * np.exp(log_current) + np.exp(log_term2) + eps)
            print(f"inner log: {self.beta * np.exp(log_current) + np.exp(log_previous) * (i - 1) / (2 * self.alpha) + eps}")
            log_next_value = np.log(self.beta * np.exp(log_current) + np.exp(log_previous) * (i - 1) / (2 * self.alpha) + eps)

            log_previous, log_current = log_current, log_next_value

        # The log_current variable now holds the value of log I_d
        return log_current
    
    def update_moments(self, norm_embd):
        try:
            self.norm_const = self.compute_Id(order=self.d) #normalising constant for distribution
            self.first_moment = np.exp(np.log(self.compute_Id(order=self.d+1)) - np.log(self.norm_const))
            self.second_moment = np.exp(np.log(self.compute_Id(order=self.d+2)) - np.log(self.norm_const))
        
        except Exception:
            print(f"==>> norm_embd: {norm_embd}")
            assert False
     
    
    def vi(self, z_i, sigma_star_vi_list, γ_vi_list, μ_vi_list, phi_var, norm_datapoint, real_cov=None):

        C = 0
        D = 0

        norm_datapoint = norm_datapoint.reshape(-1, 1)
        for k in range (0, len(z_i.probs)):
            data_group = k
            sigma = sigma_star_vi_list[data_group]
            γ = γ_vi_list[data_group]
            μ = μ_vi_list[data_group]


            #cov_0 = real_cov
            # sigma_inv = np.linalg.inv(cov_0)
            sigma_inv = jensen_approx(sigma, γ)
            # sigma_inv = sigma_inv_approx(sigma, γ, α=0.01)

            C += z_i.probs[k] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), norm_datapoint)

            D_value = z_i.probs[k] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), μ.mean)
            D_value = D_value.reshape(-1)
            D += D_value
        
        C = np.reshape(C, -1)
        D = np.reshape(D, -1)

        
        self.alpha = C/2
        self.beta = D / C

        self.update_moments(norm_datapoint)


    def function_to_maximize(self, r):
        return (r**(self.d)) * np.exp(-self.alpha * (r - self.beta)**2)

    def MLE(self):
        # We want to maximize function_to_maximize, so we minimize the negative of it
        objective = lambda r: -self.function_to_maximize(r)
        
        # Use minimize_scalar to find the maximum
        result = minimize_scalar(objective, bounds=(0, self.beta+20), method='bounded')
        
        if result.success:
            max_r = result.x
            max_value = -result.fun
            return max_r, max_value
        else:
            raise ValueError("Optimization failed")


if __name__ == "__main__":
    r_dist = R(alpha=4.0, beta=5.8, d=3)
    # print(r_dist.compute_Id(0))
    # print(np.exp(r_dist.compute_log_Id(0)))
    print(r_dist.MLE())