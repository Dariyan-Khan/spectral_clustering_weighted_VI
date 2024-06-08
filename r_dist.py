import numpy as np
from scipy.stats import norm
from sigma_inv import sigma_inv_approx, jensen_approx
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
from copy import deepcopy
import warnings

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
        # self.norm_const = self.compute_Id(order=self.d) #normalising constant for distribution
        # self.first_moment = np.exp(np.log(self.compute_Id(order=self.d+1)) - np.log(self.norm_const))
        # self.second_moment = np.exp(np.log(self.compute_Id(order=self.d+2)) - np.log(self.norm_const))

        # self.log_norm_const = self.compute_log_Id(order=self.d) # normalising constant for distribution
        # self.first_moment = np.exp(self.compute_log_Id(order=self.d+1) - self.log_norm_const)
        # self.second_moment = np.exp(self.compute_log_Id(order=self.d+2) - self.log_norm_const)

        self.log_norm_const = self.compute_log_Id(order=self.d) # normalising constant for distribution
        self.log_first_moment = self.compute_log_I_d_ratio(order=self.d+1)
        self.log_second_moment = self.compute_log_I_d_ratio(order=self.d+2)

        self.first_moment = np.exp(self.log_first_moment)
        self.second_moment = np.exp(self.log_second_moment)

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
    

    def stable_log_diff_exp(self, A, B):
        """
        Compute log(exp(A) - exp(B)) in a numerically stable manner.
        Assumes A >= B.

        This is useful in cases where self.β is negative
        """
        if A < B:
            print(f"==>> A: {A}")
            print(f"==>> B: {B}")
            raise ValueError("Function assumes that A >= B for stability.")
            A, B = B, A
        
        # Handling the case where exp(B - A) could be numerically unstable
        diff = B - A
        if diff < -np.log(np.finfo(float).max):
            # exp(B - A) is effectively zero
            return A
        else:
            return A + np.log(1 - np.exp(diff))


    
    def compute_log_Id(self, order):
        # Compute log I_0 and log I_1

        log_I_0 = 0.5 * np.log(np.pi / self.alpha) + norm.logcdf(self.beta * np.sqrt(2 * self.alpha))

        if self.beta>=0:

            log_beta = np.log(self.beta)
            log_factor = -np.log(2 * self.alpha) - self.alpha * self.beta**2

            # Combine using logsumexp for numerical stability
            log_I_1 = logsumexp([log_beta + log_I_0, log_factor])
        
        elif self.beta<0:
            large_log_term = -np.log(2 * self.alpha) - self.alpha * self.beta**2
            small_log_term = np.log(np.abs(self.beta)) + log_I_0
            log_I_1 = self.stable_log_diff_exp(large_log_term, small_log_term)



        #log_I_1 =  np.log(self.beta * np.exp(log_I_0) + (np.exp(-self.alpha * self.beta**2) / (2 * self.alpha)) + eps)  #np.log(self.beta) + log_I_0 + np.log1p(np.exp(-self.alpha * self.beta**2 - np.log(2 * self.alpha) - log_I_0))

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

            if self.beta>=0:
                first_term = np.log(self.beta) + log_current
                second_term = log_previous + np.log(i-1) - (np.log(2) + np.log(self.alpha))
                log_next_value = logsumexp([first_term, second_term])
            
            elif self.beta<0:
                large_log_term = log_previous + np.log(i-1) - (np.log(2) + np.log(self.alpha))
                small_log_term = np.log(np.abs(self.beta)) + log_current
                # print(f"==>> large_log_term: {large_log_term}")
                # print(f"==>> small_log_term: {small_log_term}")
                log_next_value = self.stable_log_diff_exp(large_log_term, small_log_term)

            log_previous, log_current = log_current, log_next_value

        # The log_current variable now holds the value of log I_d
        return log_current
    
    
    def compute_log_I_d_ratio(self, order):
        log_norm_const = self.compute_log_Id(order=self.d)

        log_I_0_ratio = 0.5 * np.log(np.pi / self.alpha) + norm.logcdf(self.beta * np.sqrt(2 * self.alpha)) - log_norm_const

        if self.beta>=0:

            log_beta = np.log(self.beta)
            log_factor = -np.log(2 * self.alpha) - self.alpha * self.beta**2 - log_norm_const

            # Combine using logsumexp for numerical stability
            log_I_1_ratio = logsumexp([log_beta + log_I_0_ratio, log_factor])
        
        elif self.beta<0:
            large_log_term = -np.log(2 * self.alpha) - self.alpha * self.beta**2 - log_norm_const
            small_log_term = np.log(np.abs(self.beta)) + log_I_0_ratio
            log_I_1_ratio = self.stable_log_diff_exp(large_log_term, small_log_term)
        


        # If order is 0 or 1, return log I_0 or log I_1 directly
        if order == 0:
            return log_I_0_ratio
        if order == 1:
            return log_I_1_ratio

        # Initialize previous two values for the recursion
        log_ratio_previous = deepcopy(log_I_0_ratio)
        log_ratio_current = deepcopy(log_I_1_ratio)

        # Recursively compute log I_d using only the last two values

        for i in range(2, order + 1):

            if self.beta>=0:
                first_term = np.log(self.beta) + log_ratio_current
                second_term = log_ratio_previous + np.log(i-1) - (np.log(2) + np.log(self.alpha))
                log_ratio_next_value = logsumexp([first_term, second_term])
            
            elif self.beta<0:
                large_log_term = log_ratio_previous + np.log(i-1) - (np.log(2) + np.log(self.alpha))
                small_log_term = np.log(np.abs(self.beta)) + log_ratio_current
                # print(f"==>> large_log_term: {large_log_term}")
                # print(f"==>> small_log_term: {small_log_term}")
                log_ratio_next_value = self.stable_log_diff_exp(large_log_term, small_log_term)

            log_ratio_previous, log_ratio_current = log_ratio_current, log_ratio_next_value

        # The log_current variable now holds the value of log I_d
        return log_ratio_current


    
    def update_moments(self, norm_embd=None,embd=None ):
        # if self.beta<-1:
        #     self.first_moment = 0
        #     self.second_moment = 0
        #     return

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning) 

            try:
                # self.norm_const = self.compute_Id(order=self.d) #normalising constant for distribution
                # self.first_moment = np.exp(np.log(self.compute_Id(order=self.d+1)) - np.log(self.norm_const))
                # self.second_moment = np.exp(np.log(self.compute_Id(order=self.d+2)) - np.log(self.norm_const))

                # self.log_norm_const = self.compute_log_Id(order=self.d) # normalising constant for distribution
                # self.first_moment = np.exp(self.compute_log_Id(order=self.d+1) - self.log_norm_const)
                # self.second_moment = np.exp(self.compute_log_Id(order=self.d+2) - self.log_norm_const)

                self.log_norm_const = self.compute_log_Id(order=self.d) # normalising constant for distribution
                self.log_first_moment = self.compute_log_I_d_ratio(order=self.d+1)
                self.log_second_moment = self.compute_log_I_d_ratio(order=self.d+2)

                self.first_moment = np.exp(self.log_first_moment)
                self.second_moment = np.exp(self.log_second_moment)


            
            except Exception:
                print(f"==>> alpha: {self.alpha}")
                print(f"==>> beta: {self.beta}")
                print(f"==>> log_norm_const: {self.log_norm_const}")
                print(f"==>> norm_embd: {norm_embd}")
                print(f"==>> embedding: {embd}")
                # print(f"==>> group: {np.argmax(z_var.probs)}")
                assert False
     
    
    def vi(self, z_i, sigma_star_vi_list, γ_vi_list, μ_vi_list, phi_var, norm_datapoint, datapoint, real_cov=None):

        C = 0
        D = 0
        D_collection = []
        z_probs_collection = []
        norm_datapoint_collection = []
        sigma_inv_collection = []
        μ_mean_collection = []

        norm_datapoint = norm_datapoint.reshape(-1, 1)
        for k in range (0, len(z_i.probs)):
            data_group = k
            sigma = sigma_star_vi_list[data_group]
            γ = γ_vi_list[data_group]
            μ = μ_vi_list[data_group]


            #cov_0 = real_cov
            # sigma_inv = np.linalg.inv(cov_0)
            sigma_inv = jensen_approx(sigma, γ)
            sigma_inv = np.reshape(sigma_inv, (self.d+1, self.d+1))
            # sigma_inv = sigma_inv_approx(sigma, γ, α=0.01)

            C += z_i.probs[k] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), norm_datapoint)

            
            D_value = z_i.probs[k] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), μ.mean)

            D_collection.append(z_i.probs[k] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), μ.mean))
            z_probs_collection.append(z_i.probs[k])
            norm_datapoint_collection.append(norm_datapoint)
            sigma_inv_collection.append(sigma_inv)
            μ_mean_collection.append(μ.mean)

            D_value = D_value.reshape(-1)
            D += D_value
        
        C = np.reshape(C, -1)
        D = np.reshape(D, -1)

        if D < 0:
            # print()
            print(f"==>> z_i.index: {z_i.index}")
            # print(f"==>> D_collection: {D_collection}")
            # print(f"==>> z_probs_collection: {z_probs_collection}")
            # print(f"==>> norm_datapoint_collection: {norm_datapoint_collection}")
            # print(f"==>> sigma_inv_collection: {sigma_inv_collection}")
            # print(f"==>> μ_mean_collection: {μ_mean_collection}")
            # print()
            ##assert False

        
        self.alpha = C / 2
        self.beta = D / C

        self.alpha = min(np.array([20.0]), self.alpha)

        self.update_moments(norm_datapoint, datapoint )


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
    d=5
    r_dist = R(alpha=5.5, beta=-5.113712833, d=d)
    first_moment = np.exp(r_dist.compute_log_Id(d+1) - r_dist.compute_log_Id(d))
    second_moment = np.exp(r_dist.compute_log_Id(d+2) - r_dist.compute_log_Id(d))
    print(f"==>> first_moment: {first_moment}")
    print(f"==>> second_moment: {second_moment}")

    first_moment_ratio = np.exp(r_dist.compute_log_I_d_ratio(d+1))
    second_moment_ratio = np.exp(r_dist.compute_log_I_d_ratio(d+2))
    print(f"==>> first_moment_ratio: {first_moment_ratio}")
    print(f"==>> second_moment_ratio: {second_moment_ratio}")