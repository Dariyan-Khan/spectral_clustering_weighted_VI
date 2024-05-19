import numpy as np
from scipy.stats import norm
from sigma_inv import sigma_inv_approx

class R():

    def __init__(self, d, α=None, β=None):

        if α is None:
            self.α = 2
        
        else:
            self.α = α
        
        if β is None:
            self.β = 2
        
        else:
            self.β = β

        self.d = d
        self.norm_const = self.compute_Id(order=self.d) #normalising constant for distribution
        self.first_moment = self.compute_Id(order=self.d+1) / self.norm_const
        self.second_moment = self.compute_Id(order=self.d+2) / self.norm_const
    
    def compute_Id(self, order):
        # Compute I_0 and I_1 which are needed for starting the recursion
        I_0 = np.sqrt(np.pi / self.α) * norm.cdf(self.β * np.sqrt(2 * self.α))
        I_1 = self.β * I_0 + np.exp(-self.α * self.β**2 / (2 * self.α)) / (2 * self.α)
        
        # If d is 0 or 1, we can return I_0 or I_1 directly
        if order == 0:
            return I_0
        if order == 1:
            return I_1
        
        # Initialize previous two values for the recursion
        previous = I_0
        current = I_1
        
        # Recursively compute I_d using only the last two values
        for i in range(2, order + 1):
            next_value = (self.β * current + previous * (i - 1) / (2 * self.α))
            previous, current = current, next_value
        
        # The current variable now holds the value of I_d
        return current
    
    def vi(self, z_i, sigma_star_vi_list, γ_vi_list, μ_vi_list, norm_datapoint):

        data_group = np.argmax(z_i.probs)
        sigma = sigma_star_vi_list[data_group]
        γ = γ_vi_list[data_group]
        μ = μ_vi_list[data_group]

        sigma_inv = sigma_inv_approx(sigma, γ, α=sigma.nu)

        self.α = norm_datapoint.T @ sigma_inv @ norm_datapoint / 2
        self.β = (norm_datapoint.T @ sigma_inv @ μ.mean) / (norm_datapoint.T @ sigma_inv @ norm_datapoint)

        self.norm_const = self.compute_Id(order=self.d) #normalising constant for distribution
        self.first_moment = self.compute_Id(order=self.d+1) / self.norm_const
        self.second_moment = self.compute_Id(order=self.d+2) / self.norm_const



if __name__ == "__main__":
    r_dist = r(α=1, β=7, d=3)
    print(r_dist.first_moment)
    print(r_dist.second_moment)

