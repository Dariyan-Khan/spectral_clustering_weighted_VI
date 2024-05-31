import numpy as np
from scipy.stats import norm
from sigma_inv import sigma_inv_approx

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
        # self.first_moment = self.compute_Id(order=self.d+1) / self.norm_const
        # self.second_moment = self.compute_Id(order=self.d+2) / self.norm_const

        self.norm_const = self.compute_Id(order=self.d) #normalising constant for distribution
        self.first_moment = self.compute_Id(order=self.d+1) / self.norm_const
        self.second_moment = self.compute_Id(order=self.d+2) / self.norm_const

        # print(f"the d used in R: {self.d}")
        # print(f"the alpha used in R: {self.α}")
        # print(f"the beta used in R: {self.β}")
        # print(f"the norm constant: {self.norm_const}")
        # print(f"self.compute_Id(order=self.d+1): {self.compute_Id(order=self.d+1)}")
        # print(f"self.compute_Id(order=self.d+2): {self.compute_Id(order=self.d+2)}")
        # print(f"the first_moment used in R: {self.first_moment}")
        # print(f"the second_moment used in R: {self.second_moment}")

        # assert False
    
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
        previous = I_0
        current = I_1
        
        # Recursively compute I_d using only the last two values
        for i in range(2, order + 1):
            next_value = (self.beta * current + previous * (i - 1) / (2 * self.alpha))
            previous, current = current, next_value
        
        # The current variable now holds the value of I_d
        return current
    
    def update_moments(self):
        self.norm_const = self.compute_Id(order=self.d) # normalising constant for distribution
        self.first_moment = self.compute_Id(order=self.d+1) / self.norm_const
        self.second_moment = self.compute_Id(order=self.d+2) / self.norm_const
    
    # def vi(self, z_i, sigma_star_vi_list, γ_vi_list, μ_vi_list, norm_datapoint):

    #     data_group = np.argmax(z_i.probs)
    #     sigma = sigma_star_vi_list[data_group]
    #     γ = γ_vi_list[data_group]
    #     μ = μ_vi_list[data_group]

    #     sigma_inv = sigma_inv_approx(sigma, γ, α=sigma.nu)

    #     norm_datapoint = norm_datapoint.reshape(-1, 1)

    #     self.α = np.matmul(np.matmul(norm_datapoint.T, sigma_inv), norm_datapoint) / 2
    #     self.β = np.matmul(np.matmul(norm_datapoint.T, sigma_inv), μ.mean) / np.matmul(np.matmul(norm_datapoint.T, sigma_inv), norm_datapoint)

    #     # self.α = norm_datapoint.T @ sigma_inv @ norm_datapoint / 2
    #     # self.β = (norm_datapoint.T @ sigma_inv @ μ.mean) / (norm_datapoint.T @ sigma_inv @ norm_datapoint)

    #     self.norm_const = self.compute_Id(order=self.d) #normalising constant for distribution
    #     self.first_moment = self.compute_Id(order=self.d+1) / self.norm_const
    #     self.second_moment = self.compute_Id(order=self.d+2) / self.norm_const
    
    def vi(self, z_i, sigma_star_vi_list, γ_vi_list, μ_vi_list, norm_datapoint):

        new_α = 0
        new_β = 0

        norm_datapoint = norm_datapoint.reshape(-1, 1)

        for k in range (0, len(z_i.probs)):
            data_group = k
            sigma = sigma_star_vi_list[data_group]
            γ = γ_vi_list[data_group]
            μ = μ_vi_list[data_group]

            sigma_inv = sigma_inv_approx(sigma, γ, α=sigma.nu)

            

            # self.α = np.matmul(np.matmul(norm_datapoint.T, sigma_inv), norm_datapoint) / 2
            # self.β = np.matmul(np.matmul(norm_datapoint.T, sigma_inv), μ.mean) / np.matmul(np.matmul(norm_datapoint.T, sigma_inv), norm_datapoint)

            new_α +=  z_i.probs[data_group] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), norm_datapoint) / 2
            new_β += z_i.probs[data_group] * (np.matmul(np.matmul(norm_datapoint.T, sigma_inv), μ.mean) / np.matmul(np.matmul(norm_datapoint.T, sigma_inv), norm_datapoint))

            # self.α = norm_datapoint.T @ sigma_inv @ norm_datapoint / 2
            # self.β = (norm_datapoint.T @ sigma_inv @ μ.mean) / (norm_datapoint.T @ sigma_inv @ norm_datapoint)

        
        self.alpha = new_α
        self.beta = new_β

        self.norm_const = self.compute_Id(order=self.d) #normalising constant for distribution
        self.first_moment = self.compute_Id(order=self.d+1) / self.norm_const
        self.second_moment = self.compute_Id(order=self.d+2) / self.norm_const



if __name__ == "__main__":
    r_dist = R(α=1, β=7, d=3)
    print(r_dist.first_moment)
    print(r_dist.second_moment)

