import numpy as np
from sigma_inv import sigma_inv_approx, sigma_expectation, jensen_approx
from scipy.special import digamma

class Z():
    
    def __init__(self, d, K, index, prior=None):

        if prior is None:
            self.prior_probs = [1/K for _ in range(K)]
            
        else:
            self.prior_probs = prior #initialise with random probabilities

        self.K = K # number of classes
        self.probs = [1/K for _ in range(K)]
        self.d = d
        self.index = index

    
    def vi(self, r_i, μ_list, sigma_star_list, γ_list, norm_datapoint, phi, weights, verbose=False, real_cov=None):

        log_probs = np.array([1/self.K for _ in range(self.K)])

        for k in range(self.K):
            μ = μ_list[k]
            sigma_star = sigma_star_list[k]
            γ = γ_list[k]

            # Using the expectation of trace as an upper bound
            # P_k_1 = -0.5 * (
            #     np.trace(
            #         (sigma_star.scale /(sigma_star.dof - self.d)) + \
            #         γ.cov + \
            #         np.outer(γ.mean, γ.mean)   
            #     ) - \
            #     self.d + sigma_star.nu
            # )


            # Using log of det of expecation
            P_k_1 = -0.5 * np.log(np.linalg.det(sigma_expectation(sigma_star, γ, ν=sigma_star.nu)))

            # cov = real_cov

            Sigma_inv = jensen_approx(sigma_star, γ) #np.linalg.inv(cov)
            Sigma_inv = np.reshape(Sigma_inv, (self.d, self.d))

            norm_datapoint = norm_datapoint.reshape(-1, 1)

            P_k_2 = -0.5 * (
                r_i.second_moment * np.matmul(norm_datapoint.T, np.matmul(Sigma_inv, norm_datapoint)) - \
                2 * r_i.first_moment * np.matmul(norm_datapoint.T, np.matmul(Sigma_inv, μ.mean)) + \
                np.trace(np.matmul( np.outer(μ.mean, μ.mean) + μ.cov, Sigma_inv))
            )
            
            P_k = P_k_1 + P_k_2

            log_probs[k] = P_k + digamma(phi.conc[k]) # np.log(phi.conc[k])


        new_probs = np.exp(log_probs - np.logaddexp.reduce(log_probs))

        new_probs = new_probs.reshape(-1,1)

        self.probs = new_probs

