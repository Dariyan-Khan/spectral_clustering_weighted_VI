import numpy as np
from sigma_inv import sigma_inv_approx
from scipy.special import digamma

class Z():
    
    def __init__(self, d, K, prior=None):

        if prior is None:
            self.prior_probs = [1/K for _ in range(K)]
            
        else:
            self.prior_probs = prior #initialise with random probabilities

        self.K = K # number of classes
        self.probs = [1/K for _ in range(K)]
        self.d = d

    
    def vi(self, r_i, μ_list, sigma_star_list, γ_list, norm_datapoint, phi):

        for k in range(self.K):
            μ = μ_list[k]
            sigma_star = sigma_star_list[k]
            γ = γ_list[k]

            P_k_1 = -0.5 * (
                np.trace(
                    (sigma_star.scale /sigma_star.dof - self.d) + \
                    γ.cov + \
                    γ.mean @ γ.mean.T   
                ) - \
                self.d 
            )

            Sigma_inv = sigma_inv_approx(sigma_star, γ)

            P_k_2 = -0.5 * (
                r_i.second_moment * norm_datapoint.T @ Sigma_inv @ norm_datapoint - \
                2 * r_i.first_moment * norm_datapoint.T @ Sigma_inv @ μ.mean + \
                np.trace((μ.mean @ μ.mean.T + μ.cov) @ Sigma_inv)
            )
            
            P_k = P_k_1 + P_k_2

            self.probs[k] = np.exp(P_k) * digamma(phi.conc[k]) / digamma(sum(phi.conc))

        
        self.probs = self.probs / sum(self.probs)

            
        
