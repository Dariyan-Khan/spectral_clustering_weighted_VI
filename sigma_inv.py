import numpy as np

def sigma_inv_approx(sigma_star, γ, α=1): # α is the term added for convergence purposes

    d = sigma_star.d

    """
    Each matrix will be of the form
    | A B|
    | B C|
    """

    def Sigma_expectation(sigma_star, γ, α=1):

        A = sigma_star.first_moment() + γ.cov + γ.mean @ γ.mean.T
        B = γ.mean
        C = 1.0

        return np.block([[A, B], [B.T, C]])


    def Sigma_sq_expectation(sigma_star, γ):

        A = sigma_star.second_moment + sigma_star.first_moment @ γ.outer_product + \
            γ.outer_product @ sigma_star.first_moment + γ.quad_gamma
        
        B = sigma_star.first_moment @ γ.mean + γ.triple_gamma + γ.mean

        C = np.trace(γ.outer_product) + 1.0

        return np.block([[A, B], [B.T, C]])
    
    

    return 3 * α * np.eye(d) - 3 * α**2 * Sigma_expectation(sigma_star, γ) + α**2 * Sigma_sq_expectation(sigma_star, γ)





         