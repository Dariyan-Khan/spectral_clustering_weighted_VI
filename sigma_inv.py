import numpy as np

def sigma_inv_approx(sigma_star, γ):

    d = sigma_star.d

    """
    Each matrix will be of the form
    | A B|
    | B C|
    """

    def Sigma_expectation(sigma_star, γ):

        A = sigma_star.first_moment() + γ.cov + γ.mean @ γ.mean.T
        B = γ.mean
        C = 1.0

        return np.block([[A, B], [B.T, C]])


    def Sigma_sq_expectation_(sigma_star, γ):

        
         