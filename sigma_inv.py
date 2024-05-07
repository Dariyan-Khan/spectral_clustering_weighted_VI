import numpy as np

def sigma_inv_approx(sigma_star, γ, α=1): # α is the term added for convergence purposes

    d = sigma_star.d

    """
    Each matrix will be of the form
    | A B|
    | B C|
    """

    def Sigma_expectation(sigma_star, γ, α=1):
        first_moment = sigma_star.first_mom()

        A = first_moment + γ.cov + γ.mean @ γ.mean.T
        B = np.expand_dims(γ.mean,axis=1)
        C = 1.0

        # print(
        #     f"""
        #     A:{A.shape},
        #     B:{B.shape},
        #     """
        # )

        return np.block([[A, B], [B.T, C]])


    def Sigma_sq_expectation(sigma_star, γ):
        first_moment = sigma_star.first_mom()
        second_moment = sigma_star.second_mom()

        γ_outer_product = γ.outer_prod()
        γ_triple = γ.three_gamma()
        γ_quad = γ.quadruple_gamma()

        A = second_moment + (first_moment @ γ_outer_product) + \
            (γ_outer_product @ first_moment) + γ_quad
        
        B = np.expand_dims((first_moment @ γ.mean) + γ_triple + γ.mean, axis=1)

        C = np.trace(γ_outer_product) + 1.0

        return np.block([[A, B], [B.T, C]])
    
    

    return 3 * α * np.eye(d) - 3 * α**2 * Sigma_expectation(sigma_star, γ) + α**2 * Sigma_sq_expectation(sigma_star, γ)





         