import numpy as np
from sigma_inv import sigma_inv_approx, jensen_approx

class Mu():

    def __init__(self, k, d, prior_mean=None, prior_cov=None):
        self.k = k
        self.d = d
        self.mean = None
        self.cov = None

        if prior_mean is not None:
            self.mean = prior_mean
        
        else:
            self.prior_mean = np.zeros(self.d)
        
        if prior_cov is not None:
            self.prior_cov = prior_cov
        
        else:
            self.prior_cov = np.eye(self.d)
    

    def is_positive_semi_definite(self, A):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    

    def vi(self, z_vi_list, r_vi_list, sigma_star_k, γ_k, phi_var, weights, datapoints, real_cov=None):

        n_k = 0
        B = 0

        for (i, data) in enumerate(datapoints.normed_embds):

            z = z_vi_list[i]
            n_k += weights[i] * z.probs[self.k] #phi_var.conc[self.k
            B += weights[i] * r_vi_list[i].first_moment * z.probs[self.k] * data


        sigma_inv_estimate = jensen_approx(sigma_star_k, γ_k) 

        # print("sigma_inv_estimate", sigma_inv_estimate)

        sigma_inv_estimate = np.reshape(sigma_inv_estimate, (self.d, self.d))

        # print("sigma_inv_estimate", sigma_inv_estimate)



        B = np.matmul(sigma_inv_estimate, B.T)
                
        A = sigma_inv_estimate*n_k + np.linalg.inv(self.prior_cov)

        # print(f"==>> sigma_inv_estimate*n_k: {sigma_inv_estimate*n_k}")

        # print(f"==>> np.linalg.inv(self.prior_cov): {np.linalg.inv(self.prior_cov)}")

        # print("A", A)
        # print(f"==>> self.is_positive_semi_definite(A): {self.is_positive_semi_definite(A)}")

        A_inv = np.linalg.inv(A)

        # print(f"==>> self.is_positive_semi_definite(A): {self.is_positive_semi_definite(A_inv)}")

        self.mean = np.matmul(A_inv, B)
        self.cov = A_inv





    