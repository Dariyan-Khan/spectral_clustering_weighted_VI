import numpy as np
from sigma_inv import sigma_inv_approx

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

    

    def vi(self, z_vi_list, r_vi_list, sigma_star_k, γ_k, datapoints):

        n_k = 0
        B = 0

    

        for (i, data) in enumerate(datapoints.normed_embds):
            z = z_vi_list[i]

            n_k += z.probs[self.k]
            B += r_vi_list[i].first_moment * z.probs[self.k] * data

        # print("curr sigma_star_k", sigma_star_k)
                
        A = sigma_inv_approx(sigma_star_k, γ_k)*n_k + np.linalg.inv(self.prior_cov)
        A_inv = np.linalg.inv(A)
    

        self.mean = A_inv @ B 
        self.cov = A_inv


    