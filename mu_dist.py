import numpy as np
from sigma_inv import sigma_inv_approx

class mu():

    def __init__(self, prior_mean, prior_cov, k, d):
        self.prior_mean = prior_mean # (mean, covariance matrix)
        self.prior_cov = prior_cov
        self.k = k
        self.d = d
        self.mean = None
        self.cov = None
    
    def vi(self, phi_vi_list, r_vi_list, sigma_star_k, γ_k, datapoints):

        mean_vec = np.zeros(self.d)
        cov_mat = np.zeros((self.d, self.d))

        for (i, data) in enumerate(datapoints.normalised):
            phi = phi_vi_list[i]
            mean_vec += r_vi_list[i].first_moment * phi[self.k] * data
        
        cov_mat = np.linalg.inv(sigma_inv_approx(sigma_star_k, γ_k))

        self.mean = mean_vec
        self.cov = cov_mat


    