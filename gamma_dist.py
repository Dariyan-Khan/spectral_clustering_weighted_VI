import numpy as np


class mu():

    def __init__(self, prior_mean, prior_cov, k, d):
        self.prior_mean = prior_mean # (mean, covariance matrix)
        self.prior_cov = prior_cov
        self.k = k
        self.d = d
        self.dim = d - 1
        self.mean = None
        self.cov = None
    
    def vi(self, phi_vi_list, r_vi_list, sigma_star_k, μ_k, datapoints):

        mean_vec = np.zeros(self.dim)
        cov_mat = np.zeros((self.dim, self.dim))
        cov_mat_inner = np.zeros((self.dim, self.dim))
        
        for (i, data) in enumerate(datapoints.normalised):
            phi = phi_vi_list[i]

            cov_mat_inner += phi[self.k] * (
                r_vi_list[i].second_moment * data[self.d]**2 - \
                2 * r_vi_list[i].first_moment * data[self.d]*μ_k[self.d] + \
                μ_k.mean[self.d]**2 + \
                μ_k.cov[self.d, self.d]
            )

            mean_vec += phi[self.k] * (
                r_vi_list[i].second_moment * data[self.d] * data[:self.d] - \
                r_vi_list[i].first_moment * data[self.d] * μ_k.mean[:self.d] - \
                r_vi_list[i].first_moment * data[:self.d] * μ_k.mean[self.d] + \
                μ_k.mean[:self.d] * μ_k.mean[self.d] + \
            )



        cov_mat = sigma_star_k.dof * np.linalg.inv(sigma_star_k.scale) @ cov_mat_inner
        cov_mat += np.linalg.inv(self.prior_cov)

        self.cov = np.linalg.inv(cov_mat)

        mean_vec = sigma_star_k.dof * np.linalg.inv(sigma_star_k.scale) @ mean_vec

        self.mean = self.cov @ mean_vec






