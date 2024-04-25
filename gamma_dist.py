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
        self.outer_prod = None
    
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
                μ_k.mean[:self.d] * μ_k.mean[self.d]
            )



        cov_mat = sigma_star_k.dof * np.linalg.inv(sigma_star_k.scale) @ cov_mat_inner
        cov_mat += np.linalg.inv(self.prior_cov)

        self.cov = np.linalg.inv(cov_mat)

        mean_vec = sigma_star_k.dof * np.linalg.inv(sigma_star_k.scale) @ mean_vec

        self.mean = self.cov @ mean_vec

        self.outer_prod = self.outer_prod()

        self.corr = self.cov / np.sqrt(np.outer(np.diag(self.cov), np.diag(self.cov)))

        self.triple_gamma = self.three_gama()
    

    def outer_prod(self):
        return self.cov + self.mean @ self.mean.T


    def three_gamma(self):
        col_vars = np.tile(np.diag(self.cov).reshape(1, self.dim), (self.dim, 1))
        A_mat  = self.cov / col_vars

        np.fill_diagonal(A_mat, 0)

        M_2_vec = np.diag(self.cov) + self.mean**2
        M_3_vec = self.mean**3 + 3 * self.mean * np.diag(self.cov)

        row_mu = np.tile(self.mean.reshape(1,self.dim), (self.dim,1))
        col_mu = row_mu.T

        B_mat = row_mu - col_mu*A_mat

        np.fill_diagonal(B_mat, 0)

        return M_3_vec  + (A_mat @ M_3_vec) + (B_mat @ M_2_vec)


    def quadruple_gamma(self):
        quad_mat =  np.zeros((self.dim, self.dim))

        col_vars = np.tile(np.diag(self.cov).reshape(1, self.dim), (self.dim, 1))
        row_vars = col_vars.T

        A_mat  = self.cov / col_vars
        np.fill_diagonal(A_mat, 0)

        A_sq_mat = A_mat**2

        row_mu = np.tile(self.mean.reshape(1,self.dim), (self.dim,1))
        col_mu = row_mu.T

        B_mat = 2 * A_mat * (row_mu - col_mu*A_mat)
        np.fill_diagonal(B_mat, 0)

        C_mat = (row_mu - A_mat * col_mu)**2 + row_vars * (1-self.corr**2)
        np.fill_diagonal(C_mat, 0)

        M_2_vec = np.diag(self.cov) + self.mean**2
        M_3_vec = self.mean**3 + 3 * self.mean * np.diag(self.cov)
        M_4_vec = self.mean**4 + 6 * self.mean**2 * np.diag(self.cov) + 3 * np.diag(self.cov)**2

        M_4_vec + 4 * (A_sq_mat @ M_4_vec) + (B_mat @ M_3_vec) + (C_mat @ M_2_vec)

        # diagonal terms

        np.fill_diagonal(quad_mat, M_4_vec + 4 * (A_sq_mat @ M_4_vec) + (B_mat @ M_3_vec) + (C_mat @ M_2_vec))


        # off-diagonal terms

        








       

    


        






