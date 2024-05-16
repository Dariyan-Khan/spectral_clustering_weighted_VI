import numpy as np


class Gamma():

    def __init__(self, k, d, prior_mean=None, prior_cov=None):
        self.k = k
        self.d = d
        self.dim = d - 1
        self.mean = None
        self.cov = None

        if prior_mean is not None:
            self.mean = prior_mean
        
        else:
            self.mean = np.zeros(self.dim)
        
        if prior_cov is not None:
            self.prior_cov = prior_cov
        
        else:
            self.prior_cov = np.eye(self.dim)
    
    def vi(self, z_vi_list, r_vi_list, sigma_star_k, μ_k, datapoints):

        mean_vec = np.zeros(self.dim)
        cov_mat = np.zeros((self.dim, self.dim))
        cov_mat_inner = np.zeros((self.dim, self.dim))
        
        for (i, data) in enumerate(datapoints.normed_embds):
            z = z_vi_list[i]

            cov_mat_inner += z.probs[self.k] * (
                r_vi_list[i].second_moment * data[self.d-1]**2 - \
                2 * r_vi_list[i].first_moment * data[self.d-1]*μ_k.mean[self.d-1] + \
                μ_k.mean[self.d-1]**2 + \
                μ_k.cov[self.d-1, self.d-1]
            )

            mean_vec += z.probs[self.k] * (
                r_vi_list[i].second_moment * data[self.d-1] * data[:self.d-1] - \
                r_vi_list[i].first_moment * data[self.d-1] * μ_k.mean[:self.d-1] - \
                r_vi_list[i].first_moment * data[:self.d-1] * μ_k.mean[self.d-1] + \
                μ_k.mean[:self.d-1] * μ_k.mean[self.d-1]
            )



        cov_mat = sigma_star_k.dof * np.matmul(np.linalg.inv(sigma_star_k.scale), cov_mat_inner)
        cov_mat += np.linalg.inv(self.prior_cov)

        # print(np.linalg.det(cov_mat), "cov mat det")

        self.cov = np.linalg.inv(cov_mat)

        mean_vec.reshape(-1,1)

        mean_vec = np.matmul(sigma_star_k.dof * np.linalg.inv(sigma_star_k.scale), mean_vec)

        self.mean = np.matmul(self.cov, mean_vec)

        self.outer_product = self.outer_prod()

        self.corr = self.cov / np.sqrt(np.outer(np.diag(self.cov), np.diag(self.cov)))

        # self.triple_gamma = self.three_gama()

        # self.quad_gamma = self.quadruple_gamma()
    

    def outer_prod(self):
        return self.cov + np.outer(self.mean, self.mean)


    def three_gamma(self):
        col_vars = np.tile(np.diag(self.cov).reshape(1, self.dim), (self.dim, 1)) # if we change rows, should stay the same 
        A_mat  = self.cov / col_vars

        np.fill_diagonal(A_mat, 0)

        M_2_vec = np.diag(self.cov) + self.mean**2
        M_3_vec = self.mean**3 + 3 * self.mean * np.diag(self.cov)

        col_mu = np.tile(self.mean.reshape(1,self.dim), (self.dim,1))  # if we change rows, should stay the same 
        row_mu = col_mu.T  # if we change cols, should stay the same 

        B_mat = row_mu - col_mu*A_mat

        np.fill_diagonal(B_mat, 0)

        return M_3_vec  + (A_mat @ M_3_vec) + (B_mat @ M_2_vec)


    def quadruple_gamma(self):

        self.corr = self.cov / np.sqrt(np.outer(np.diag(self.cov), np.diag(self.cov)))

        quad_mat =  np.zeros((self.dim, self.dim))

        col_vars = np.tile(np.diag(self.cov).reshape(1, self.dim), (self.dim, 1)) # if we change rows, should stay the same 
        row_vars = col_vars.T # if we change cols, should stay the same 

        A_mat  = self.cov / col_vars
        np.fill_diagonal(A_mat, 0)

        A_sq_mat = A_mat**2

        col_mu = np.tile(self.mean.reshape(1,self.dim), (self.dim,1))
        row_mu = col_mu.T

        B_mat = 2 * A_mat * (row_mu - col_mu*A_mat)
        np.fill_diagonal(B_mat, 0)

        C_mat = (row_mu - A_mat * col_mu)**2 + row_vars * (1-self.corr**2)
        np.fill_diagonal(C_mat, 0)

        M_2_vec = np.diag(self.cov) + self.mean**2
        M_3_vec = self.mean**3 + 3 * self.mean * np.diag(self.cov)
        M_4_vec = self.mean**4 + 6 * self.mean**2 * np.diag(self.cov) + 3 * np.diag(self.cov)**2

        # diagonal terms

        np.fill_diagonal(quad_mat, M_4_vec + 4 * (A_sq_mat @ M_4_vec) + (B_mat @ M_3_vec) + (C_mat @ M_2_vec))


        # off-diagonal terms

        B_mat_off_diag = col_mu - (row_mu * (A_mat / row_vars))

        P_tensor = np.zeros((self.dim, self.dim, self.dim))
        Q_tensor = np.zeros((self.dim, self.dim, self.dim))

        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    if i == j or k == i or k == j:
                        continue

                    P_tensor[i, j, k] = self.mean[i] * np.sqrt(self.cov[i,i]) * (self.corr[i, k] - self.corr[j,k] * self.corr[i,j]) / \
                                        (np.sqrt(self.cov[k,k]) * (1-self.corr[j,k]**2))

                    Q_tensor[i, j, k] = (self.mean[j]**2 + self.cov[j,j]) * \
                                        (
                                            np.sqrt(self.cov[i,i]) * (self.corr[i, j] - self.corr[j,k] * self.corr[i,k]) / \
                                         (np.sqrt(self.cov[j,j]) * (1-self.corr[j,k]**2))
                                        )
                    
                    Q_tensor[i, j, k] += self.mean[j] * (
                        self.mean[i] + (
                            (
                                np.sqrt(self.cov[i,i]) * np.sqrt(self.cov[j,j]) * \
                                self.mean[k] * (self.corr[i,j] * self.corr[j,k] - self.corr[i,k]) + \
                                np.sqrt(self.cov[i,i]) * np.sqrt(self.cov[k,k]) * \
                                self.mean[j] * (self.corr[i,k] * self.corr[j,k] - self.corr[i,j])
                            ) / \
                                ((1 - self.corr[j,k]**2) * np.sqrt(self.cov[j,j]) * np.sqrt(self.cov[k,k]))
                    )
                    )

        for i in range(self.dim):
            for j in range(self.dim):
                if i == j:
                    continue

                quad_mat[i,j] = A_mat[i,j]*M_4_vec[i] + B_mat_off_diag[i,j] * M_3_vec[i] + \
                                A_mat[j,i]*M_4_vec[j] + B_mat_off_diag[j,i] * M_3_vec[j]
                
                for k in range(self.dim):
                    if k == i or k == j:
                        continue

                    quad_mat[i,j] += P_tensor[i,j,k] * M_3_vec[k] + Q_tensor[i,j,k] * M_2_vec[k]
        

        return quad_mat


                
                    
                    



                   

                










       

    


        






