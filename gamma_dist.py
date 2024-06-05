import numpy as np
from copy import copy, deepcopy


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
        
        self.nu = None
    
    def vi(self, z_vi_list, r_vi_list, sigma_star_k, μ_k, phi_var, datapoints, real_cov=None):

        mean_vec = np.zeros(self.dim)
        cov_mat = np.zeros((self.dim, self.dim))
        cov_mat_inner = np.zeros((self.dim, self.dim))
        
        mean_vec = np.expand_dims(mean_vec, axis=1)

        μ_k_mean_last_term = np.array([μ_k.cov[self.d-1, i] + μ_k.mean[self.d-1]*μ_k.mean[i] for i in range(self.dim)])
        # print(f"==>> μ_k_mean_last_term: {μ_k_mean_last_term}")

        for (i, data) in enumerate(datapoints.normed_embds):
            z = z_vi_list[i]

            cov_mat_inner += z.probs[self.k] * (
                r_vi_list[i].second_moment * data[self.d-1]**2 - \
                2 * r_vi_list[i].first_moment * data[self.d-1]*μ_k.mean[self.d-1] + \
                μ_k.mean[self.d-1]**2 + \
                μ_k.cov[self.d-1, self.d-1]
            )

            data = data.reshape(-1, 1)


            mean_vec += z.probs[self.k] * (
                r_vi_list[i].second_moment * data[self.d-1] * data[:self.d-1] - \
                r_vi_list[i].first_moment * data[self.d-1] * μ_k.mean[:self.d-1] - \
                r_vi_list[i].first_moment * data[:self.d-1] * μ_k.mean[self.d-1] + \
                μ_k_mean_last_term
                # μ_k.mean[:self.d-1] * μ_k.mean[self.d-1]
            )

            # print(f"==>> mean_vec: {mean_vec}")

        # mean_vec = np.expand_dims(mean_vec, axis=1)

        if sigma_star_k.scale.size == 1:
            scale_mat = copy(np.array([sigma_star_k.scale]))
        else:
            scale_mat = copy(sigma_star_k.scale)

        cov_mat = (1/self.nu) * sigma_star_k.dof * np.matmul(np.linalg.inv(scale_mat), cov_mat_inner)

       # print(f"==>> cov_mat: {cov_mat}")

        cov_mat += np.linalg.inv(self.prior_cov)
        #print(f"==>> cov_mat: {cov_mat}")


        # print(np.linalg.det(cov_mat), "cov mat det")

        self.cov = np.linalg.inv(cov_mat)

        mean_vec = (1/np.sqrt(self.nu)) * sigma_star_k.dof * np.matmul(np.linalg.inv(scale_mat), mean_vec)

        # print(f"==>> mean_vec: {mean_vec}")

        self.mean = np.matmul(self.cov, mean_vec)
        # self.mean = self.mean.reshape(-1,1)

        self.outer_product = self.outer_prod()

        if self.cov.size == 1:
            std_cov = np.reshape(self.cov, (1, 1))
        
        else:
            std_cov = deepcopy(self.cov)

        std_devs = np.sqrt(np.diag(std_cov))
        
        self.corr = self.cov / np.outer(std_devs, std_devs)
    


    def outer_prod(self):
        return self.cov + np.outer(self.mean, self.mean)


    def three_gamma(self):

        if self.cov.size == 1:
            s_cov = np.reshape(self.cov, (1, 1))
        
        else:
            s_cov = deepcopy(self.cov)

        std_devs = np.sqrt(np.diag(s_cov))
        # print(std_devs, "std_devs")
        self.corr = self.cov / np.outer(std_devs, std_devs)

        three_vec = np.zeros(self.dim)

        for i in range(self.dim):
            three_vec[i] = self.mean[i]**3 + 3 * self.mean[i] * self.cov[i,i]
            for j in range(self.dim):
                if j == i:
                    continue

                A_ij = self.corr[i,j] * (np.sqrt(self.cov[i,i]) / np.sqrt(self.cov[j,j]))

                B_ij = self.mean[i] - self.corr[i,j] * self.mean[j]* (np.sqrt(self.cov[i,i]) / np.sqrt(self.cov[j,j]))

                three_vec[i] += A_ij * (self.mean[j]**3 + 3 * self.mean[j] * self.cov[j,j])
                three_vec[i] += B_ij * (self.mean[j]**2 + self.cov[j,j])
    
        three_vec = three_vec.reshape(-1, 1)

        return three_vec


    def quadruple_gamma(self):

        if self.cov.size == 1:
            s_cov = np.reshape(self.cov, (1, 1))
        
        else:
            s_cov = deepcopy(self.cov)

        std_devs = np.sqrt(np.diag(s_cov))
        # print(std_devs, "std_devs")

        self.corr = self.cov / np.outer(std_devs, std_devs)


        quad_mat =  np.zeros((self.dim, self.dim))

       
        # diagonal terms

        for i in range(self.dim):
            quad_mat[i,i] = self.mean[i]**4 + 6 * self.mean[i]**2 * self.cov[i,i] + 3 * self.cov[i,i]**2
                
            for j in range(self.dim):
                if j == i:
                    continue

                A_ij = self.corr[i,j]**2 * self.cov[i,i] / self.cov[j,j]

                B_ij = 2.0 * self.corr[i,j] * (np.sqrt(self.cov[i,i]) / np.sqrt(self.cov[j,j])) * (self.mean[i] - self.corr[i,j] * self.mean[j]* (np.sqrt(self.cov[i,i]) / np.sqrt(self.cov[j,j]) ))

                C_ij = (self.mean[i] - np.sqrt(A_ij) * self.mean[j])**2 + self.cov[i,i] * (1 - self.corr[i,j]**2)

                quad_mat[i,i] += A_ij * (self.mean[j]**4 + 6 * self.mean[j]**2 * self.cov[j,j] + 3 * self.cov[j,j]**2)
                quad_mat[i,i] += B_ij * (self.mean[j]**3 + 3*self.mean[j]*self.cov[j,j])
                quad_mat[i,i] += C_ij * (self.mean[j]**2 + self.cov[j,j])


        # off-diagonal terms

        # B_mat_off_diag = col_mu - (row_mu * (A_mat / row_vars))

        P_tensor = np.zeros((self.dim, self.dim, self.dim))
        Q_tensor = np.zeros((self.dim, self.dim, self.dim))
        R_tensor = np.zeros((self.dim, self.dim, self.dim))

        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    if i == j or k == i or k == j:
                        continue

                    # print(f"==>> i: {i}, j: {j}, k: {k}")
                    
                    A_dash = np.sqrt(self.cov[i,i]) * (self.corr[i, j] - self.corr[j,k] * self.corr[i,k]) / (np.sqrt(self.cov[j,j]) * (1-self.corr[j,k]**2))
                    B_dash = np.sqrt(self.cov[i,i]) * (self.corr[i, k] - (self.corr[j,k] * self.corr[i,j])) / (np.sqrt(self.cov[k,k]) * (1-self.corr[j,k]**2))
                    C_dash = self.mean[i] + (
                            (
                                np.sqrt(self.cov[i,i]) * np.sqrt(self.cov[j,j]) * self.mean[k] * (self.corr[i,j] * self.corr[j,k] - self.corr[i,k]) + \
                                np.sqrt(self.cov[i,i]) * np.sqrt(self.cov[k,k]) * self.mean[j] * (self.corr[i,k] * self.corr[j,k] - self.corr[i,j])

                            ) / ((1 - self.corr[j,k]**2) * np.sqrt(self.cov[j,j]) * np.sqrt(self.cov[k,k]))
                        )
                    
                    P_tensor[i, j, k] = A_dash * ((self.cov[j,j] * self.corr[j,k]**2) / self.cov[k,k]) + \
                                        B_dash * ((np.sqrt(self.cov[j,j]) * self.corr[j,k]) / np.sqrt(self.cov[k,k]))
                    
                    
                    Q_tensor[i, j, k] = ((2 * A_dash * np.sqrt(self.cov[j,j]) * self.corr[j,k]) / self.cov[k,k]) * (self.mean[j]*np.sqrt(self.cov[k,k]) - self.mean[k] * self.corr[j,k] * np.sqrt(self.cov[j,j])) + \
                                        B_dash * (self.mean[j] - (np.sqrt(self.cov[j,j]) * self.corr[j,k] * self.mean[k] / np.sqrt(self.cov[k,k])) ) + \
                                        C_dash * (np.sqrt(self.cov[j,j]) * self.corr[j,k] / np.sqrt(self.cov[k,k]))


                    R_tensor[i,j, k] = A_dash * (self.mean[j]**2 + (self.mean[k]**2 * self.cov[j,j] * self.corr[j,k]**2 / self.cov[k,k]) - (2 * self.mean[j] * np.sqrt(self.cov[j,j]) * self.corr[j,k]*self.mean[k] / np.sqrt(self.cov[k,k])) + (1-self.corr[j,k]**2)*self.cov[j,j]) + \
                                        C_dash * (self.mean[j] - ((np.sqrt(self.cov[j,j]) * self.corr[j,k] * self.mean[k]) / np.sqrt(self.cov[k,k])))
                    


                    

        for i in range(self.dim):
            for j in range(self.dim):
                if i == j:
                    continue


                A_ij = self.corr[i,j] * np.sqrt(self.cov[j,j] / self.cov[i,i])
                A_ji = self.corr[j,i] * np.sqrt(self.cov[i,i] / self.cov[j,j])

                B_ij = self.mean[j] - self.corr[i,j] * self.mean[i] * np.sqrt(self.cov[j,j] / self.cov[i,i])
                B_ji = self.mean[i] - self.corr[j,i] * self.mean[j] * np.sqrt(self.cov[i,i] / self.cov[j,j])

                quad_mat[i,j] = A_ij * (self.mean[i]**4 + 6 * self.mean[i]**2 * self.cov[i,i] + 3 * self.cov[i,i]**2) + \
                                B_ij * (self.mean[i]**3 + 3 * self.mean[i] * self.cov[i,i])
                
                #print(f"quad_mat[i,j]: {quad_mat[i,j]}")
                quad_mat_term_2 = A_ji * (self.mean[j]**4 + 6 * self.mean[j]**2 * self.cov[j,j] + 3 * self.cov[j,j]**2) + \
                                B_ji * (self.mean[j]**3 + 3 * self.mean[j] * self.cov[j,j])
                
                #print(f"quad_mat_term_2: {quad_mat_term_2}")

                quad_mat[i,j] += A_ji * (self.mean[j]**4 + 6 * self.mean[j]**2 * self.cov[j,j] + 3 * self.cov[j,j]**2) + \
                                B_ji * (self.mean[j]**3 + 3 * self.mean[j] * self.cov[j,j])
                
                

                off_diag_sum_term = 0
                for k in range(self.dim):
                    if k == i or k == j:
                        continue
                
                    quad_mat[i,j] += P_tensor[i,j,k] * (self.mean[k]**4 + 6*self.mean[k]**2*self.cov[k,k] + 3*self.cov[k,k]**2) + \
                                    Q_tensor[i,j,k] * (self.mean[k]**3 + 3 * self.mean[k]*self.cov[k,k]) + \
                                    R_tensor[i,j,k] * (self.mean[k]**2 + self.cov[k,k])
                    
                    off_diag_sum_term += P_tensor[i,j,k] * (self.mean[k]**4 + 6*self.mean[k]**2*self.cov[k,k] + 3*self.cov[k,k]**2) + \
                                    Q_tensor[i,j,k] * (self.mean[k]**3 + 3 * self.mean[k]*self.cov[k,k]) + \
                                    R_tensor[i,j,k] * (self.mean[k]**2 + self.cov[k,k])
                    
                
                # print(f"==>> off_diag_sum_term: {off_diag_sum_term}")
                assert False
        

        return quad_mat


                
            



                   

                










       

    


        