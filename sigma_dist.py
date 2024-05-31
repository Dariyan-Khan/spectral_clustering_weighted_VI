import numpy as np

class Sigma_Star():

    def __init__(self, k, d, prior_scale=None, prior_dof=None):

        if prior_scale is None:
            self.prior_scale = np.eye(d-1)
        
        else:
            self.prior_scale = prior_scale # (scale matrix, degrees of freedom)
        
        if prior_dof is None:
            self.prior_dof = d + 3
        
        else:
            self.prior_dof = prior_dof

        self.k = k
        self.d = d
        self.dim = d - 1
        self.scale = None
        self.dof = None
        self.first_moment = None
        self.nu = None
    
    def X_i_matrix(self, r_i, μ_k, γ_k, norm_data): #norm data means that the data point is normalised
        X_i = np.zeros((self.dim, self.dim))

        for l in range(0, self.dim):
            for m in range(0, self.dim):
                first_term = r_i.second_moment * norm_data[l] * norm_data[m] - \
                            μ_k.mean[l]*r_i.first_moment*norm_data[m] - \
                            μ_k.mean[m]*r_i.first_moment*norm_data[l] + \
                            μ_k.mean[l]*μ_k.mean[m] + \
                            μ_k.cov[l, m]
                
                second_term = np.sqrt(self.nu) * γ_k.mean[l] * (
                    r_i.second_moment * norm_data[self.d-1] * norm_data[m] - \
                    r_i.first_moment * norm_data[self.d-1] * μ_k.mean[m] - \
                    r_i.first_moment * norm_data[m] * μ_k.mean[self.d-1] + \
                    μ_k.mean[m] * μ_k.mean[self.d-1] + \
                    μ_k.cov[m, self.d-1]
                )
    
                third_term = np.sqrt(self.nu) * γ_k.mean[m] * (
                    r_i.second_moment * norm_data[self.d-1] * norm_data[l] - \
                    r_i.first_moment * norm_data[self.d-1] * μ_k.mean[l] - \
                    r_i.first_moment * norm_data[l] * μ_k.mean[self.d-1] + \
                    μ_k.mean[l] * μ_k.mean[self.d-1] + \
                    μ_k.cov[l, self.d-1]
                )
    
                fourth_term = self.nu * (γ_k.mean[l] * γ_k.mean[m] + γ_k.cov[l, m]) * (
                    r_i.second_moment * norm_data[self.d-1]**2 - \
                    2 * r_i.first_moment * norm_data[self.d-1] * μ_k.mean[self.d-1] + \
                    μ_k.mean[self.d-1]**2 + \
                    μ_k.cov[self.d-1, self.d-1]
                )

                X_i[l, m] = first_term - second_term - third_term + fourth_term
        
        return X_i
            


    def vi(self, z_vi_list, r_vi_list, μ_k, γ_k, phi_var, datapoints):

        scale_mat = self.prior_scale
        dof = self.prior_dof

        for (i, data) in enumerate(datapoints.normed_embds):
            z = z_vi_list[i]

            # print(f"==>> z.probs[self.k].shape: {z.probs[self.k].shape}")
            # print(f"==>> z.probs: {z.probs}")
            # print(f"==>> z.probs[self.k]: {z.probs[self.k]}")
            # print(f"==>> self.X_i_matrix(r_vi_list[i], μ_k, γ_k, data).shape: {self.X_i_matrix(r_vi_list[i], μ_k, γ_k, data).shape}")
            # print(f"==>> scale_mat.shape: {scale_mat.shape}")

            scale_mat += phi_var.conc[self.k] * self.X_i_matrix(r_vi_list[i], μ_k, γ_k, data)
            dof += phi_var.conc[self.k]
        
        self.scale = scale_mat
        self.dof = max(dof, self.d+3)  # added max here

        self.first_moment = self.first_mom()
        self.second_moment = self.second_mom()

    
    def first_mom(self):
        return self.scale / (self.dof - self.dim - 1)
    
    def second_mom(self):
        assert self.dof > self.dim + 3, "Degrees of freedom must be greater than d + 2 for second moment formula"

        #c_2 = 1 / ((self.dof - self.d +1) * (self.dof - self.d) * (self.dof - self.d - 2))

        c_2 = (self.dof - self.dim) * (self.dof - self.dim - 1) * (self.dof - self.dim - 3)
        c_2 = 1 / c_2

        c_1 = (self.dof - self.dim - 2) * c_2

        return (c_1+c_2) * (self.scale @ self.scale) + c_2 * np.trace(self.scale) * self.scale








