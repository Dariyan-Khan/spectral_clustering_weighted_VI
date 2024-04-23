import numpy as np

class Sigma_star():

    def __init__(self, prior_scale, prior_dof, k, d):
        self.prior_scale = prior_scale # (scale matrix, degrees of freedom)
        self.prior_dof = prior_dof
        self.k = k
        self.d = d
        self.dim = d - 1
        self.scale = None
        self.dof = None
    
    def X_i_matrix(self, r_i, μ_k, γ_k, norm_data): #norm data means that the data point is normalised
        X_i = np.zeros((self.dim, self.dim))

        for l in range(0, self.dim):
            for m in range(0, self.dim):
                first_term = r_i.second_moment * norm_data[l] * norm_data[m] - \
                            μ_k.mean[l]*r_i.first_moment*norm_data[m] - \
                            μ_k.mean[m]*r_i.first_moment*norm_data[l] + \
                            μ_k.mean[l]*μ_k.mean[m] + \
                            μ_k.cov[l, m]
                
                second_term = γ_k.mean[l] * (
                    r_i.second_moment * norm_data[self.d] * norm_data[m] - \
                    r_i.first_moment * norm_data[self.d] * μ_k.mean[m] - \ 
                    r_i.first_moment * norm_data[m] * μ_k.mean[self.d] + \
                    μ_k.mean[m] * μ_k.mean[self.d] + \
                    μ_k.cov[m, self.d]
                )
    
                third_term = γ_k.mean[m] * (
                    r_i.second_moment * norm_data[self.d] * norm_data[l] - \
                    r_i.first_moment * norm_data[self.d] * μ_k.mean[l] - \ 
                    r_i.first_moment * norm_data[l] * μ_k.mean[self.d] + \
                    μ_k.mean[l] * μ_k.mean[self.d] + \
                    μ_k.cov[l, self.d]
                )
    
                fourth_term = (γ_k.mean[l] * γ_k.mean[m] + γ_k.cov[l, m]) * (
                    r_i.second_moment * norm_data[self.d]**2 - \
                    2 * r_i.first_moment * norm_data[self.d] * μ_k.mean[self.d] + \
                    μ_k.mean[self.d]**2 + \
                    μ_k.cov[self.d, self.d]
                )

                X_i[l, m] = first_term - second_term - third_term + fourth_term
        
        return X_i
            




    def vi(self, phi_vi_list, r_vi_list, μ_k, γ_k, datapoints):

        scale_mat = self.prior_scale
        dof = self.prior_dof

        for (i, data) in enumerate(datapoints.normalised):
            phi = phi_vi_list[i]
            scale_mat += phi[self.k] * self.X_i_matrix(r_vi_list[i], μ_k, γ_k, data)
            dof += phi[self.k]
        
        self.scale = scale_mat
        self.dof = dof








