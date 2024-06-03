import numpy as np
from copy import copy

class Sigma_Star():

    def __init__(self, k, d, prior_scale=None, prior_dof=None):

        if prior_scale is None:
            self.prior_scale = np.eye(d-1)
        
        else:
            self.prior_scale = copy(prior_scale) # (scale matrix, degrees of freedom)
        
        if prior_dof is None:
            self.prior_dof = d + 3
        
        else:
            self.prior_dof = copy(prior_dof)

        self.k = k
        self.d = d
        self.dim = d - 1
        self.scale = None
        self.dof = None
        self.first_moment = None
        self.nu = None
    
    def X_i_matrix(self, r_i, μ_k, γ_k, norm_data): #norm data means that the data point is normalised
        X_i = np.zeros((self.dim, self.dim))

        # print(f"==>> self.nu: {self.nu}")

        for l in range(0, self.dim):
            for m in range(0, self.dim):
                first_term = r_i.second_moment * norm_data[l] * norm_data[m] - \
                            μ_k.mean[l]*r_i.first_moment*norm_data[m] - \
                            μ_k.mean[m]*r_i.first_moment*norm_data[l] + \
                            μ_k.mean[l]*μ_k.mean[m] + \
                            μ_k.cov[l, m]
                
                second_term = (1 / np.sqrt(self.nu)) * γ_k.mean[l] * (
                    r_i.second_moment * norm_data[self.d-1] * norm_data[m] - \
                    r_i.first_moment * norm_data[self.d-1] * μ_k.mean[m] - \
                    r_i.first_moment * norm_data[m] * μ_k.mean[self.d-1] + \
                    μ_k.mean[m] * μ_k.mean[self.d-1] + \
                    μ_k.cov[m, self.d-1]
                )
    
                third_term = (1 / np.sqrt(self.nu)) * γ_k.mean[m] * (
                    r_i.second_moment * norm_data[self.d-1] * norm_data[l] - \
                    r_i.first_moment * norm_data[self.d-1] * μ_k.mean[l] - \
                    r_i.first_moment * norm_data[l] * μ_k.mean[self.d-1] + \
                    μ_k.mean[l] * μ_k.mean[self.d-1] + \
                    μ_k.cov[l, self.d-1]
                )
    
                fourth_term = (1 / self.nu) * (γ_k.mean[l] * γ_k.mean[m] + γ_k.cov[l, m]) * ( # I think it depends heavily on the γ_k term because when I remove it
                                                                                            # We seem to be fine 
                    r_i.second_moment * norm_data[self.d-1]**2 - \
                    2 * r_i.first_moment * norm_data[self.d-1] * μ_k.mean[self.d-1] + \
                    μ_k.mean[self.d-1]**2 + \
                    μ_k.cov[self.d-1, self.d-1]
                )
                print(f"==>> first_term: {first_term}")
                print(f"==>> second_term: {second_term}")
                print(f"==>> third_term: {third_term}")
                print(f"==>> fourth_term: {fourth_term}")
                X_i[l, m] = first_term - second_term - third_term + fourth_term
        
        return X_i
            


    def vi(self, z_vi_list, r_vi_list, μ_k, γ_k, phi_var, datapoints):

        scale_mat = copy(self.prior_scale)
        dof = copy(self.prior_dof)        

        for (i, data) in enumerate(datapoints.normed_embds):
            z = z_vi_list[i]

            X_i_mat = self.X_i_matrix(r_vi_list[i], μ_k, γ_k, data)

            if X_i_mat.size == 1:
                X_i_mat = X_i_mat.reshape(-1)

            scale_mat += z.probs[self.k] * X_i_mat
            dof += z.probs[self.k] 
        

        self.scale = scale_mat
        self.dof = max(dof, self.d+3)  # added max here

        self.first_moment = self.first_mom()
        self.second_moment = self.second_mom()

    def mode(self):
        return self.scale / (self.dof + self.dim + 1)

    def first_mom(self):
        return self.scale / (self.dof - self.dim - 1)
    
    def second_mom(self):
        assert self.dof > self.dim + 3, "Degrees of freedom must be greater than d + 2 for second moment formula"

        #c_2 = 1 / ((self.dof - self.d +1) * (self.dof - self.d) * (self.dof - self.d - 2))

        c_2 = (self.dof - self.dim) * (self.dof - self.dim - 1) * (self.dof - self.dim - 3)
        c_2 = 1 / c_2

        c_1 = (self.dof - self.dim - 2) * c_2

        if self.scale.size == 1:
            scale_mat = self.scale.reshape(1, 1)

        return (c_1+c_2) * (scale_mat @ scale_mat) + c_2 * np.trace(scale_mat) * scale_mat


if __name__ == "__main__":
    s = Sigma_Star(0, 3)
    s.scale = np.array([[1, 2], [2, 3]])
    s.dof = 5

    print(s.fi)









