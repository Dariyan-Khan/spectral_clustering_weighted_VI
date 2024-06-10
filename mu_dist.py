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

    

    def vi(self, z_vi_list, r_vi_list, sigma_star_k, γ_k, phi_var, weights, datapoints, real_cov=None):

        n_k = 0
        B = 0

        for (i, data) in enumerate(datapoints.normed_embds):

            assert len(data.shape) <= 2
            # print(f"==>> r_vi_list[i].first_moment: {r_vi_list[i].first_moment}")
            z = z_vi_list[i]
            n_k += weights[i] * z.probs[self.k] #phi_var.conc[self.k
            B += weights[i] * r_vi_list[i].first_moment * z.probs[self.k] * data

        # print(f"==>> B.shape: {B.shape}")


        #cov_0 = real_cov

        #sigma_inv_estimate = np.linalg.inv(cov_0)
        sigma_inv_estimate = jensen_approx(sigma_star_k, γ_k)
        sigma_inv_estimate = np.reshape(sigma_inv_estimate, (self.d, self.d))
        print(f"==>> sigma_inv_estimate: {sigma_inv_estimate}")

        # print(f"==>> sigma_inv_estimate.shape: {sigma_inv_estimate.shape}")

        # print(f"==>> B.shape: {B.shape}")

        print(f"==>> sigma_inv_estimate.shape: {sigma_inv_estimate.shape}")
        print(f"==>> B.shape: {B.shape}")

        B = np.matmul(sigma_inv_estimate, B.T)
                
        A = sigma_inv_estimate*n_k + np.linalg.inv(self.prior_cov)


        A_inv = np.linalg.inv(A)

        #B = np.matmul(A_inv, np.matmul(B.T, sigma_inv_estimate))

        print(f"==>> B: {B}")
        print(f"==>> B.shape: {B.shape}")

        self.mean = np.matmul(A_inv, B)
        self.cov = A_inv

        # self.mean = self.mean / np.linalg.norm(self.mean)




    