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

    

    def vi(self, z_vi_list, r_vi_list, sigma_star_k, γ_k, phi_var, datapoints):

        n_k = 0
        B = 0

        for (i, data) in enumerate(datapoints.normed_embds):


            z = z_vi_list[i]
            n_k += z.probs[self.k] #phi_var.conc[self.k
            B += r_vi_list[i].first_moment * z.probs[self.k] * data


        cov_0 = np.array([[0.1, 0.05], [0.05, 0.1]])

        sigma_inv_estimate = np.linalg.inv(cov_0)

        # print(f"==>> sigma_inv_estimate.shape: {sigma_inv_estimate.shape}")

        # print(f"==>> B.shape: {B.shape}")

        B = np.matmul(sigma_inv_estimate, B.T)
                
        A = sigma_inv_estimate*n_k + np.linalg.inv(self.prior_cov)


        A_inv = np.linalg.inv(A)

        #B = np.matmul(A_inv, np.matmul(B.T, sigma_inv_estimate))

        print(f"==>> B: {B}")
        print(f"==>> B.shape: {B.shape}")

        self.mean = np.matmul(A_inv, B)
        self.cov = A_inv

        # self.mean = self.mean / np.linalg.norm(self.mean)




    