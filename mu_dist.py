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

            print(f"==>> data: {data}")

            z = z_vi_list[i]

            # print("z_probs", z.probs)

            n_k += z.probs[self.k] #phi_var.conc[self.k]
        
            B += r_vi_list[i].first_moment * z.probs[self.k] * data.T

            

        # print("curr sigma_star_k", sigma_star_k)

        #sigma_inv_estimate = sigma_inv_approx(sigma_star_k, γ_k, α=sigma_star_k.nu)

        #print()

        print(n_k, "n_k")

        cov_0 = np.array([[0.1, 0.05], [0.05, 0.1]])

        sigma_inv_estimate = np.linalg.inv(cov_0)

        B = np.matmul(B, sigma_inv_estimate)
                
        A = sigma_inv_estimate*n_k + np.linalg.inv(self.prior_cov)

        # print(f"==>> sigma_inv_estimate*n_k: {sigma_inv_estimate*n_k}")

        # print(f"==>> np.linalg.inv(self.prior_cov): {np.linalg.inv(self.prior_cov)}")


        A_inv = np.linalg.inv(A)

        # print(B.shape, "B shape")
        # print(A_inv.shape, "A_inv shape")
        # print(sigma_inv_estimate.shape, "sigma_inv_estimate shape")

        B = np.matmul(A_inv, np.matmul(sigma_inv_estimate, B.T))

        # print()

        # print("B_matrix", B)

        # print("A_inverse", A_inv)
    
        #B = np.reshape(B, (-1, 1))

        self.mean = np.matmul(A_inv, B)
        self.cov = A_inv

        # self.mean = self.mean / np.linalg.norm(self.mean)




    