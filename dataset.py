import numpy as np
from scipy.linalg import orthogonal_procrustes

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from tqdm import tqdm

from mu_dist import Mu
from sigma_dist import Sigma_Star
from gamma_dist import Gamma

from r_dist import R
from z_dist import Z
from phi_dist import Phi

from dataset_initialisation import GMM_Init

from scipy.stats import beta

np.random.seed(42)

class Dataset():
    
    def __init__(self, embds, emb_dim=2, N=1000, K=2):
        self.d = emb_dim
        self.N = N
        self.K = K
        self.embds = embds  # of size (N, self.d) where N is the number
        # print("self.embds", self.embds)
        self.normed_embds = self.embds / np.linalg.norm(self.embds, axis=1)[:, np.newaxis]

        self.means_vars = [Mu(i, self.d) for i in range(self.K)]
        self.sigma_star_vars = [Sigma_Star(i, self.d) for i in range(self.K)]
        self.gamma_vars = [Gamma(i, self.d) for i in range(self.K)]
 
        self.r_vars = [R(self.d-1) for _ in range(self.N)]
        self.z_vars = [Z(self.d, self.K) for _ in range(self.N)]
        self.phi_var = Phi(self.K)



    def dataset_vi(self, max_iter=10, run_init=False):

        self.print_progress(epoch=0)
        

        for epoch in range(1, max_iter+1):

            # for k in range(self.K):

            #     self.means_vars[k].vi(self.z_vars, self.r_vars, self.sigma_star_vars[k], self.gamma_vars[k], self)
            #     self.sigma_star_vars[k].vi(self.z_vars, self.r_vars, self.means_vars[k], self.gamma_vars[k], self)
            #     self.gamma_vars[k].vi(self.z_vars, self.r_vars, self.sigma_star_vars[k], self.means_vars[k], self)

            for i in range(self.N):
                self.r_vars[i].vi(self.z_vars[i], self.sigma_star_vars, self.gamma_vars, self.means_vars, self.phi_var, self.normed_embds[i]) 
                # self.z_vars[i].vi(self.r_vars[i], self.means_vars, self.sigma_star_vars, self.gamma_vars, self.normed_embds[i], self.phi_var, verbose=i<10)
            
            # self.phi_var.vi(self.z_vars)
        
            self.print_progress(epoch)
            
            
    
    def print_progress(self, epoch):

        print(f"""Iteration {epoch} results:
                  
                    μ_0_mean: {self.means_vars[0].mean}
                    μ_0_cov: {self.means_vars[0].cov}

                    μ_1_mean: {self.means_vars[1].mean}
                    μ_1_cov: {self.means_vars[1].cov}


                 _____________________________________________________________________

                    sigma_0_scale: {self.sigma_star_vars[0].scale}
                    sigma_0_dof: {self.sigma_star_vars[0].dof}

                    sigma_1_scale: {self.sigma_star_vars[1].scale}
                    sigma_1_dof: {self.sigma_star_vars[1].dof}


                _____________________________________________________________________

                    gamma_0_mean: {self.gamma_vars[0].mean}
                    gamma_0_cov: {self.gamma_vars[0].cov}

                    gamma_1_mean: {self.gamma_vars[1].mean}
                    gamma_1_cov: {self.gamma_vars[1].cov}

                _____________________________________________________________________

                    First 10 z probs: {[x.probs for x in self.z_vars[:10]]}

                _____________________________________________________________________
                    
                    r_first_alpha: {[x.alpha for x in self.r_vars[:10]]}
                    r_first_beta: {[x.beta for x in self.r_vars[:10]]}
                    r_first moment: {[x.first_moment for x in self.r_vars[:10]]}
                    r_second moment: {[x.second_moment for x in self.r_vars[:10]]}

                    true r_values: {[np.linalg.norm(self.embds[i]) for i in range(10)]}

                    MLE_r_values: {[x.MLE() for x in self.r_vars[:10]]}

                 _____________________________________________________________________

                    phi_probs: {self.phi_var.conc}
                  
                  
                  """)



if __name__ == '__main__':

    μ_0 = np.array([0.75,0.25])
    μ_1 = np.array([0.25, 0.75])

    # μ_1 = μ_1 / np.linalg.norm(μ_1)
    # μ_2 = μ_2 / np.linalg.norm(μ_2)

    μ_0 = np.array([0.75,0.25])
    cov_0 = np.array([[0.1, 0.05], [0.05, 0.1]])

    μ_1 = np.array([0.25, 0.75])
    cov_1 = np.array([[0.1, 0.05], [0.05, 0.1]])

    gamma_prior_cov = np.array([0.1])

    ν = cov_0[1,1]


    # Number of samples to generate
    n_samples = 1000

    # Generate samples alternately
    samples = np.zeros((n_samples, 2))
    for i in range(n_samples):
        if i % 2 == 0:
            samples[i] = np.random.multivariate_normal(μ_0, cov_0)
        else:
            samples[i] = np.random.multivariate_normal(μ_1, cov_1)

    
    ds = Dataset(samples, emb_dim=2, N=1000, K=2)

    for i in range(0,len(ds.z_vars)):
        ds.z_vars[i].probs = [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]

    assumed_dof = 5 #= d+3

    ds.means_vars[0].prior_cov = cov_0
    ds.means_vars[0].mean = μ_0
    ds.means_vars[0].cov = cov_0

    ds.means_vars[1].prior_cov = cov_1
    ds.means_vars[1].mean = μ_1
    ds.means_vars[1].cov = cov_1


    ds.gamma_vars[0].prior_cov = np.array([gamma_prior_cov / np.sqrt(ν)])
    ds.gamma_vars[0].mean = np.array([cov_0[0, 1]]) / np.sqrt(ν)
    ds.gamma_vars[0].cov = np.array([gamma_prior_cov / ν])
    ds.gamma_vars[0].nu = cov_0[-1,-1]

    ds.gamma_vars[1].prior_cov = np.array([gamma_prior_cov / np.sqrt(ν)])
    ds.gamma_vars[1].mean = np.array([cov_1[0, 1]]) / np.sqrt(ν)
    ds.gamma_vars[1].cov = np.array([gamma_prior_cov / ν])
    ds.gamma_vars[1].nu = cov_1[-1,-1]

    ds.sigma_star_vars[0].prior_scale = np.array([cov_0[0,0] * (assumed_dof - ds.d)])
    ds.sigma_star_vars[0].dof = 5
    ds.sigma_star_vars[0].prior_dof = 5
    ds.sigma_star_vars[0].scale = np.array([cov_0[0,0] - ds.gamma_vars[0].mean ** 2])
    ds.sigma_star_vars[0].nu = cov_1[-1,-1]                


    ds.sigma_star_vars[1].prior_scale = np.array([cov_1[0,0] - ds.gamma_vars[0].mean ** 2])
    ds.sigma_star_vars[1].dof = 5
    ds.sigma_star_vars[1].prior_dof = 5
    ds.sigma_star_vars[1].scale = np.array([[cov_1[0,0] * (assumed_dof - ds.d)]])
    ds.sigma_star_vars[1].nu = cov_1[-1,-1]

    full_sigma_inv_estimates = [np.linalg.inv(cov_mat) for cov_mat in [cov_0, cov_1]]

    
    C=0
    D=0
    for i, r_var in enumerate(ds.r_vars):
        norm_datapoint = ds.normed_embds[i]
        norm_datapoint = norm_datapoint.reshape(-1, 1)

        for k in range (0, len(ds.z_vars[i].probs)):
            data_group = k
            sigma = ds.sigma_star_vars[data_group]
            γ = ds.gamma_vars[data_group]
            μ = ds.means_vars[data_group]

            sigma_inv = full_sigma_inv_estimates[data_group]

            C += ds.phi_var.conc[k] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), norm_datapoint)
            D += ds.phi_var.conc[k] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), μ.mean)


        r_var.alpha = C / 2
        r_var.beta = D / C
    


        r_var.update_moments(norm_datapoint)
        
        # initialise phi variables

    ds.phi_var.conc[0] = n_samples // 2
    ds.phi_var.conc[1] = n_samples // 2 

   

    ds.dataset_vi(max_iter=5)


    # α = 7
    # β = 2
    # prior = lambda : beta.rvs(α, β)

    

    # prior = lambda : 0.5

    # ds = Synthetic_data(μ_1, μ_2, prior, N_t=1000)

    # # ds.dataset_vi(max_iter=3)   

    # ds.gaussian_mm_init()

    # assumed_dof = ds.sigma_star_vars[0].dof
    # print(f"==>> assumed_dof: {assumed_dof}")

    # print(f"==>> ds.gamma_vars[0].prior_cov: {ds.gamma_vars[0].prior_cov}")

    # print(f"==>> ds.sigma_star_vars[0].prior_scale: {ds.sigma_star_vars[0].prior_scale}")
    # print(f"==>> ds.sigma_star_vars[0].scale: {ds.sigma_star_vars[0].scale}")

    # for i in range(0,len(ds.z_vars)):
    #     ds.z_vars[i].probs = [0.0, 1.0] if i % 2 == 0 else [1.0, 0.0]


    # ds.means_vars[0].prior_cov = cov_1
    # ds.means_vars[0].mean = μ_1
    # ds.means_vars[0].cov = cov_1

    # ds.gamma_vars[0].prior_cov = np.array([gamma_prior_cov / np.sqrt(ν)])
    # ds.gamma_vars[0].mean = np.array([cov_1[0, 1]]) / np.sqrt(ν)
    # ds.gamma_vars[0].cov = np.array([gamma_prior_cov / ν])
    # ds.gamma_vars[0].nu = cov_1[-1,-1]

    # ds.sigma_star_vars[0].prior_scale = np.array([cov_1[0,0] * (assumed_dof - ds.d)])
    # ds.sigma_star_vars[0].scale = np.array([cov_1[0,0] - ds.gamma_vars[0].mean ** 2])
    # ds.sigma_star_vars[0].nu = cov_1[-1,-1]                

    
    # ds.means_vars[1].prior_cov = cov_2[1,1]
    # ds.means_vars[1].mean = μ_2
    # ds.means_vars[1].cov = cov_2

    # ds.gamma_vars[1].prior_cov = np.array([gamma_prior_cov / np.sqrt(ν)])
    # ds.gamma_vars[1].mean = np.array([cov_2[0, 1]]) / np.sqrt(ν)
    # ds.gamma_vars[1].cov = np.array([gamma_prior_cov / ν])
    # ds.gamma_vars[1].nu = cov_2[-1,-1]

    # ds.sigma_star_vars[1].prior_scale = np.array([cov_2[0,0] - ds.gamma_vars[0].mean ** 2])
    # ds.sigma_star_vars[1].scale = np.array([[cov_2[0,0] * (assumed_dof - ds.d)]])
    # ds.sigma_star_vars[1].nu = cov_2[-1,-1]

    # full_sigma_inv_estimates = [np.linalg.inv(cov_mat) for cov_mat in [cov_1, cov_2]]

    # for i, (r_var, label) in enumerate(zip(ds.r_vars, ds.gmm.labels)):
    #     curr_data = ds.normed_embds[i]

    #     # r_var.alpha = np.random.uniform(3.0, 3.5)
    #     # r_var.beta = np .random.uniform(0.2, 0.3)
    #     # r_var.update_moments()

    #     r_var.alpha = 0.5 * np.matmul(curr_data.T, np.matmul(full_sigma_inv_estimates[label], curr_data))

    #     beta_numerator = np.matmul(curr_data.T, np.matmul(full_sigma_inv_estimates[label], ds.means_vars[label].mean)) 
    #     beta_denom = np.matmul(curr_data.T, np.matmul(full_sigma_inv_estimates[label], curr_data))

    #     r_var.beta = beta_numerator / beta_denom

    #     r_var.update_moments()
        
    #     # initialise phi variables

    # for k in range(ds.K):
    #     # count number of labels in group k
    #     num_labels = sum([1 for lab in ds.gmm.labels if lab == k])
    #     ds.phi_var.conc[k] = num_labels + ds.phi_var.prior_conc[k]


# ds.dataset_vi(max_iter=3, run_init=False)






    
