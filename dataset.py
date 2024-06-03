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

np.random.seed(44)

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



    def dataset_vi(self, max_iter=10, run_init=False, real_cov=None):

        self.print_progress(epoch=0, num_els=10, real_cov=real_cov)
        

        for epoch in range(1, max_iter+1):

            for k in range(self.K):
                # pass
                # self.means_vars[k].vi(self.z_vars, self.r_vars, self.sigma_star_vars[k], self.gamma_vars[k], self.phi_var, self, real_cov=real_cov)
                self.sigma_star_vars[k].vi(self.z_vars, self.r_vars, self.means_vars[k], self.gamma_vars[k], self.phi_var, self)
                self.gamma_vars[k].vi(self.z_vars, self.r_vars, self.sigma_star_vars[k], self.means_vars[k], self.phi_var, self, real_cov=real_cov)

            for i in range(self.N):
                pass
                # self.r_vars[i].vi(self.z_vars[i], self.sigma_star_vars, self.gamma_vars, self.means_vars, self.phi_var, self.normed_embds[i]) 
                # self.z_vars[i].vi(self.r_vars[i], self.means_vars, self.sigma_star_vars, self.gamma_vars, self.normed_embds[i], self.phi_var, verbose=i<10, real_cov=real_cov)
            
            # self.phi_var.vi(self.z_vars)
        
            self.print_progress(epoch, num_els=10, real_cov=real_cov)
            
            
    
    def print_progress(self, epoch , num_els=10, real_cov=None):
        

        ν = real_cov[-1,-1]
        real_gamma = np.array([cov_1[0, 1]]) / np.sqrt(ν)

        real_sigma_star = real_cov[0,0] - real_gamma ** 2



        num_elements = len(self.z_vars)
        num_correct = 0
        for i in range(num_elements):
            if i % 2 == 0:
                num_correct += self.z_vars[i].probs[0] > self.z_vars[i].probs[1]
            else:
                num_correct += self.z_vars[i].probs[1] > self.z_vars[i].probs[0]

        print(f"""Iteration {epoch} results:
                  
                    μ_0_mean: {self.means_vars[0].mean}
                    μ_0_cov: {self.means_vars[0].cov}

                    μ_1_mean: {self.means_vars[1].mean}
                    μ_1_cov: {self.means_vars[1].cov}


                 _____________________________________________________________________

                    sigma_0_scale: {self.sigma_star_vars[0].scale}
                    sigma_0_prior_scale: {self.sigma_star_vars[0].prior_scale}
                    sigma_0_dof: {self.sigma_star_vars[0].dof}
                    sigma_0_prior_dof: {self.sigma_star_vars[0].prior_dof}
                    sigma_0_first_moment: {self.sigma_star_vars[0].first_moment}
                    sigma_0_mode: {self.sigma_star_vars[0].mode()}
                    real_sigma_star: {real_sigma_star}


                    sigma_1_scale: {self.sigma_star_vars[1].scale}
                    sigma_1_prior_scale: {self.sigma_star_vars[1].prior_scale}
                    sigma_1_dof: {self.sigma_star_vars[1].dof}
                    sigma_1_prior_dof: {self.sigma_star_vars[1].prior_dof}
                    sigma_1_first_moment: {self.sigma_star_vars[1].first_moment}
                    sigma_1_mode: {self.sigma_star_vars[1].mode()}
                    real_sigma_star: {real_sigma_star}




                _____________________________________________________________________

                    gamma_0_mean: {self.gamma_vars[0].mean}
                    gamma_0_cov: {self.gamma_vars[0].cov}
                    real_gamma_0: {real_gamma}

                    gamma_1_mean: {self.gamma_vars[1].mean}
                    gamma_1_cov: {self.gamma_vars[1].cov}
                    real_gamma_1: {real_gamma}

                _____________________________________________________________________

                    First 10 z probs: {[x.probs for x in self.z_vars[:num_els]]}
                    average number in first group: {sum([x.probs[0] for x in self.z_vars])}
                    average number in second group: {sum([x.probs[1] for x in self.z_vars])}
                    fraction correct: {num_correct / num_elements}


                _____________________________________________________________________
                    
                    r_first_alpha: {[x.alpha for x in self.r_vars[:num_els]]}
                    r_first_beta: {[x.beta for x in self.r_vars[:num_els]]}
                    r_first moment: {[x.first_moment for x in self.r_vars[:num_els]]}
                    r_second moment: {[x.second_moment for x in self.r_vars[:num_els]]}

                    true r_values: {[np.linalg.norm(self.embds[i]) for i in range(num_els)]}

                    MLE_r_values: {[x.MLE() for x in self.r_vars[:num_els]]}

                 _____________________________________________________________________

                    phi_probs: {self.phi_var.conc}
                  
                  
                  """)



if __name__ == '__main__':

    # μ_1 = μ_1 / np.linalg.norm(μ_1)
    # μ_2 = μ_2 / np.linalg.norm(μ_2)

    μ_0 = np.array([0.75,0.25])
    cov_0 = np.array([[0.01, 0.005], [0.005, 0.01]])

    μ_1 = np.array([0.25, 0.75])
    cov_1 = np.array([[0.01, 0.005], [0.005, 0.01]])

    gamma_prior_cov = np.array([0.001]) #0.01*np.array([0.1])

    ν = cov_0[1,1]


    # Number of samples to generate
    n_samples = 10

    # Generate samples alternately
    samples = np.zeros((n_samples, 2))
    for i in range(n_samples):
        if i % 2 == 0:
            samples[i] = np.random.multivariate_normal(μ_0, cov_0)
        else:
            samples[i] = np.random.multivariate_normal(μ_1, cov_1)

    #print(f"==>> samples: {samples}")

    
    ds = Dataset(samples, emb_dim=2, N=n_samples, K=2)

    for i in range(0,len(ds.z_vars)):
        ds.z_vars[i].probs = [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]

    assumed_dof = 5 #= d+3

    ds.means_vars[0].prior_cov = cov_0
    ds.means_vars[0].mean = μ_0
    ds.means_vars[0].cov = cov_0

    ds.means_vars[1].prior_cov = cov_1
    ds.means_vars[1].mean = μ_1
    ds.means_vars[1].cov = cov_1


    ds.gamma_vars[0].prior_cov = np.array([gamma_prior_cov / ν])
    ds.gamma_vars[0].mean = np.array([cov_0[0, 1]]) / np.sqrt(ν)
    ds.gamma_vars[0].cov = np.array([gamma_prior_cov / ν])
    ds.gamma_vars[0].nu = cov_0[-1,-1]

    ds.gamma_vars[1].prior_cov = np.array([gamma_prior_cov / ν])
    ds.gamma_vars[1].mean = np.array([cov_1[0, 1]]) / np.sqrt(ν)
    ds.gamma_vars[1].cov = np.array([gamma_prior_cov / ν])
    ds.gamma_vars[1].nu = cov_1[-1,-1]

    ds.sigma_star_vars[0].prior_scale = np.array([cov_0[0,0] - ds.gamma_vars[0].mean ** 2]) * (assumed_dof - ds.d)  # np.array([cov_0[0,0] * (assumed_dof - ds.d)])
    ds.sigma_star_vars[0].dof = 5
    ds.sigma_star_vars[0].prior_dof = 5
    ds.sigma_star_vars[0].scale = np.array([cov_0[0,0] - ds.gamma_vars[0].mean ** 2]) * (assumed_dof - ds.d)
    ds.sigma_star_vars[0].nu = cov_1[-1,-1]           

    ds.sigma_star_vars[0].first_moment = ds.sigma_star_vars[0].first_mom()
    ds.sigma_star_vars[0].second_moment = ds.sigma_star_vars[0].second_mom()     


    ds.sigma_star_vars[1].prior_scale = np.array([cov_1[0,0] - ds.gamma_vars[1].mean ** 2]) * (assumed_dof - ds.d)  #np.array([[cov_1[0,0] * (assumed_dof - ds.d)]])
    ds.sigma_star_vars[1].dof = 5
    ds.sigma_star_vars[1].prior_dof = 5
    ds.sigma_star_vars[1].scale = np.array([cov_1[0,0] - ds.gamma_vars[1].mean ** 2]) * (assumed_dof - ds.d)
    ds.sigma_star_vars[1].nu = cov_1[-1,-1]

    ds.sigma_star_vars[1].first_moment = ds.sigma_star_vars[1].first_mom()
    ds.sigma_star_vars[1].second_moment = ds.sigma_star_vars[1].second_mom()

    full_sigma_inv_estimates = [np.linalg.inv(cov_mat) for cov_mat in [cov_0, cov_1]]

    
    
    for i, (r_var, z_var) in enumerate(zip(ds.r_vars, ds.z_vars)):
        C=0
        D=0
        norm_datapoint = ds.normed_embds[i]
        norm_datapoint = norm_datapoint.reshape(-1, 1)

        for k in range (0, len(ds.z_vars[i].probs)):
            data_group = k
            sigma = ds.sigma_star_vars[data_group]
            γ = ds.gamma_vars[data_group]
            μ = ds.means_vars[data_group]

            sigma_inv = full_sigma_inv_estimates[data_group]

            C += z_var.probs[k] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), norm_datapoint)
            D += z_var.probs[k] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), μ.mean)


        r_var.alpha = C / 2
        r_var.beta = D / C

        r_var.update_moments(norm_datapoint)

        # r_var.first_moment = np.linalg.norm(ds.embds[i])

        
        # initialise phi variables

    # ds.phi_var.conc[0] = n_samples // 2
    # ds.phi_var.conc[1] = n_samples // 2 


    ds.dataset_vi(max_iter=5, real_cov=cov_0)






    
