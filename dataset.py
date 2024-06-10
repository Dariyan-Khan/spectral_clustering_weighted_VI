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

import geomstats.backend as gs
import geomstats.geometry.hypersphere as hypersphere
from geomstats.learning.frechet_mean import FrechetMean

np.random.seed(44)


def vector_projection(u, v):
        # Compute the dot product of vectors u and v
        dot_product = np.dot(v, u)
        
        # Compute the norm squared of vector u
        norm_u_squared = np.dot(u, u)
        
        # Calculate the projection of v onto u
        projection = (dot_product / norm_u_squared) * u
        
        return projection

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
 
        self.r_vars = [R(self.d-1, index) for index in range(self.N)]
        self.z_vars = [Z(self.d, self.K, index) for index in range(self.N)]
        self.phi_var = Phi(self.K)
        self.synthetic=True
        self.true_labels = None

        max_norm = max([np.linalg.norm(x) for x in self.embds])

        self.weights = [np.linalg.norm(x) / max_norm for x in self.embds]



    def dataset_vi(self, max_iter=10, run_init=False, real_cov=None):

        self.print_progress(epoch=0, num_els=10, real_cov=real_cov)
        

        for epoch in range(1, max_iter+1):

            for k in range(self.K):
                # pass
                self.means_vars[k].vi(self.z_vars, self.r_vars, self.sigma_star_vars[k], self.gamma_vars[k], self.phi_var,self.weights, self, real_cov=real_cov)
                self.sigma_star_vars[k].vi(self.z_vars, self.r_vars, self.means_vars[k], self.gamma_vars[k], self.phi_var, self.weights, self)
                self.gamma_vars[k].vi(self.z_vars, self.r_vars, self.sigma_star_vars[k], self.means_vars[k], self.phi_var, self.weights, self, real_cov=real_cov)

            for i in range(self.N):
                self.r_vars[i].vi(self.z_vars[i], self.sigma_star_vars, self.gamma_vars, self.means_vars, self.phi_var, self.weights, self.normed_embds[i], real_cov=real_cov) 
                self.z_vars[i].vi(self.r_vars[i], self.means_vars, self.sigma_star_vars, self.gamma_vars, self.normed_embds[i], self.phi_var, self.weights, verbose=i<10, real_cov=real_cov)
            
            self.phi_var.vi(self.z_vars, self.weights)


            for k in range(self.K):
                self.update_weights(k)
        
            self.print_progress(epoch, num_els=10, real_cov=real_cov)
        
    
    def update_weights(self, k):
        embds_in_cluster = []
        index_of_embds_in_cluster = []
        max_norm = 0
        sphere = hypersphere.Hypersphere(dim=self.d-1)

        for i in range(self.N):
            if np.argmax(self.z_vars[i].probs) == k:
                embds_in_cluster.append(self.embds[i])
                index_of_embds_in_cluster.append(i)
                max_norm = max(max_norm, np.linalg.norm(self.embds[i]))
        
        proj_embds_in_cluster = np.array([(embd /np.linalg.norm(embd)) * max_norm for embd in embds_in_cluster])

        proj_embds_in_cluster = np.array(embds_in_cluster)

        mean = FrechetMean(sphere)
        mean.fit(proj_embds_in_cluster)
        frechet_mean = mean.estimate_

        orthogonally_proj_points = [vector_projection(frechet_mean, embd) for embd in proj_embds_in_cluster]

        for i, (w_index, orth_proj) in enumerate(zip(index_of_embds_in_cluster, orthogonally_proj_points)):
            self.weights[w_index] = min(np.linalg.norm(orth_proj) / max_norm, 1.0)
    



            
            
    
    def print_progress(self, epoch , num_els=10, real_cov=None):
        

        ν = real_cov[-1,-1]
        real_gamma = np.array([real_cov[0, 1]]) / np.sqrt(ν)

        real_sigma_star = real_cov[0,0] - real_gamma ** 2 



        # num_elements = len(self.z_vars)
        # num_correct = 0
        # for i in range(num_elements):
        #     if i % 2 == 0:
        #         num_correct += self.z_vars[i].probs[0] > self.z_vars[i].probs[1]
        #     else:
        #         num_correct += self.z_vars[i].probs[1] > self.z_vars[i].probs[0]

        
        num_elements = len(self.z_vars)
        num_correct = 0
        #print(f"==>> self.true_labels: {self.true_labels}")
        if self.synthetic:
            for (z_var, true_label) in zip(self.z_vars, self.true_labels):
                num_correct += z_var.probs[true_label] > 0.5
        
        fraction_correct = num_correct / num_elements

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
                    fraction correct: {fraction_correct}


                _____________________________________________________________________
                    
                    r_first_alpha: {[x.alpha for x in self.r_vars[:num_els]]}
                    r_first_beta: {[x.beta for x in self.r_vars[:num_els]]}
                    r_first moment: {[x.first_moment for x in self.r_vars[:num_els]]}
                    r_second moment: {[x.second_moment for x in self.r_vars[:num_els]]}

                    true r_values: {[np.linalg.norm(self.embds[i]) for i in range(num_els)]}

                    MLE_r_values: {[x.MLE() for x in self.r_vars[:num_els]]}

                 _____________________________________________________________________

                    phi_probs: {self.phi_var.conc}

                _____________________________________________________________________

                    weights: {self.weights[:num_els]}
                  
                  
                  """)



if __name__ == '__main__':


    # μ_1 = μ_1 / np.linalg.norm(μ_1)
    # μ_2 = μ_2 / np.linalg.norm(μ_2)

    μ_0 = np.array([0.75,0.25])
    cov_0 = np.array([[0.1, 0.05], [0.05, 0.1]])

    μ_1 = np.array([0.25, 0.75])
    cov_1 = np.array([[0.1, 0.05], [0.05, 0.1]])

    gamma_prior_cov = np.array([0.001]) #0.01*np.array([0.1])

    ν = cov_0[1,1]


    # Number of samples to generate
    n_samples = 1000

    # Generate samples alternately
    samples = np.zeros((n_samples, 2))
    true_labels = np.zeros(n_samples, dtype=int)

    # for i in range(n_samples):
    #     if i % 2 == 0:
    #         samples[i] = np.random.multivariate_normal(μ_0, cov_0)
    #         true_labels[i] = 0

    #     else:
    #         samples[i] = np.random.multivariate_normal(μ_1, cov_1)
    #         true_labels[i] = 1

        
    for i in range(n_samples):
        if np.random.random() < 0.5:
            samples[i] = np.random.multivariate_normal(μ_0, cov_0)
            true_labels[i] = 0

        else:
            samples[i] = np.random.multivariate_normal(μ_1, cov_1)
            true_labels[i] = 1

    #print(f"==>> samples: {samples}")

    
    ds = Dataset(samples, emb_dim=2, N=n_samples, K=2)
    ds.true_labels = true_labels

    # for i in range(0,len(ds.z_vars)):
    #     ds.z_vars[i].probs = [0.8, 0.2] if i % 2 == 0 else [0.2, 0.8]
    
    for i in range(0,len(ds.z_vars)):
        if ds.true_labels[i] == 0:
            ds.z_vars[i].probs = [0.8, 0.2] 
        else:
            ds.z_vars[i].probs = [0.2, 0.8]

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
    
        C = np.reshape(C, -1)
        D = np.reshape(D, -1)

        r_var.alpha = C / 2
        r_var.beta = D / C

        r_var.update_moments(norm_datapoint)

        # r_var.first_moment = np.linalg.norm(ds.embds[i])

        
        # initialise phi variables

    # ds.phi_var.conc[0] = n_samples // 2
    # ds.phi_var.conc[1] = n_samples // 2 


    ds.dataset_vi(max_iter=10, real_cov=cov_0)






    
