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

class Dataset():
    
    def __init__(self, adj_mat, emb_dim, K=None):
        self.adj_mat = adj_mat
        self.d = emb_dim
        self.N = len(adj_mat)
        self.K = K
        self.embds = self.spectral_emb() # of size (N, self.d) where N is the number of samples
        # print("self.embds", self.embds)
        self.normed_embds = self.embds / np.linalg.norm(self.embds, axis=1)[:, np.newaxis]
        self.best_k_means = None

        self.means_vars = [Mu(i, self.d) for i in range(self.K)]
        self.sigma_star_vars = [Sigma_Star(i, self.d) for i in range(self.K)]
        self.gamma_vars = [Gamma(i, self.d) for i in range(self.K)]
 
        self.r_vars = [R(self.d-1) for _ in range(self.N)]
        self.z_vars = [Z(self.d, self.K) for _ in range(self.N)]
        self.phi_var = Phi(self.K)


            
    def spectral_emb(self):
        eigvals, eigvecs = np.linalg.eig(self.adj_mat)
        sorted_indexes = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[sorted_indexes]
        eigvecs = eigvecs[:,sorted_indexes]
        embedding_dim = self.d
        eigvecs_trunc = eigvecs[:,:embedding_dim]
        eigvals_trunc = np.diag(np.sqrt(np.abs(eigvals[:embedding_dim])))
        spectral_embedding = eigvecs_trunc @ eigvals_trunc
        return spectral_embedding
    
    def gaussian_mm_init(self):


        gmm = GMM_Init(self.normed_embds, n_components=self.K)
        self.gmm = gmm

        mu_cov = gmm.mu_prior_cov_estimate()
        gamma_cov = gmm.gamma_prior_cov_estimate()
        scale_mat, dof = gmm.sigma_prior_params_estimate()

        # initialise the z variables

        for z_var, data in zip(self.z_vars, self.normed_embds):
            data = data.reshape(1, -1)

            predicted_probs = self.gmm.fitted_gmm.predict_proba(data)
            predicted_probs = predicted_probs[0]
            z_var.probs = predicted_probs
        
        # initialise the mean gamma and sigma


        for k in range(self.K):
            self.means_vars[k].prior_cov = mu_cov
            self.means_vars[k].mean = gmm.cluster_centres[k]
            self.means_vars[k].cov =  mu_cov

            #print(f"mu_cov {k} det:", np.linalg.det(mu_cov))

            self.sigma_star_vars[k].prior_scale = scale_mat
            self.sigma_star_vars[k].prior_dof = dof

            self.sigma_star_vars[k].scale = gmm.sigma_star_estimates[k] * (dof - self.d)
            self.sigma_star_vars[k].dof = max(dof, self.d +3) # gmm.sigma_star_inits[k] * (dof - self.d)

            self.sigma_star_vars[k].nu = gmm.cluster_covs[k][-1,-1]

            #print(f"scale_mat {k} det:", np.linalg.det(scale_mat))
                

            self.gamma_vars[k].prior_cov = gamma_cov
            self.gamma_vars[k].mean = gmm.gamma_estimates[k]
            self.gamma_vars[k].cov = gamma_cov

            self.gamma_vars[k].nu = gmm.cluster_covs[k][-1,-1]


        # initialise the r variables

        full_sigma_inv_estimates = [np.linalg.inv(cov_mat) for cov_mat in gmm.cluster_covs]

        print("sigma inv estimates:", full_sigma_inv_estimates)


        for i, (r_var, label) in enumerate(zip(self.r_vars, gmm.labels)):
            C=0
            D=0
            norm_datapoint = self.normed_embds[i]
            norm_datapoint = norm_datapoint.reshape(-1, 1)

            for k in range (0, len(self.z_vars[i].probs)):
                data_group = k
                sigma = ds.sigma_star_vars[data_group]
                γ = ds.gamma_vars[data_group]
                μ = ds.means_vars[data_group]

                sigma_inv = full_sigma_inv_estimates[data_group]

            C += ds.phi_var.conc[k] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), norm_datapoint)
            D += ds.phi_var.conc[k] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), μ.mean)


            r_var.alpha = C / 2
            r_var.beta = D / C

            r_var.update_moments()
        
        # initialise phi variables

        for k in range(self.K):
            # count number of labels in group k
            num_labels = sum([1 for lab in gmm.labels if lab == k])
            self.phi_var.conc[k] = num_labels + self.phi_var.prior_conc[k]
        


    
    def dataset_vi(self, max_iter=1000):


        self.gaussian_mm_init() # initialize the means, sigma_star and gamma distributions

        self.print_progress(0)

        for i, (z_var, data) in enumerate(zip(self.z_vars, self.normed_embds)):
            data = data.reshape(1, -1)

            predicted_probs = self.gmm.fitted_gmm.predict_proba(data)
            predicted_probs = predicted_probs[0]
            z_var.probs = np.array([1.0,0.0]) if i % 2 == 0 else np.array([0.0,1.0])  

        # for epoch in tqdm(range(max_iter), desc="Performing VI"):
        for epoch in range(1,max_iter+1):

             # for k in range(self.K):

            for k in range(self.K):

                self.means_vars[k].vi(self.z_vars, self.r_vars, self.sigma_star_vars[k], self.gamma_vars[k], self.phi_var, self)
                self.sigma_star_vars[k].vi(self.z_vars, self.r_vars, self.means_vars[k], self.gamma_vars[k], self.phi_var, self)
                self.gamma_vars[k].vi(self.z_vars, self.r_vars, self.sigma_star_vars[k], self.means_vars[k], self.phi_var, self)

            for i in range(self.N):
                self.r_vars[i].vi(self.z_vars[i], self.sigma_star_vars, self.gamma_vars, self.means_vars, self.phi_var, self.normed_embds[i]) 
                # self.z_vars[i].vi(self.r_vars[i], self.means_vars, self.sigma_star_vars, self.gamma_vars, self.normed_embds[i], self.phi_var, verbose=i<10)
            
            self.phi_var.vi(self.z_vars)
            
            
            self.print_progress(epoch)
    
    def print_progress(self, epoch, num_els=10):

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


                    sigma_1_scale: {self.sigma_star_vars[1].scale}
                    sigma_1_prior_scale: {self.sigma_star_vars[1].prior_scale}
                    sigma_1_dof: {self.sigma_star_vars[1].dof}
                    sigma_1_prior_dof: {self.sigma_star_vars[1].prior_dof}
                    sigma_1_first_moment: {self.sigma_star_vars[1].first_moment}
                    sigma_1_mode: {self.sigma_star_vars[1].mode()}


                _____________________________________________________________________

                    gamma_0_mean: {self.gamma_vars[0].mean}
                    gamma_0_cov: {self.gamma_vars[0].cov}

                    gamma_1_mean: {self.gamma_vars[1].mean}
                    gamma_1_cov: {self.gamma_vars[1].cov}

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


class Synthetic_data(Dataset):

    def __init__(self, μ_1, μ_2, prior, N_t=1000):
        # prior has to be a function which takes no arguments and spits out a sample

        # assert len(μ_1) == 2, "μ_1 and μ_2 have to be a 2D vector"
        # assert len(μ_1) == len(μ_2), "μ_1 and μ_2 have to be a 2D vector"

        self.μ_1 = μ_1
        self.μ_2 = μ_2
        self.mean_len = len(μ_1)
        self.N_t=N_t
        self.adj_mat, self.bern_params = self.simulate_adj_mat(prior, μ_1, μ_2)
         # number of data points

        super().__init__(self.adj_mat, emb_dim=self.mean_len, K=2)


    def find_delta_inv(self, μ_1, μ_2, exp_rho):
        μ_1_outer = np.outer(μ_1, μ_1)
        μ_2_outer = np.outer(μ_2, μ_2)

        Δ = exp_rho**2 * (1 / 2) * (μ_1_outer + μ_2_outer)

        Δ_inv = np.linalg.inv(Δ) 
        return Δ_inv

    def exp_X1_inner_func(self, x, ρ, μ):
        return (np.dot(x, ρ*μ) - (np.dot(x, ρ*μ)**2)) * np.outer(ρ*μ, ρ*μ)

    def covariance_estimate(self, x, μ_1, μ_2, prior, exp_rho, N_ρ=1000, N_t=1000):
        ρ_samples_1 = np.array([prior() for _ in range(N_ρ)])
        ρ_samples_2 = np.array([prior() for _ in range(N_ρ)])
        μ_1_integral_estimate = (1 / N_ρ) * sum(self.exp_X1_inner_func(x, ρ, μ_1) for ρ in ρ_samples_1)
        μ_2_integral_estimate = (1 / N_ρ) * sum(self.exp_X1_inner_func(x, ρ, μ_2) for ρ in ρ_samples_2)
        exp_X1_func_estimate = 0.5 * (μ_1_integral_estimate + μ_2_integral_estimate)
        Δ_inv = self.find_delta_inv(μ_1, μ_2, exp_rho)
        return (Δ_inv @ exp_X1_func_estimate @ Δ_inv) / N_t

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    def simulate_adj_mat(self, prior, μ_1, μ_2):
        μ_mat = np.stack((μ_1, μ_2), axis=1)
        # bern_params = [(prior(), np.random.randint(0,2)) for _ in range(self.N_t)]
        bern_params = [(prior(), i % 2) for i in range(self.N_t)]
        adj_mat = np.zeros((self.N_t, self.N_t))

        for i in range(self.N_t):
            ρ_i, μ_i = bern_params[i][0], μ_mat[:, bern_params[i][1]]
            for j in range(i):
                ρ_j, μ_j = bern_params[j][0], μ_mat[:, bern_params[j][1]]

                adj_mat[i,j] = np.random.binomial(1, ρ_i * ρ_j * np.dot(μ_i, μ_j))

                adj_mat[j,i] = adj_mat[i,j]
            
            adj_mat[i,i] = 1
        
        assert self.check_symmetric(adj_mat)

        return adj_mat, bern_params
    
    def spectral_emb(self):
        μ_mat = np.stack((self.μ_1, self.μ_2), axis=1)
        eigvals, eigvecs = np.linalg.eig(self.adj_mat)
        sorted_indexes = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[sorted_indexes]
        eigvecs = eigvecs[:,sorted_indexes]
        embedding_dim = self.d # should be 2
        eigvecs_trunc = eigvecs[:,:embedding_dim]
        eigvals_trunc = np.diag(np.sqrt(np.abs(eigvals[:embedding_dim])))
        spectral_embedding = eigvecs_trunc @ eigvals_trunc
        true_means = np.zeros((self.N_t, self.mean_len))

        for i in range(self.N_t):
            ρ_i, μ_i = self.bern_params[i][0], μ_mat[:, self.bern_params[i][1]]
            true_means[i, :] =  ρ_i * μ_i

        # print("true_means", true_means)

        best_orthog_mat = orthogonal_procrustes(spectral_embedding, true_means)

        spectral_embedding = spectral_embedding @ best_orthog_mat[0]

        # print("spectral_embedding", spectral_embedding)

        # print("min norm", min(spectral_embedding, key=np.linalg.norm))

        return spectral_embedding


if __name__ == '__main__':

    μ_1 = np.array([0.75, 0.25])
    μ_2 = np.array([0.25, 0.75])

    α = 7
    β = 2
    prior = lambda : 0.5 #beta.rvs(α, β)

    ds = Synthetic_data(μ_1, μ_2, prior, N_t=1000)

    # Set means to be the true value and see what happens

    μ_list = [μ_1, μ_2]

    for k in range(ds.K):
        ds.means_vars[k].mean = μ_list[k]


    ds.dataset_vi(max_iter=5)

    ##true_labels = ds.true_labels
    # max_probs = [np.argmax(z.probs) for z in ds.z_vars]
    # label_difference = np.sum(np.array(true_labels) != np.array(max_probs))

    # print()
    # value_counts = np.bincount(max_probs)
    # print(value_counts)
    # print("Label Difference:", label_difference)

    # print("True Labels:", true_labels[:10])

    # print("Max Probs:", max_probs[:100])







    
