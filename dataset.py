import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from tqdm import tqdm

from mu_dist import Mu
from sigma_dist import Sigma_Star
from gamma_dist import Gamma

from r_dist import R
from z_dist import Z
from phi_dist import Phi

from dataset_initialisation import GMM_Init

from scipy.stats import beta

from sklearn.metrics import adjusted_rand_score




np.random.seed(44)

class Dataset():
    
    def __init__(self, adj_mat, emb_dim, K=None, synthetic=False):
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
        
        # for synthetic dataset
        self.synthetic=synthetic
        self.reversed_labels=False


            
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


            # data = data.reshape(1, -1)

            # # Get predicted probabilities from the fitted GMM model
            # predicted_probs = self.gmm.fitted_gmm.predict_proba(data)[0]

            # # Find the index of the maximum probability
            # max_index = np.argmax(predicted_probs)

            # # Set the maximum probability to 0.8
            # new_probs = np.full(self.K, (1 - 0.8) / (self.K - 1))  # Distribute the remaining 0.2 equally among other groups
            # new_probs[max_index] = 0.8

            # # Update the probs attribute for the current z_var
            # z_var.probs = new_probs


        for k in range(self.K):
            self.means_vars[k].prior_cov = mu_cov
            self.means_vars[k].mean = gmm.cluster_centres[k]
            self.means_vars[k].cov =  mu_cov

            #print(f"mu_cov {k} det:", np.linalg.det(mu_cov))

            self.sigma_star_vars[k].prior_scale = gmm.sigma_star_estimates[k] * (dof - self.d) #scale_mat
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

        # print("sigma inv estimates:", full_sigma_inv_estimates)


        for i, (r_var, z_var) in enumerate(zip(self.r_vars, self.z_vars)):
            C=0
            D=0

            norm_datapoint = self.normed_embds[i]
            norm_datapoint = norm_datapoint.reshape(-1, 1)

            D_collection = []

            

            for k in range (0, len(z_var.probs)):
                data_group = k
                sigma = self.sigma_star_vars[data_group]
                γ = self.gamma_vars[data_group]
                μ = self.means_vars[data_group]

                sigma_inv = full_sigma_inv_estimates[data_group]


                C += z_var.probs[k] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), norm_datapoint)
                D += z_var.probs[k] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), μ.mean)
                D_collection.append(z_var.probs[k] * np.matmul(np.matmul(norm_datapoint.T, sigma_inv), μ.mean))

            

            r_var.alpha = C / 2
            r_var.beta = D / C

            # print(f"r_var.alpha: {r_var.alpha}")
            # assert False
            r_var.alpha = min(np.array([[20.0]]), r_var.alpha)

            r_var.update_moments()

        # initialise phi variables

        # self.phi_var.vi(self.z_vars)

        # for k in range(self.K):
        #     # count number of labels in group k
        #     num_labels = sum([1 for lab in gmm.labels if lab == k])
        #     self.phi_var.conc[k] = num_labels + self.phi_var.prior_conc[k]
        


    
    def dataset_vi(self, max_iter=1000):


        self.gaussian_mm_init() # initialize the means, sigma_star and gamma distributions

        if max_iter == 0:
            return

        self.print_progress(0)

        # for epoch in tqdm(range(max_iter), desc="Performing VI"):
        for epoch in range(1,max_iter+1):

             # for k in range(self.K):

            for k in range(self.K):

                self.means_vars[k].vi(self.z_vars, self.r_vars, self.sigma_star_vars[k], self.gamma_vars[k], self.phi_var, self)
                self.sigma_star_vars[k].vi(self.z_vars, self.r_vars, self.means_vars[k], self.gamma_vars[k], self.phi_var, self)
                self.gamma_vars[k].vi(self.z_vars, self.r_vars, self.sigma_star_vars[k], self.means_vars[k], self.phi_var, self)

            for i in range(self.N):
                self.r_vars[i].vi(self.z_vars[i], self.sigma_star_vars, self.gamma_vars, self.means_vars, self.phi_var, self.normed_embds[i], self.embds[i]) 
                self.z_vars[i].vi(self.r_vars[i], self.means_vars, self.sigma_star_vars, self.gamma_vars, self.normed_embds[i], self.phi_var, verbose=i<10)
            
            if epoch>5:
                self.phi_var.vi(self.z_vars)
            
            
            self.print_progress(epoch)
    
    def print_progress(self, epoch, num_els=10):

        predicted_labels = [np.argmax(z.probs) for z in self.z_vars]

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
                    Adjusted Rand Score: {adjusted_rand_score(self.true_labels, predicted_labels)}

                _____________________________________________________________________
                    
                    r_first_alpha: {[x.alpha for x in self.r_vars[:num_els]]}
                    r_first_beta: {[x.beta for x in self.r_vars[:num_els]]}
                    r_first moment: {[x.first_moment for x in self.r_vars[:num_els]]}
                    r_second moment: {[x.second_moment for x in self.r_vars[:num_els]]}

                    true r_values: {[np.linalg.norm(self.embds[i]) for i in range(num_els)]}

                 _____________________________________________________________________

                    phi_probs: {self.phi_var.conc}
                  
                  
                  """)
        
class Dataset_From_Files(Dataset):
    
    def __init__(self, emb_file, label_file, emb_dim=None, synthetic=False):
        self.embds, self.true_labels, self.true_names, self.K = self.get_data_from_files(emb_file, label_file,emb_dim)

        self.clean_data()

        self.N, self.d = self.embds.shape[0], self.embds.shape[1]

        self.normed_embds = self.embds / np.linalg.norm(self.embds, axis=1)[:, np.newaxis]
        self.best_k_means = None

        self.means_vars = [Mu(i, self.d) for i in range(self.K)]
        self.sigma_star_vars = [Sigma_Star(i, self.d) for i in range(self.K)]
        self.gamma_vars = [Gamma(i, self.d) for i in range(self.K)]
 
        self.r_vars = [R(self.d-1) for _ in range(self.N)]
        self.z_vars = [Z(self.d, self.K) for _ in range(self.N)]
        self.phi_var = Phi(self.K)
        
        # for synthetic dataset
        self.synthetic=synthetic
        self.reversed_labels=False
    
    def clean_data(self):
        filtered_embds = []
        filtered_labels = []
        filtered_names = []

        for embd, label, name in zip(self.embds, self.true_labels, self.true_names):
            if np.linalg.norm(embd) >= 1e-8:
                filtered_embds.append(embd)
                filtered_labels.append(label)
                filtered_names.append(name)

        unique_values = np.unique(filtered_labels)
        num_unique_values = len(unique_values)

        self.embds = np.array(filtered_embds)
        self.true_labels = np.array(filtered_labels)
        self.true_names = np.array(filtered_names)
        self.K = num_unique_values


    
    def get_data_from_files(self, emb_file, label_file, emb_dim=None):

        embds_df = pd.read_csv(emb_file)
        embds_df = embds_df.to_numpy()

        if emb_dim is not None:
            embds_df = embds_df[:,:emb_dim]

        labels_df = pd.read_csv(label_file)
        labels_df = labels_df.to_numpy()

        true_labels_df = labels_df[:,1]
        true_labels_df = np.array(true_labels_df, dtype=int)

        true_party_names = labels_df[:,2]

        unique_values = np.unique(true_labels_df)
        num_unique_values = len(unique_values)



        return embds_df, true_labels_df, true_party_names, num_unique_values
    
    
    
    def print_progress(self, epoch, num_els=10):

        predicted_labels = [np.argmax(z.probs) for z in self.z_vars]


        print(f"Iteration {epoch} results:\n")

        # Loop through each group to display means and covariance matrices
        for k in range(self.K):
            print(f"μ_{k}_mean: {self.means_vars[k].mean}")
            print(f"μ_{k}_cov: {self.means_vars[k].cov}\n")
            print()

        print("_____________________________________________________________________")
        print()

        # Loop through each group to display sigma variables
        for k in range(self.K):
            print(f"sigma_{k}_scale: {self.sigma_star_vars[k].scale}")
            print(f"sigma_{k}_prior_scale: {self.sigma_star_vars[k].prior_scale}")
            print(f"sigma_{k}_dof: {self.sigma_star_vars[k].dof}")
            print(f"sigma_{k}_prior_dof: {self.sigma_star_vars[k].prior_dof}")
            print(f"sigma_{k}_first_moment: {self.sigma_star_vars[k].first_moment}")
            print(f"sigma_{k}_mode: {self.sigma_star_vars[k].mode()}\n")
            print()

        print("_____________________________________________________________________")
        print()

        # Loop through each group to display gamma variables
        for k in range(self.K):
            print(f"gamma_{k}_mean: {self.gamma_vars[k].mean}")
            print(f"gamma_{k}_cov: {self.gamma_vars[k].cov}\n")
            print()

        print("_____________________________________________________________________")
        print()

        # Display z probs
        print(f"First {num_els} z probs: {[x.probs for x in self.z_vars[:num_els]]}\n")
        print(f"Adjusted Rand Score: {adjusted_rand_score(self.true_labels, predicted_labels)}")
        print()

        print("_____________________________________________________________________")
        print()

        # Display r vars
        print(f"r_first_alpha: {[x.alpha for x in self.r_vars[:num_els]]}")
        print(f"r_first_beta: {[x.beta for x in self.r_vars[:num_els]]}")
        print(f"r_first moment: {[x.first_moment for x in self.r_vars[:num_els]]}")
        print(f"r_second moment: {[x.second_moment for x in self.r_vars[:num_els]]}")
        print(f"true r_values: {[np.linalg.norm(self.embds[i]) for i in range(num_els)]}\n")
        print()

        print("_____________________________________________________________________")
        print()

        # Display phi probabilities
        print(f"phi_probs: {self.phi_var.conc}")


if __name__ == '__main__':

    ds = Dataset_From_Files(emb_file="./data_files/camera18_embedding.csv",
                            label_file="./data_files/camera18_node_labels.csv",
                            emb_dim=4)
    
    ds.dataset_vi(max_iter=30)

 







    
