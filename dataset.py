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

from scipy.stats import beta, entropy

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
import matplotlib.pyplot as plt




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
    
    def compute_angles(self):
        d = self.normed_embds.shape[1]  # Dimension of the vectors
        angles = np.zeros((self.normed_embds.shape[0], d - 1))

        for i, vec in enumerate(self.normed_embds):
            for j in range(d - 1):
                if j == 0:
                    norm = np.linalg.norm(vec[:j+2])
                    theta = np.arccos(vec[j+1] / norm)
                    angles[i, j] = theta if vec[j] >= 0 else 2 * np.pi - theta
                else:
                    norm = np.linalg.norm(vec[:j+2])
                    angles[i, j] = 2 * np.arccos(vec[j+1] / norm)

        return angles
    
    def gaussian_mm_init(self):

        self.angles = self.compute_angles()

        kmeans = KMeans(n_clusters=self.K, random_state=42)  # random_state for reproducibility

        kmeans_fitted = kmeans.fit(self.embds)

        mutual_info_score = normalized_mutual_info_score(self.true_labels, kmeans_fitted.labels_)
        fowlkes_mallows = fowlkes_mallows_score(self.true_labels, kmeans_fitted.labels_)

        print(f"==>> mutual_info_score: {mutual_info_score:.3f}")
        print(f"==>> fowlkes_mallows: {fowlkes_mallows:.3f}")

        assert False

        gmm = GMM_Init(self.angles, n_components=self.K)

        adj_rand_score = adjusted_rand_score(self.true_labels, gmm.labels)
        print(f"==>> adj_rand_score: {adj_rand_score}")
        assert False


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
            
            self.phi_var.vi(self.z_vars)
            
            
            self.print_progress(epoch)
    
    def print_progress(self, epoch, num_els=10):

        predicted_labels = [np.argmax(z.probs) for z in self.z_vars]
        entropies = [entropy(z_i.probs) for z_i in self.z_vars]

        # Calculate average entropy
        average_entropy = sum(entropies) / len(entropies)

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

                    First 10 z probs: {[x.probs for x in self.z_vars[:num_els]]}
                    Adjusted Rand Score: {adjusted_rand_score(self.true_labels, predicted_labels)}
                    Average Entropy: {average_entropy}

                _____________________________________________________________________
                  
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
        entropies = [entropy(z_i.probs) for z_i in self.z_vars]

        # Calculate average entropy
        average_entropy = sum(entropies) / len(entropies)

        # print(f"""Iteration {epoch} results:
                  
        #             μ_0_mean: {self.means_vars[0].mean}
        #             μ_0_cov: {self.means_vars[0].cov}

        #             μ_1_mean: {self.means_vars[1].mean}
        #             μ_1_cov: {self.means_vars[1].cov}


        #          _____________________________________________________________________

        #             sigma_0_scale: {self.sigma_star_vars[0].scale}
        #             sigma_0_prior_scale: {self.sigma_star_vars[0].prior_scale}
        #             sigma_0_dof: {self.sigma_star_vars[0].dof}
        #             sigma_0_prior_dof: {self.sigma_star_vars[0].prior_dof}
        #             sigma_0_first_moment: {self.sigma_star_vars[0].first_moment}
        #             sigma_0_mode: {self.sigma_star_vars[0].mode()}


        #             sigma_1_scale: {self.sigma_star_vars[1].scale}
        #             sigma_1_prior_scale: {self.sigma_star_vars[1].prior_scale}
        #             sigma_1_dof: {self.sigma_star_vars[1].dof}
        #             sigma_1_prior_dof: {self.sigma_star_vars[1].prior_dof}
        #             sigma_1_first_moment: {self.sigma_star_vars[1].first_moment}
        #             sigma_1_mode: {self.sigma_star_vars[1].mode()}


        #         _____________________________________________________________________

        #             gamma_0_mean: {self.gamma_vars[0].mean}
        #             gamma_0_cov: {self.gamma_vars[0].cov}

        #             gamma_1_mean: {self.gamma_vars[1].mean}
        #             gamma_1_cov: {self.gamma_vars[1].cov}

        #         _____________________________________________________________________

        #             First 10 z probs: {[x.probs for x in self.z_vars[:num_els]]}
        #             Adjusted Rand Score: {adjusted_rand_score(self.true_labels, predicted_labels)}
        #             Average Entropy: {average_entropy}

        #         _____________________________________________________________________
                    
        #             r_first_alpha: {[x.alpha for x in self.r_vars[:num_els]]}
        #             r_first_beta: {[x.beta for x in self.r_vars[:num_els]]}
        #             r_first moment: {[x.first_moment for x in self.r_vars[:num_els]]}
        #             r_second moment: {[x.second_moment for x in self.r_vars[:num_els]]}

        #             true r_values: {[np.linalg.norm(self.embds[i]) for i in range(num_els)]}

        #          _____________________________________________________________________

        #             phi_probs: {self.phi_var.conc}
                  
                  
        #           """)

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

                    Adjusted Rand Score: {adjusted_rand_score(self.true_labels, predicted_labels)}
                    Average Entropy: {average_entropy}

                _____________________________________________________________________
                  
                  """)


if __name__ == '__main__':

    ds = Dataset_From_Files(emb_file="./data_files/camera18_embedding.csv",
                            label_file="./data_files/camera18_node_labels.csv",
                            emb_dim=4)
    
    ds.dataset_vi(max_iter=1)



    # Spectral Embeddings
    
    # plt.rc('font', size=8)  # Default text sizes
    # plt.rc('axes', titlesize=8)  # Axes title font size
    # plt.rc('legend', fontsize=8)  # Legend font size
    # plt.rc('xtick', labelsize=10)  # X-axis tick label font size
    # plt.rc('ytick', labelsize=10)  # Y-axis tick label font size
    # plt.rcParams['mathtext.fontset'] = 'stix'
    # plt.rcParams['font.family'] = 'STIXGeneral'

    # max_label = 7  # Highest group label, assuming labels are 0 through 6
    # colors = ['red', 'blue', 'green', 'olive', 'orange', 'purple', 'cyan']
    # markers = ['o', '^', 's', 'x', '+', 'd', '*']

    # # Extract groups and names, only including non-empty groups in the plot
    # groups = []
    # group_names = []
    # for i in range(max_label + 1):
    #     if ds.true_labels[ds.true_labels == i].size > 0:
    #         groups.append(ds.embds[ds.true_labels == i][:, :2])
    #         group_names.append(np.unique(ds.true_names[ds.true_labels == i])[0])

    # # Create the plot with specific figure size
    # fig, ax = plt.subplots(figsize=(8.4, 6))

    # # Plot each group with dynamic checking and labels
    # for i, group in enumerate(groups):
    #     ax.scatter(group[:, 0], group[:, 1], c=colors[i % len(colors)], marker=markers[i % len(markers)], label=group_names[i])

    # # Set axis labels and limits
    # ax.set_xlabel('X', fontsize=10)
    # ax.set_ylabel('Y', fontsize=10)
    # ax.set_xlim(-0.2, 1.0)
    # ax.set_ylim(-0.4, 0.4)

    # # Add grid for better visibility
    # ax.grid(True)

    # # Add legend only if there are valid groups
    # if groups:
    #     ax.legend()

    # # Adjust layout to prevent clipping
    # plt.tight_layout()

    # # Define the file path for saving the image, adjust as needed
    # plt.savefig('/Users/dariyankhan/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Work (one drive)/Imperial/year_4/M4R/images/Italy_Gov_Data/spectral_embeddings.pdf', bbox_inches='tight')

    # # Show the plot
    # plt.show()

    # Define the file path for saving the image, adjust as needed
    # Show the plot


    # plt.rc('font', size=8)  # Default text sizes
    # plt.rc('axes', titlesize=8)  # Axes title font size
    # plt.rc('legend', fontsize=8)  # Legend font size
    # plt.rc('xtick', labelsize=10)  # X-axis tick label font size
    # plt.rc('ytick', labelsize=10)  # Y-axis tick label font size
    # plt.rcParams['mathtext.fontset'] = 'stix'
    # plt.rcParams['font.family'] = 'STIXGeneral'

    # max_label = 7  # Highest group label, assuming labels are 0 through 6
    # colors = ['red', 'blue', 'green', 'olive', 'orange', 'purple', 'cyan']
    # markers = ['o', '^', 's', 'x', '+', 'd', '*']

    # # Extract groups and names, only including non-empty groups in the plot
    # groups = []
    # group_names = []
    # for i in range(max_label + 1):
    #     if ds.true_labels[ds.true_labels == i].size > 0:
    #         groups.append(ds.normed_embds[ds.true_labels == i][:, :2])
    #         group_names.append(np.unique(ds.true_names[ds.true_labels == i])[0])

    # # Create the plot with specific figure size
    # fig, ax = plt.subplots(figsize=(8.4, 6))

    # # Plot each group with dynamic checking and labels
    # for i, group in enumerate(groups):
    #     ax.scatter(group[:, 0], group[:, 1], c=colors[i % len(colors)], marker=markers[i % len(markers)], label=group_names[i])

    # # Set axis labels and limits
    # ax.set_xlabel('X', fontsize=10)
    # ax.set_ylabel('Y', fontsize=10)
    # ax.set_xlim(0.0, 1.2)
    # ax.set_ylim(-0.75, 0.75)

    # # Add grid for better visibility
    # ax.grid(True)

    # # Add legend only if there are valid groups
    # if groups:
    #     ax.legend()

    # # Adjust layout to prevent clipping
    # plt.tight_layout()

    # # Define the file path for saving the image, adjust as needed
    # plt.savefig('/Users/dariyankhan/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Work (one drive)/Imperial/year_4/M4R/images/Italy_Gov_Data/normed_spectral_embeddings.pdf', bbox_inches='tight')

    # # Show the plot
    # plt.show()






 
    







    
