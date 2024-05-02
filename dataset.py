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


class  Dataset():
    
    def __init__(self, adj_mat, emb_dim, K=None):
        self.adj_mat = adj_mat
        self.d = emb_dim
        self.N = len(adj_mat)
        self.K = K
        self.embds = self.spectral_emb() # of size (N, self.d) where N is the number of samples
        self.normed_embds = self.embds / np.linalg.norm(self.embds, axis=1)[:, np.newaxis]
        self.best_k_means = None
        self.generate_best_k_means()

        self.means_vars = [Mu(i) for i in range(self.K)]
        self.sigma_star_vars = [Sigma_Star(i, self.d) for i in range(self.K)]
        self.gamma_vars = [Gamma(i, self.d) for i in range(self.K)]
 
        self.r_vars = [R(self.d) for _ in range(self.N)]
        self.z_vars = [Z(self.d, self.K) for _ in range(self.N)]
        self.phi_vars = [Phi(self.K) for _ in range(self.N)]


            
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
    
    def k_means_init(self, clusters_to_check=list(range(2, 11))):
        # Initialize K centroids by randomly selecting K points from the dataset

        best_silh_score = -10
        best_num_clusters = 1
        best_k_means = None

        for n in tqdm(clusters_to_check, desc="Performing KMeans"):

            kmeans = KMeans(n_clusters=n, random_state=42)  # random_state for reproducibility

            kmeans_fitted = kmeans.fit(self.embds)

            curr_silh_score = silhouette_score(self.embds, kmeans_fitted.predict(self.embds))

            if curr_silh_score > best_silh_score:
                best_silh_score = curr_silh_score
                best_num_clusters = n
                best_k_means = kmeans_fitted
        
        
        return best_k_means, best_num_clusters
    

    
    def generate_best_k_means(self):

        if self.K is None:
            best_k_means, best_num_clusters = self.k_means_init(clusters_to_check=list(range(2, min(11, self.N))))
            self.K = best_num_clusters
        
        else:
            kmeans = KMeans(n_clusters=self.K, random_state=42)  # random_state for reproducibility
            best_k_means = kmeans.fit(self.embds)
        
        self.best_k_means = best_k_means
        


class Synthetic_data(Dataset):

    def __init__(self, adj_mat, μ_1, μ_2, prior, N_t=1000):
        # prior has to be a function which takes no arguments and spits out a sample

        assert len(μ_1) == 2, "μ_1 and μ_2 have to be a 2D vector"
        assert len(μ_1) == len(μ_2), "μ_1 and μ_2 have to be a 2D vector"

        self.μ_1 = μ_1
        self.μ_2 = μ_2
        self.adj_mat, self.bern_params = self.simulate_adj_mat(prior, μ_1, μ_2)
        self.N_t=N_t # number of data points

        super().__init__(adj_mat, 2, 2)


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
        bern_params = [(prior(), np.random.randint(0,2)) for _ in range(self.N_t)]
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
        true_means = np.zeros((self.N_t, 2))

        for i in range(self.N_t):
            ρ_i, μ_i = self.bern_params[i][0], μ_mat[:, self.bern_params[i][1]]
            true_means[i, :] =  ρ_i * μ_i

        best_orthog_mat = orthogonal_procrustes(spectral_embedding, true_means)
        spectral_embedding = spectral_embedding @ best_orthog_mat[0]
        return spectral_embedding


if __name__ == '__main__':

    adj_matrix = [
    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 0, 1, 0]
]

    ds = Dataset(
        adj_matrix,
        emb_dim=3,
        K=2
    )

    b_k_means = ds.best_k_means

    print(b_k_means.cluster_centers_)
    print(b_k_means.labels_)