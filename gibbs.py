from dataset import Synthetic_data
import numpy as np
from scipy.stats import multivariate_normal, invwishart, beta, truncnorm
from dataset_initialisation import GMM_Init

class GibbsDataset(Synthetic_data):

    def __init__(self, N_t, K):
        μ_1 = np.array([0.75, 0.25])
        μ_2 = np.array([0.25, 0.75])

        α = 2
        β = 7
        prior = lambda : np.random.uniform(0.8, 1)

        super().__init__(μ_1, μ_2, prior, N_t=N_t)

        n_samples = 1000
        samples = np.zeros((n_samples, 2))
        true_labels = np.zeros(n_samples, dtype=int)
        cov_1 = np.array([[0.01, 0.005], [0.005, 0.01]])
        cov_2 = np.array([[0.01, 0.005], [0.005, 0.01]])

        for i in range(n_samples):
            if i % 2 == 0:
                samples[i] = np.random.multivariate_normal(μ_1, cov_1)
                true_labels[i] = 0

            else:
                samples[i] = np.random.multivariate_normal(μ_2, cov_2)
                true_labels[i] = 1
        
        self.embds = samples
        self.normed_embds = self.embds / np.linalg.norm(self.embds, axis=1).reshape(-1, 1)

        self.mean_samples = []
        self.sigma_samples = []
        self.r_samples = []
        self.z_samples = []
        self.pi_samples = []
        self.K = K

        self.prior_df = 2
        self.prior_scale_mat = np.eye(self.mean_len)

        self.pi_prior = 1.0 / self.K

    def Sigma_prior(self, x):
        return invwishart.pdf(x, self.prior_df, scale=self.prior_scale_mat)

    
    def calculate_density(self, index, mu_zi, Sigma_zi, verbose=False, proposal=False):
        """
        Calculate the probability density of the vector tilde{x}    r_i for a multivariate normal distribution
        characterized by mean mu_zi and covariance Sigma_zi.
        """
        if not proposal:
            curr_r_i = self.r_samples[-1][index]
            d = len(mu_zi)
            density = curr_r_i **(d-1) * multivariate_normal.pdf(self.embds[index], mean=mu_zi, cov=Sigma_zi)
        
        else:
            # curr_r_i = i
            d = len(mu_zi)
            density = index **(d-1) * multivariate_normal.pdf(index, mean=mu_zi, cov=Sigma_zi)

        if verbose:
            pass
            # print(Sigma_zi, "Sigma_zi")
            # print(self.embds[index], "embds[i]")
            # print(mu_zi, "mu_zi")
            # print(curr_r_i, "curr_r_i")
            # print(f"==>> density: {density}")
        # assert False
        return density
        


    def z_update(self, i):
        z_probs = [] # list to contain z_probs that we then sample from
        #print(self.pi_samples, "pi samples")
        for k in range(self.K):
            z_probs.append(self.pi_samples[-1][k] * self.calculate_density(i, self.mean_samples[-1][k], self.sigma_samples[-1][k]))
        
        z_probs = np.array(z_probs)

        if np.isnan(z_probs).any():
            print(z_probs, "z_probs")
            z_probs = np.array([0.5, 0.5])
            #assert False
        
        try:
            z_probs = z_probs / np.sum(z_probs)

        # print(z_probs, "z_probs")
            z_probs = z_probs.reshape(-1)

        except Exception:
            z_probs = np.array([0.5, 0.5])

        if np.isnan(z_probs).any():
            print(z_probs, "z_probs")
            z_probs = np.array([0.5, 0.5])

        z_i = np.random.choice(self.K, p=z_probs)
        return z_i
    
    
    def sample_r_i(self, i, mu_zi, Sigma_zi):
        """
        Generate a sample for r_i using a truncated normal distribution centered at mu_zi
        with the covariance matrix Sigma_zi as a scale parameter.
        """
        a, b = 0, np.inf  # Truncation limits
        scale = np.sqrt(np.diag(Sigma_zi))  # Scale is the standard deviation
        curr_r_i = self.r_samples[-1][i]
        lower, upper = (a - mu_zi) / scale, (b - mu_zi) / scale
        proposal_r_i = truncnorm.rvs(0, np.inf, loc=curr_r_i, scale=0.5, size=1)
        
        # Calculate the densities
        current_density = self.calculate_density(i, mu_zi, Sigma_zi)
        proposal_density = self.calculate_density(proposal_r_i, mu_zi, Sigma_zi, proposal=True)
        
        # Calculate acceptance probability
        acceptance_probability = min(1, proposal_density / current_density)
        
        # Accept or reject the proposal
        if np.random.rand() < acceptance_probability:
            return proposal_r_i
        else:
            return curr_r_i
    
    
    def mean_and_sigma_sample_density(self, mu_k, sigma_k, k):
        prob = 1
        group_mean = self.mean_samples[-1][k]
        group_sigma = self.sigma_samples[-1][k]

        for i, group in enumerate(self.z_samples[-1]):
            if group == k:
                # print(f"==>> self.calculate_density(i, mu_k, sigma_k): {self.calculate_density(i, mu_k, sigma_k)}")
                prob *= self.calculate_density(i, mu_k, sigma_k, verbose=True)
                print(prob, "prob")
        
        prob *= multivariate_normal.pdf(mu_k, mean=group_mean, cov=group_sigma)
        prob *= self.Sigma_prior(sigma_k)
        if isinstance(prob, (list, np.ndarray)) and len(prob) == 1:
            prob = prob[0]  # Extract the single element
        return prob
    

    
    def sample_mean_and_sigma(self, k):
        # Current parameters
        current_mu_k = self.mean_samples[-1][k]
        current_sigma_k = self.sigma_samples[-1][k]
        
        # Proposal generation
        proposed_mu_k = multivariate_normal.rvs(mean=current_mu_k, cov=current_sigma_k)
        proposed_sigma_k = invwishart.rvs(df=len(current_sigma_k) + 1, scale=current_sigma_k)
        
        # Compute the densities for the current and proposed parameters
        current_density = self.mean_and_sigma_sample_density(current_mu_k, current_sigma_k, k)
        #print(f"==>> current_density: {current_density}")
        proposed_density = self.mean_and_sigma_sample_density(proposed_mu_k, proposed_sigma_k, k)
        #print(f"==>> proposed_density: {proposed_density}")
        
        # Compute acceptance probability
        acceptance_ratio = proposed_density / current_density

        # print(acceptance_ratio, "acceptance ratio")
        
        # Accept or reject the proposal based on the computed probability
        if np.random.rand() < acceptance_ratio:
            return proposed_mu_k, proposed_sigma_k
        else:
            return current_mu_k, current_sigma_k
    


    
    def sample_pi(self):
        num_in_each_group = np.zeros(self.K)
        for group in self.z_samples[-1]:
            num_in_each_group[group] += 1
        
        # Sample from the Dirichlet distribution with parameters (n_1 + self.pi_prior, ..., n_K + self.pi_prior)

        new_pi = np.random.dirichlet(num_in_each_group + self.pi_prior)

        return new_pi
    


    def gibbs_sampler(self, max_iters=100):

        # Initialize the parameters
        # print(self.normed_embds, "normed embds")
        gmm = GMM_Init(self.normed_embds, n_components=self.K)

        self.mean_samples = [[x  for x in gmm.cluster_centres]]
        self.sigma_samples = [[x for x in gmm.cluster_covs]]
        self.r_samples = [[np.linalg.norm(x) for x in self.normed_embds]]
        self.z_samples = [lab for lab in gmm.labels]
        self.pi_samples = [np.array([1/self.K for _ in range(self.K)])]

        self.prior_df = 3
        self.prior_scale_mat = np.mean(gmm.cluster_covs, axis=0)


        for t in range(max_iters):
            # Update z
            new_z = [self.z_update(i) for i in range(self.N_t)]
            self.z_samples.append(new_z)

            # Update r
            new_r = [self.sample_r_i(i, self.mean_samples[-1][self.z_samples[-1][i]], self.sigma_samples[-1][self.z_samples[-1][i]]) for i in range(self.N_t)]
            self.r_samples.append(new_r)

            # Update mu and sigma
            new_mu = []
            new_sigma = []
            for k in range(self.K):
                mu_k, sigma_k = self.sample_mean_and_sigma(k)
                new_mu.append(mu_k)
                new_sigma.append(sigma_k)
            self.mean_samples.append(new_mu)
            self.sigma_samples.append(new_sigma)

            # Update pi
            new_pi = self.sample_pi()
            self.pi_samples.append(new_pi)
            # assert False


if __name__ == "__main__":
    N_t = 30
    K = 2
    gibbs = GibbsDataset(N_t, K)
    gibbs.gibbs_sampler(max_iters=100)
    print(gibbs.mean_samples, "mean samples")
    print(gibbs.sigma_samples, "sigma samples")
    print(gibbs.r_samples, "r samples")
    print(gibbs.z_samples, "z samples")
    print(gibbs.pi_samples, "pi samples")
