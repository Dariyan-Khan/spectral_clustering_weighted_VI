import numpy as np
from sklearn.mixture import GaussianMixture

class GMM_Init():

    def __init__(self, dataset, n_components=2):
        self.dataset = dataset
        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        self.fitted_gmm = gmm.fit(dataset)
    

    def mu_cov_estimate(self):
        cluster_centres = self.fitted_gmm.means_

        print(cluster_centres, "cluster centres")

        return np.cov(cluster_centres.T)
    
    def scaled_cluster_covs(self):
        # scaled so bottom right element is 1

        cluster_covs = self.fitted_gmm.covariances_
        return [cov/cov[-1,-1] for cov in cluster_covs]
    
    def gamma_estimate(self):
        cluster_covs = self.fitted_gmm.covariances_
        gamma_estimates = np.array([cov[:-1,-1] for cov in self.scaled_cluster_covs()])
        # print(gamma_estimates, "gamma estimates")
        gamma_cov_estimate = np.cov(gamma_estimates.T)
        # print(gamma_cov_estimate, "gamma cov estimate")

        return gamma_cov_estimate


    def sigma_estimate(self):
        cluster_covs = self.fitted_gmm.covariances_
        d = len(cluster_covs[0])

        gamma_cov_estimate = self.gamma_estimate()

        sigma_star_estimates = [cov[:-1,:-1]-gamma_cov_estimate for cov in self.scaled_cluster_covs()]
        sigma_star_estimates = np.array(sigma_star_estimates)
        dim = len(sigma_star_estimates[0])


        dof = d+3 # Set it to be d+3 for now
        sigma_star_mean = np.mean(sigma_star_estimates, axis = 0)
        scale = sigma_star_mean * (dof - (dim + 1))

        return scale, dof