import numpy as np
from sklearn.mixture import GaussianMixture

class GMM_Init():

    def __init__(self, dataset, n_components=2):
        self.dataset = dataset
        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        self.fitted_gmm = gmm.fit(dataset)

        self.cluster_centres = self.fitted_gmm.means_
        self.cluster_covs = self.fitted_gmm.covariances_
        
        self.gamma_inits = []
        self.sigma_star_inits = []

        for c_cov in self.cluster_covs:
            self.gamma_inits.append(c_cov[:-1,-1])

            # print("outer product:", np.outer(c_cov[:-1,-1], c_cov[:-1,-1]))

            self.sigma_star_inits.append(c_cov[:-1,:-1] - np.outer(c_cov[:-1,-1], c_cov[:-1,-1]))
        


    def mu_prior_cov_estimate(self):
        cluster_centres = self.fitted_gmm.means_

        print(cluster_centres, "cluster centres")



        return np.cov(cluster_centres.T)
    
    def scaled_cluster_covs(self):
        # scaled so bottom right element is 1
        cluster_covs = self.fitted_gmm.covariances_
        return [cov/cov[-1,-1] for cov in cluster_covs]
    
    def gamma_prior_cov_estimate(self):
        cluster_covs = self.fitted_gmm.covariances_

        # print(f"""
        # Cluster cov 1:
              
        #       {cluster_covs[0]}
    
        # Cluster cov 2:

        #     {cluster_covs[1]}

        #       """)
        # print(cluster_covs, "cluster covs")
        # gamma_estimates = np.array([cov[:-1,-1] for cov in self.scaled_cluster_covs()])
        gamma_estimates = np.array([cov[:-1,-1] for cov in cluster_covs])
        # print(gamma_estimates, "gamma estimates")

        print(gamma_estimates, "gamma estimates")
        gamma_cov_estimate = np.cov(gamma_estimates.T)
        print(gamma_cov_estimate, "gamma cov estimate")

        return gamma_cov_estimate


    def sigma_prior_params_estimate(self):
        cluster_covs = self.fitted_gmm.covariances_
        d = len(cluster_covs[0])

        gamma_cov_estimate = self.gamma_prior_cov_estimate()

        # sigma_star_estimates = [cov[:-1,:-1]-gamma_cov_estimate for cov in self.scaled_cluster_covs()]

        # sigma_star_estimates = [cov[:-1,:-1]-gamma_cov_estimate for cov in cluster_covs]
        sigma_star_estimates = np.array(self.sigma_star_inits)
        dim = len(sigma_star_estimates[0])


        dof = d+3 # Set it to be d+3 for now
        sigma_star_mean = np.mean(sigma_star_estimates, axis = 0)
        scale = sigma_star_mean * (dof - (dim + 1))

        return scale, dof