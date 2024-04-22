import numpy as np

class phi():

    def __init__(self, prior, K):
        self.prior_params = prior
        self.params = None
        self.K = K # number of classes

        assert len(self.prior_params) == self.K, "Number of classes must match number of prior parameters"
    
    def weighted_vi(self, z_list, phi_vi_list): # Need to pass in a list of z distributions for each point

        assert len(z_list) == len(phi_vi_list), "Number of z distributions must match number of phi_vi distributions"

        exp_els_in_cluster = self.prior_params
        for z, phi_vi in zip(z_list, phi_vi_list):
            exp_els_in_cluster[z.cluster] += phi_vi.params[z.cluster]
        
        self.params = exp_els_in_cluster
        



            


