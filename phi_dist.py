import numpy as np

class phi():

    def __init__(self, α_prior, K):
        self.prior_conc = [α_prior/K for _ in range(K)] #conc= concentration parameter
        self.conc = None
        self.K = K # number of classes

        assert len(self.prior_params) == self.K, "Number of classes must match number of prior parameters"
    
    def vi(self, z_vi_list): # Need to pass in a list of z distributions for each point

        assert len(z_vi_list) == len(phi_vi_list), "Number of z distributions must match number of phi_vi distributions"

        exp_els_in_cluster = self.prior_conc
        for z in z_vi_list:
            for k in range(self.K):
                exp_els_in_cluster[k] += z.probs[k]
        
        self.conc = exp_els_in_cluster
        



            


