import numpy as np

class Phi():

    def __init__(self, K, α_prior=None):
        
        if α_prior is None:
            α_prior = 1
        
        self.prior_conc = [α_prior/K for _ in range(K)] #conc= concentration parameter
        self.conc = [α_prior/K for _ in range(K)] #at least just for initialisation
        self.K = K # number of classes

        assert len(self.prior_conc) == self.K, "Number of classes must match number of prior parameters"
    
    def vi(self, z_vi_list): # Need to pass in a list of z distributions for each point

        exp_els_in_cluster = self.prior_conc
        for z_i in z_vi_list:
            for k in range(self.K):
                exp_els_in_cluster[k] += z_i.probs[k]
        
        self.conc = exp_els_in_cluster
        



            


