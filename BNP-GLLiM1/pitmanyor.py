"""pitmanyor module

.. module:: pitmanyor
    :synopsis: A module dedicated to implementation of Pitman-Yor processes


.. topic:: pitmanyor.py summary

    A module dedicated to implementation of Pitman-Yor processes.
    Dirichlet processes are recovered by setting sigma to 0.

    :Code status: in development
    :Documentation status: to be completed
    :Author: 
    :Revision: $Id: pitmanyor.py$
    :Usage: >>> from pitmanyor import *
 
"""

import numpy as np
import scipy.special as special
from vbem_utils_numba import update_s_dp, update_s_py, update_gamma_sb, \
    compute_dp_means,py_samples,compute_py_means,corr_alpha_sigma, \
    compute_log_taus, compute_pi_tau_entropy, compute_z_tau_entropy,\
    update_s_dp_brent    
import math


class PitManYor:
    """
    Class implementing Pitman-Yor processes.
    Dirichlet processes are recovered by setting sigma to 0.
    
    :Attributes:
        `mean_sigma` (float) - for PY processes: approximation of marginal mean
            for sigma under variational distribution (?)
        `alpha` (np.array) - arameters s (2-dimensional) of alpha 
            variational posteriors
        `tau` (np.array) - Parameters gamma (2-dimensional) of tau 
            variational posteriors. 
        `samples` (dict) - samples used for approximating means by importance
            sampling (?) in PY processes.
    
    :Remark:
        `tau[k]` = 1 for k >= K-1 (indexing from 0) so their parameters are stored 
            for k < K-1 only
    
    """    
    
    def __init__(self, k_trunc):
        """
        for sigma = 0 we get the Dirichlet Process
        """
        self.mean_sigma = 0.0
        # Parameters s (2-dimensional) of alpha variational posteriors        
        self.alpha = np.array([0.0, 0.0], dtype=np.float64)
        # Parameters gamma (2-dimensional) of tau variational posteriors        
        # self.tau = np.zeros((k_trunc, 2), dtype=np.float64)  
        # TestT 2022/10/02.
        self.tau = np.zeros((k_trunc-1, 2), dtype=np.float64)           
        # only for pyp 
        self.samples = {}

        
    def __str__(self):
        return "PitManYor"
    
    def initialize(self, s1, s2, m_s):
        self.mean_sigma = m_s
        self.alpha[0] = s1
        self.alpha[1] = s2
        
    # ## Update the hyperparameters s1 and s2
    # def update_alpha(self, model, log_one_minus_tau, mean_alpha, mean_log_aps, 
    #                          k_trunc):
    #     """
    #     VE-alpha step 
    #     DP / PYP with gamma prior on alpha    
    #     Update hyperparameters s1 and s2
    #     """
        
    #     s1 = self.alpha[0]
    #     s2 = self.alpha[1]
       
    #     mean_sigma = self.mean_sigma
        
    #     if model == 'dp' or model == 'dp-mrf':
    #         hat_s1, hat_s2 = update_s_dp(log_one_minus_tau, s1, s2, k_trunc)
    #     elif model == 'pyp' or model == 'pyp-mrf':
    #         hat_s1, hat_s2 = update_s_py(mean_alpha, mean_sigma, 
    #                                       mean_log_aps, s1, s2)
    #     else:
    #         print("Warning: Please check the model name.")
    #         raise SystemExit

    #     self.alpha[0] = hat_s1
    #     self.alpha[1] = hat_s2
    
    ## TestT 2022/10/03
    def update_alpha(self, model, log_one_minus_tau, mean_alpha, mean_log_aps, 
                             k_trunc, alpha_estim="q_alpha"):
        """
        VE-alpha step 
        DP / PYP with gamma prior on alpha    
        Update hyperparameters s1 and s2
        """
    
        s1 = self.alpha[0]
        s2 = self.alpha[1]
       
        mean_sigma = self.mean_sigma
        
        if model == 'dp' or model == 'dp-mrf':
            if alpha_estim == "free_energy":
                hat_s1, hat_s2 = update_s_dp_brent(log_one_minus_tau, s1, s2, k_trunc)
            elif alpha_estim == "q_alpha":
                hat_s1, hat_s2 = update_s_dp(log_one_minus_tau, s1, s2, k_trunc)
            else:
                msg = "Bad value for alpha_estim: " + str(alpha_estim)
                raise ValueError(msg)
        elif model == 'pyp' or model == 'pyp-mrf':
            hat_s1, hat_s2 = update_s_py(mean_alpha, mean_sigma, 
                                          mean_log_aps, s1, s2)
        else:
            print("Warning: Please check the model name.")
            raise SystemExit

        self.alpha[0] = hat_s1
        self.alpha[1] = hat_s2        
        
    # Update the hyperparameters gamma_k
    def update_tau(self, mean_alpha, mean_sigma, nk, k_trunc):

        """
        Update gamma_k hyperparameters of tau posteriors

        Parameters
        ----------
        mean_alpha : np.float64
            Expectation of variational posterior on alpha.
        mean_sigma : np.float64
            Expectation of variational posterior on sigma.
        nk : np.float64
            Sums of variational posteriors of cluster variables z.
        k_trunc : int
            truncation values of nb clusters.

        Returns
        -------
        float
            Variational parameters gamma of tau posteriors ([k, 2] np.array).

        """
        
        self.tau = update_gamma_sb(nk, mean_alpha, mean_sigma, k_trunc)
        
        # ------------just for test 
        log_pi_tilde = np.zeros(k_trunc, dtype=np.float64)
        log_one_minus_tau_tilde = 0.0
        for k in range(k_trunc-1):
            tau_tilde = self.tau[k, 0] / (self.tau[k, 0] + self.tau[k, 1])
            log_pi_tilde[k] = np.log(tau_tilde) + log_one_minus_tau_tilde
            log_one_minus_tau_tilde = log_one_minus_tau_tilde + np.log(1.0-tau_tilde)
            
            #print('test_tau = ', np.exp(log_pi_tilde[k]))
            assert(np.exp(log_pi_tilde[k]) < 1.0 and np.exp(log_pi_tilde[k]) > 0.0)
        
        return self.tau

            
    def compute_log_Tau(self):
        taus = np.copy(self.tau)
        log_tau, log_one_minus_tau = compute_log_taus(taus)
        return log_tau, log_one_minus_tau
    
    
    def compute_means(self, model, log_tau, log_one_minus_tau, k_trunc, nb_samples):
        """
        Compute variational posteriors mean_alpha, mean_sigma
        for VB-E-(alpha,sigma) step in VBEM.
        """
        s1 = float(self.alpha[0])
        s2 = float(self.alpha[1])

        # s1 = self.alpha[0]
        # s2 = self.alpha[1]
        
        if model == 'dp' or model == 'dp-mrf':
            # update_s_dp already performed in update_alpha
            # s1_dp, s2_dp = update_s_dp(log_one_minus_tau, s1, s2, k_trunc)
            # TestT 2022/10/03            
            # s1_dp, s2_dp = update_s_dp(log_one_minus_tau, s1, s2, k_trunc)
            # mean_alpha, mean_sigma = compute_dp_means(log_tau, log_one_minus_tau, 
            #                                           s1_dp, s2_dp, k_trunc)
            mean_alpha, mean_sigma = compute_dp_means(log_tau, log_one_minus_tau, 
                                                      s1, s2, k_trunc)            
            mean_log_aps = 0.0
            
        elif model == 'pyp' or model == 'pyp-mrf':
            # update_s_dp already performed in update_alpha           
            # s1_pyp, s2_pyp = update_s_dp(log_one_minus_tau, s1, s2, k_trunc)
            alpha_samples, sigma_samples, weights = py_samples(
                                                    log_tau, log_one_minus_tau, 
                                                    s1, s2, 
                                                    k_trunc, nb_samples)   
            # store the samples in order to re-use them in function corr_alpha
            self.samples["alpha"] = alpha_samples
            self.samples["sigma"] = sigma_samples
            self.samples["weights"] = weights
            
            mean_alpha, mean_sigma, mean_log_aps = compute_py_means(
                                                    alpha_samples, 
                                                    sigma_samples, 
                                                    weights)
            # update sigma 
            self.mean_sigma = mean_sigma

        else:
            print("Warning: Please check the model name.")
            raise SystemExit

        return mean_alpha, mean_sigma, mean_log_aps
    

    def corr_alpha_sigma_func(self, mean_alpha, mean_sigma):	
        """
        In PYP NP priors, if the expectations of variational parameters 
        alpha and sigma are estimated by importance sampling,
        return sample correlation between alpha and sigma.
        """
        corr_as = corr_alpha_sigma(mean_alpha, mean_sigma, 
                                    self.samples["alpha"],
                                    self.samples["sigma"], 
                                    self.samples["weights"])
        
        return corr_as


    def elbo(self, beta_z, q_z, neighborhood, nb_contrib):
        """
        Compute PY contribution to entropy (elbo): alpha and tau
        
        :Parameters:
            `beta_z` (float) - beta estimate at current iteration
            `q_z` (([nb_data, k_trunc] np.array)) - variational approximation 
                of state distribution
            `neighborhood` (np.array) - matrix of neighbors 
              (-1 means no further neighbor)
            `nb_contrib` (np.array) - matrix of contributions of neighbors 
              to free energy              
        """
        hat_s1, hat_s2 = self.alpha # hyperparameters of gamma distribution
        # Remove H_alpha?
        # H_alpha = hat_s1 - math.log(hat_s2) + special.loggamma(hat_s1) + \
        #    (1-hat_s1) * special.digamma(hat_s1)
        H_tau = 0
        for k in range(self.tau.shape[0]):
            # Psi of gammak1 and gammak2
            psi1 = special.digamma(self.tau[k,0])
            psi2 = special.digamma(self.tau[k,1])
            psis = special.digamma(self.tau[k,0] + self.tau[k,1])
            H_tau += special.loggamma(self.tau[k,0]) + \
                special.loggamma(self.tau[k,1])  - \
                special.loggamma(self.tau[k,0]+self.tau[k,1]) - \
                (self.tau[k,0]-1) * (psi1 - psis) - \
                (self.tau[k,1]-1) * (psi2 - psis)

        H_alpha_tau = self.entropy_alpha_tau(self.alpha[0], self.alpha[1])
        
        # print('H_alpha_tau = ', H_alpha_tau)
        
        # H_cross_alpha = self.entropy_s_alpha(s)
        H_z_tau = self.entropy_z_tau(beta_z, q_z, neighborhood, nb_contrib)
        
        # print('H_z_tau = ', H_z_tau)
        
        return(H_tau + H_alpha_tau + H_z_tau)

    def entropy_alpha_tau(self, s1, s2):
        """
        Return (joint) entropy of log p(tau[k]|alpha)
                
        :Parameters:
            `s1, s2` - hyperparameters of variational alpha distribution, 
                do not depend on k
        """
        k_trunc = self.tau.shape[0]+1
        e = np.zeros((k_trunc-1, 1), dtype=np.float64) 
        t = special.digamma(s1) - math.log(s2)
        r = s1 / s2 - 1.
        for k in range(k_trunc-1):
            # Psi of gammak1 and gammak2
            e[k] = t + r * (special.digamma(self.tau[k,1])- \
             special.digamma(self.tau[k,0] + self.tau[k,1]))
                
        return(e.sum())
               
    def entropy_z_tau(self, beta_z, q_z, neighborhood, nb_contrib):
        """
        (joint) Entropy of log p(z|tau, beta)

        :Parameters:
            `beta_z` (float) - beta estimate at current iteration
            `q_z` (([nb_data, k_trunc] np.array)) - variational approximation 
                of state distribution
            `neighborhood` (np.array) - matrix of neighbors 
              (-1 means no further neighbor)
            `nb_contrib` (np.array) - matrix of contributions of neighbors 
              to free energy 
        """
        if beta_z == 0:
            return compute_pi_tau_entropy(beta_z, q_z, np.copy(self.tau))[3]
        else:
            return(compute_z_tau_entropy(beta_z, q_z, np.copy(self.tau), 
                                         neighborhood, nb_contrib))
        