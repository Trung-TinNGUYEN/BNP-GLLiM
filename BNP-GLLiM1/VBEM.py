#####################################################################################
#
#####################################################################################
import numpy as np
from gaussian import Gaussian 
from pitmanyor import PitManYor
from vbem_utils_numba import expectation_z, posterior_Gaussian_gate, gate_weight_post,\
    inversion, discrete_entropy, compute_q_z_tilde, normalize_exp_qz, neighbour_contributions
    #update_hyper_expert_init, compute_q_z_tilde, normalize_exp_qz
from potts_model import update_beta_z
from scipy.stats import multivariate_normal
from numpy.linalg import inv
import sys

class VBEM():
    
    """ 
    Implementation of VBEM algorithm for the simplest BNP-MRF-GLLiM.
    

    Attributes
    ----------
    sampling_model : str
        sampling distribution to use (e.g. "gaussian", "poisson").
    model : str
        dp, dp-mrf, pyp, pyp-mrf.
    k_trunc : int
        cluster truncation number.   
    process : object
        a PitManYor object (representing BNP).
    py : 
        a Distribution object (representing the sampling distribution).
    """
    
    # Initialization
    def __init__(self, sampling_model, model, k_trunc, nb_init):
        """
        

        Parameters
        ----------
        sampling_model : str
            sampling distribution to use (e.g. "gaussian", "poisson").
        model : str
            dp, dp-mrf, pyp, pyp-mrf.
        k_trunc : int
            cluster truncation number.
        nb_init : int
            number of trials in label initialization.

        Raises
        ------
        SystemExit
            DESCRIPTION.

        Returns
        -------
        None.
        """

        self.sampling_model = sampling_model
        self.model = model
        self.k_trunc = k_trunc
        self.process = PitManYor(k_trunc)
        
        # Initialize the emission distribution
        # Should rather use a factory method, see
        # https://realpython.com/factory-method-python/#introducing-factory-method        
        if sampling_model == "gaussian":
            self.py = Gaussian("[x,y]", nb_init) 
        else:
            raise SystemExit("Error: check the sampling model")
            
    def __str__(self):
        return "model " + self.model \
                + "\n sampling model: " + str(self.sampling_model) \
                + "\n truncation k: " + str(self.k_trunc)

    def elbo(self, data_X, data_Y, q_z, beta_z, neighborhood, nb_contrib,
             nb_data, dim_data_X, dim_data_Y, k_trunc):
        """
        Compute elbo (up to constants wrt hyperparameters Phi and variational
        parameters)
        
        :Parameters:
            `data` ([nb_data, dim_data] np.array) - Sample
            `q_z` (([nb_data, k_trunc] np.array)) - variational approximation 
                of state distribution
            `beta_z` (float) - Current value of beta
            `neighborhood` (np.array) - matrix of neighbors 
              (-1 means no further neighbor)
            `nb_contrib` (np.array) - matrix of contributions of neighbors 
              to free energy              
        """
        # compute PY contribution to elbo: alpha and tau
        e = self.process.elbo(beta_z, q_z, neighborhood, nb_contrib)
              
        # state entropy
        e += discrete_entropy(q_z) 
        
        # q_theta-related entropy terms
        # phi-related entropy terms.
        
        e += self.py.entropy_GateExperts(q_z, data_X, data_Y, nb_data,
                                        dim_data_X, dim_data_Y, k_trunc)        

        return(e)

    def run(self, data_X, data_Y, neighborhood, threshold, maxi, seed,
            nbinit, spatial_interaction):
        """
        Run VBEM algorithm to estimate (hyper)-parameters and compute variational
        posterior approximation for the simplest BNP-MRF-GLLiM models.

        Parameters
        ----------
        data_X : ([nb_data, dim_data_X] np.array)
            Input sample.          
        data_Y : ([nb_data, dim_data_Y] np.array)
            Output sample.   
        neighborhood : np.array
            matrix of neighbors (-1 means no further neighbor).
        threshold : float
            Threshold on relative difference in q_z used as stopping criterion.
        maxi : int
            number of runs for state initialization.
        seed : float
            seed used for random number generation.
        nbinit : int
            number of runs for state initialization.
        spatial_interaction : int
            = 1 : Spatial interaction modeled by Markov random fields: BNP-MRF-GLLiM.
            = 0 : Without spatial interaction: BNP-GLLiM.

        Returns
        -------
        c_est : 
            MAP clustering.
        q_z :
            Variational approximation q_z. 
        alpha_stored : 
            Expectations of BNP alpha parameter over iterations.
        sigma_stored : 
            Expectations of BNP sigma parameter over iterations.
        corr_as : 
            Estimated correlation of BNP alpha and sigma parameters 
            over iterations.
        beta_stored : 
            Sequence of estimated beta.
        delta_qz_stored : 
            Sequence of relative differences in q_z over iterations.            
        
        Remark
        -------
            Except for parameter alpha, the variational parameters and estimates
            resulting from M steps are identical and stored in self. Variational 
            parameters for tau are stored in temporary variables and also in
            self.process.tau.       
        
        """ 
        
        # nb_data: number of data points, dim_data_X: data dimension
        (nb_data, dim_data_X) = np.shape(data_X)
        dim_data_Y = np.shape(data_Y)[1]
        k_trunc = self.k_trunc
        model = self.model
            
        #------------------- Initialization -----------------------                     
                        
        # Number of samples for importance sampling
        nb_samples = 10000
        
        # Init sampling distribution parameters, q_z and Markov Random Field parmeters
        q_z, s1, s2, mean_sigma, beta_z_0,\
            A_hat, b_hat, Sigma_hat,\
            c_hat, Gamma_hat = self.py.init(data_X, data_Y, nb_data, dim_data_X,
                                            dim_data_Y, k_trunc, seed, nbinit)
        
        mean_alpha = s1/s2
        mean_log_aps = 0.0
        corr_as = 0.0
        #TestT 2022/10/03
        # gamma_sb = np.zeros((k_trunc, 2), dtype=np.float64)
        
        # Init A_hat, b_hat, Sigma_hat
        
	   # Init process
        self.process.initialize(s1, s2, mean_sigma) 
           
        nk = q_z.sum(axis=0)
        # print("Initial nk:", nk)
    
        # Store the inference results
        # Note that the length of delta_qz_stored is one unit
        # shorter than the others.
        A_hat_stored = [A_hat]
        b_hat_stored = [b_hat]
        Sigma_hat_stored = [Sigma_hat]
        
        c_hat_stored = [c_hat]
        Gamma_hat_stored = [Gamma_hat]
        
        gamma_sb_stored = []
        alpha_stored = [mean_alpha]
        sigma_stored = [mean_sigma]
        beta_stored = [beta_z_0]
        delta_qz_stored = [0.0]
        
        elbo_stored = []
        import math
        elbo_old = -nb_data * 1e10 * math.log(k_trunc)
        max_elbo = -nb_data * 1e10 * math.log(k_trunc)
        
        ### This is for the general BNP-MRF-GLLiM
        # m_hat_stored = [m_niw_0]
        # lambda_hat_stored = [lambda_niw_0]
        # psi_hat_stored = [psi_niw_0]
        # nu_hat_stored = [nu_niw_0]
            
        # psi_niw = np.zeros((k_trunc, dim_data, dim_data), dtype=np.float64)
        # m_niw = np.zeros((k_trunc, dim_data), dtype=np.float64)
        # lambda_niw = np.zeros(k_trunc, dtype=np.float64)
        # nu_niw = np.zeros(k_trunc, dtype=np.float64)
        
        # TestT
        # A_hat_storedT = A_hat.T
        # b_hat_storedT = b_hat.T
        # Sigma_hat_storedT = Sigma_hat.T
        
        ## Using array to store.
        # alpha_stored = np.array([mean_alpha], dtype=np.float64)
        # sigma_stored = np.array([mean_sigma], dtype=np.float64)
        # beta_stored = np.array([beta_z_0], dtype=np.float64)
        # delta_qz_stored = np.array([0.0], dtype=np.float64)

        #######
        ### Forward density.

        # A_hatS = np.zeros((k_trunc, dim_data_X, dim_data_Y), dtype=np.float64)
        # b_hatS = np.zeros((k_trunc, dim_data_X), dtype=np.float64)
        # Sigma_hatS = np.zeros((k_trunc, dim_data_X, dim_data_X), dtype=np.float64) 
        
        # c_hatS = np.zeros((k_trunc, dim_data_Y), dtype=np.float64)
        # Gamma_hatS = np.zeros((k_trunc, dim_data_Y, dim_data_Y), dtype=np.float64) 
        
        ## Instead of using a 3D array to store A_hat, we can store it as a 
        ## list of 2D arrays to simplify the calculation of 2D arrays.
        
        # self.A_hatS = [np.empty((dim_data_X, dim_data_Y)) for k in range(k_trunc)]
        # self.b_hatS = [np.empty((dim_data_X, 1)) for k in range(k_trunc)]
        # self.Sigma_hatS = [np.empty((dim_data_X, dim_data_X)) for k in range(k_trunc)]
        
        # self.c_hatS = [np.empty((dim_data_Y, 1)) for k in range(k_trunc)]
        # self.Gamma_hatS = [np.empty((dim_data_Y, dim_data_Y)) for k in range(k_trunc)]

        
        # -------------------- Iteration ----------------------
        
        # if (model == 'dp-mrf' or model == 'pyp-mrf'):
        #         print("Smoothing activated.")
 
        # Start the VBEM algorithm
        for iters in range(maxi):  # Stopping criterion

            q_z_old = np.copy(q_z)
            beta_z_0_old = beta_z_0

            ###################################################################                
            #---------------------- VB-E-steps ------------
            ###################################################################
            
            ##############################################            
            #----- VB-E-tau step:
            ##############################################                # update tau's posterior
            # gamma_sb are the variational parameters gamma of tau posteriors
            # Note that the other variational parameters are identical to M-step
            # estimates and thus stored in self. 
            # gamma_sb is also in self.process.tau            
            gamma_sb = self.process.update_tau(mean_alpha, mean_sigma, nk, k_trunc)
            # E[log(tau)] and E[log(1-tau)]
            log_tau, log_one_minus_tau = self.process.compute_log_Tau()

            ##############################################            
            #----- VB-E-(alpha,sigma) step:
            ##############################################   

            # TestT
            self.process.update_alpha(self.model, log_one_minus_tau, mean_alpha,
                                           mean_log_aps, k_trunc)
             
            mean_alpha, mean_sigma, mean_log_aps = self.process.compute_means(
                                        self.model, log_tau, 
                                        log_one_minus_tau, k_trunc, nb_samples)
            
            ##############################################            
            #----- VB-E-theta* step + VB-M-rho step:
            ##############################################
            # Calculate variational posterior q_theta* by updating 
            # hyperparameters rho = (m,lambda,psi,nu) using for VB-E-theta* step.
            # Similar to standard Bayesian finite mixtures.
            # # update distribution p(y|theta)       
            # self.py.up_date_posterior(data_X, q_z, nk, nb_data, dim_data_X, k_trunc)
            # It should be self.py.update_posterior instead of self.py.up_date_posterior
            #self.py.update_posterior(data_X, q_z, nk, nb_data, dim_data_X, k_trunc)       
            #self.py.update_posterior(data_X, q_z, nk, nb_data, dim_data_X, k_trunc)
            
            # m_niw, psi_niw, lambda_niw, nu_niw = self.py.update_posterior(data_X, q_z, 
            #                                             nk, nb_data, dim_data_X, k_trunc)
           
            # m_niw, psi_niw, lambda_niw, nu_niw = py.update_posterior(data_X, q_z,
            #                                                       nk, nb_data, 
            #                                                       dim_data_X, k_trunc)

            ##############################################
            #----- VB-E-Z step: update clusters's labels. 
            ##############################################
            # TrungTin Nguyen. [Eq. (38) Lu et al., 2020)].
            # Compute -0.5 * expectation of log covariance determinant 
            # for Gaussian distributions under variational theta* distribution.            
            #exp_log = self.py.expectation_likelihood(data_X, nb_data, dim_data_X, k_trunc)
            exp_log = self.py.expectation_likelihood(data_X, data_Y, nb_data,
                                                     dim_data_X, dim_data_Y, k_trunc)
            
            q_z = expectation_z(q_z, beta_z_0, exp_log, log_tau, log_one_minus_tau,
                                nb_data, k_trunc, neighborhood)
            
            
            #----------------------Check that everything goes right ----------
            
            # Compute local_energy for multiple reuse
            nb_contrib = neighbour_contributions(q_z, nb_data, k_trunc, 
                                     beta_z_0, neighborhood)
            
            # cluster counts
            nk = q_z.sum(axis=0)
            
            
            # assert(np.min(np.sum(q_z, axis=1)) >= 0.99)
            # _q_z_tilde = compute_q_z_tilde(beta_z_0, q_z, gamma_sb, 
            #                     nb_data, k_trunc, neighborhood)
            # # Normalize q_z_tilde:
            # _q_z_tilde = normalize_exp_qz(_q_z_tilde, nb_data, k_trunc)
            # assert(np.min(np.sum(_q_z_tilde , axis=1)) >= 0.99)
            
            ## TestT
            assert(np.min(np.sum(q_z, axis=1)) >= 0.99)
            _q_z_tilde = compute_q_z_tilde(beta_z_0, q_z, gamma_sb, 
                                           nb_data, k_trunc, nb_contrib)
            # Mormalize q_z_tilde:
            _q_z_tilde = normalize_exp_qz(_q_z_tilde, nb_data, k_trunc)
            assert(np.min(np.sum(_q_z_tilde , axis=1)) >= 0.99)   
                  
            #-------------------------------------------------------------
           
            ###################################################################                            
            #------------------- VB-M-steps ------------
            ###################################################################                

            ##############################################            
            #----- VB-M-beta step:-----
            ##############################################    
         #    # Update potts model.
         #    if spatial_interaction != 0.0:
         #        beta_z_0 = update_beta_z(q_z, gamma_sb, model, self.sampling_model,
    				 # nb_data, k_trunc, beta_z_0, neighborhood, iters)
            
            ##############################################    
            #----- VB-M-(s1,s2,a) step:
            ##############################################    
            # Update process parameters (s1,s2) of alpha variational posterior                
            self.process.update_alpha(model, log_one_minus_tau, mean_alpha,
                                           mean_log_aps, k_trunc)
            
            # mean_alpha, mean_sigma, mean_log_aps = \
            #     self.process.compute_means(self.model, log_tau, 
            #                                 log_one_minus_tau, k_trunc, 
            #                                 nb_samples)            

            ##############################################
            #----- VB-M-(A,b,Sigma) step:
            ##############################################    
            # Update GLLiM experts' parameters (M-mapping-step). 
            #
            #self.py.update_posterior(data_X, q_z, nk, nb_data, dim_data_X, k_trunc)
            # self.py.update_hyper_expert(data_X, q_z, nk, nb_data, dim_data_X, k_trunc,
            #                         data_Y, dim_data_Y)
            
            ## First way: returning A_hat,b_hat, Sigma_hat
            A_hat, b_hat, Sigma_hat = self.py.update_hyper_expert(data_X, q_z, nk, nb_data,
                                                                dim_data_X, k_trunc,                                                           
                                                                data_Y, dim_data_Y)

            ## Second way: directly call A_hat,b_hat, Sigma_hat using self function.          
            # A_hat = self.py.hyper_expert["A_hat"]
            # b_hat = self.py.hyper_expert["b_hat"]
            # Sigma_hat = self.py.hyper_expert["Sigma_hat"]
            
            
            
            
            
            ##############################################
            #----- VB-M-(c,Gamma) step:
            ##############################################    
            # Update GLLiM gating' parameters (M-mapping-step). 
            #
            
            ## First way: returning c_hat, Gamma_hat
            c_hat, Gamma_hat = self.py.update_hyper_gate(data_X, q_z, nk, nb_data, 
                                                         dim_data_X, k_trunc)
            # print('c_hat_test = ', c_hat)
            # print('Gamma_hat_test = ', Gamma_hat)

            ## Second way: directly call A_hat,b_hat, Sigma_hat using self function.          
            # A_hat = self.py.hyper_expert["A_hat"]
            # b_hat = self.py.hyper_expert["b_hat"]
            # Sigma_hat = self.py.hyper_expert["Sigma_hat"]
            
            
            
            
            ###################################################################
           
            # --------------------------------------------------------------
            # Compute elbo
            elbo = self.elbo(data_X, data_Y, q_z, beta_z_0, neighborhood, nb_contrib,
                             nb_data, dim_data_X, dim_data_Y, k_trunc)
            # --------------------------------------------------------------
            
            # Difference in free energy
            delta_elbo = elbo - elbo_old
            elbo_old = elbo
                      
            if elbo > max_elbo:
                max_elbo = elbo
                # best_model = self.copy(q_z, mean_alpha, mean_sigma, beta_z_0)
                # best_iter = iters
                
            # --------------------------------------------------------------           
            # # cluster counts
            # nk = q_z.sum(axis=0)
            
            delta_qz = np.linalg.norm(q_z-q_z_old) / np.linalg.norm(q_z_old)
            delta_beta = beta_z_0 - beta_z_0_old
            
            
            # Display each iteration
            print(iters+1, "nk:", nk)
            print(iters+1, "delta_qz=" + str(delta_qz))
            print(iters+1, "elbo=" + str(elbo))
            """
            print(iters+1, "delta_qz=" + str(delta_qz))
            print(iters+1, "alpha=" + str(mean_alpha))
            print(iters+1, "sigma=" + str(mean_sigma))
            print(iters+1, "beta=" + str(beta_z_0))
            print(iters+1, "delta_beta=" + str(delta_beta))
            """
            print("---------------------------------")
            
            ## Saving estimation using list instead of array to keep the matrix
            # structure for density estimation.
            A_hat_stored.append(A_hat)
            b_hat_stored.append(b_hat)
            Sigma_hat_stored.append(Sigma_hat)
            
            c_hat_stored.append(c_hat)
            Gamma_hat_stored.append(Gamma_hat)
            
            gamma_sb_stored.append(gamma_sb)
            alpha_stored.append(mean_alpha)
            sigma_stored.append(mean_sigma)
            beta_stored.append(beta_z_0)
            delta_qz_stored.append(delta_qz)
            
            elbo_stored = np.append(elbo_stored, elbo)   
            
            ### This is for the general BNP-MRF-GLLiM
            # m_hat_stored.append(m_niw)
            # lambda_hat_stored.append(lambda_niw)
            # psi_hat_stored.append(psi_niw)
            # nu_hat_stored.append(nu_niw)
            
            ## Using append to store data as 
            # A_hat_stored = np.append(A_hat_stored, A_hat.T, axis = 0)
            # b_hat_stored = np.append(b_hat_stored, b_hat.T, axis = 0)
            # Sigma_hat_stored = np.append(Sigma_hat_stored, Sigma_hat.T, axis = 0)

            # alpha_stored = np.append(alpha_stored, mean_alpha)
            # sigma_stored = np.append(sigma_stored, mean_sigma)
            # beta_stored = np.append(beta_stored, beta_z_0)
            # delta_qz_stored = np.append(delta_qz_stored, delta_qz)
            
            ###
            ### Remove empty clusters.
            ec = [True]*self.k_trunc
            nec = 0
            normalizing_weighted_input, normalizing_weighted_output =\
                self.py.normalizing_weighted_input_output(data_X, q_z, nk, nb_data,
                                                          dim_data_X, self.k_trunc,
                                                          data_Y, dim_data_Y)    
            for k in range(self.k_trunc):
                # if (np.sum(normalizing_weighted_input[k]) == 0)or(np.sum(q_z, axis=0)[k] < 1):
                #     print('Empty cluster = ', k)
                # print('normalizing_weighted_input[k] = ', normalizing_weighted_input[k])
                # print('normalizing_weighted_output[k] = ', normalizing_weighted_output[k])

                # if (np.sum(normalizing_weighted_input[k]) == 0):
                if np.sum(q_z, axis=0)[k] == sys.float_info.epsilon:    
                    print('Empty cluster = ', k)
                    nec +=1
                    ec[k] = False
                    print('Cluster', str(k),'has been removed')
            if nec > 0:
                
                self.k_trunc -= nec
                k_trunc = self.k_trunc
                q_z = q_z[:,ec]       
                self.py.hyper_expert["A_hat"] = A_hat[ec,:,:]
                
                
                # print('q_z_remove = ', q_z)
                print('q_z_remove_shape = ', q_z.shape)   
                print('A_hat_remove_shape = ', self.py.hyper_expert["A_hat"])
            
            # Stopping criterion
            # deltas = np.array([delta_beta, delta_qz, delta_elbo])
            # deltas = np.array([delta_qz, delta_elbo])  
            # TestingT
            deltas = np.array([delta_qz, delta_elbo])    
            min_deltas = min(abs(deltas))            
            # to do: Should this stopping criterion be changed?
            if self.sampling_model == "gaussian":
                if self.py.convergence(model, delta_beta, min_deltas,
                                        threshold, spatial_interaction):
                    break
            
            elif self.py.convergence(model, delta_beta, delta_qz,
                                      threshold, spatial_interaction):
                    break
                       
        # Determine the optimal clustering using MAP (maximum a posteriori)
        c_est = np.argmax(q_z, axis=1)

        # Compute the correlation between alpha and sigma for PYP
        if model == 'pyp' or model == 'pyp-mrf':
            corr_as = self.process.corr_alpha_sigma_func(mean_alpha, mean_sigma)
        ##  
        #### Bayesian inversion of the parameters.
        ##
        # A_hatS = np.zeros((k_trunc, dim_data_X, dim_data_Y), dtype=np.float64)
        # b_hatS = np.zeros((k_trunc, dim_data_X), dtype=np.float64)
        # Sigma_hatS = np.zeros((k_trunc, dim_data_X, dim_data_X), dtype=np.float64) 
        
        # c_hatS = np.zeros((k_trunc, dim_data_Y), dtype=np.float64)
        # Gamma_hatS = np.zeros((k_trunc, dim_data_Y, dim_data_Y), dtype=np.float64) 
        
        # for k in range(k_trunc):
        #     Sigma_hatS[k] = inv(inv(Gamma_hat[k]) + (A_hat[k].T).dot(inv(Sigma_hat[k])).\
        #                         dot(A_hat[k]))
        #     A_hatS[k] = Sigma_hatS[k].dot(A_hat[k].T).dot(inv(Sigma_hat[k]))
        #     b_hatS[k] = Sigma_hatS[k].dot(inv(Gamma_hat[k]).dot(c_hat[k]) -\
        #                                   (A_hat[k].T).dot(inv(Sigma_hat[k])).dot(b_hat[k]))
        #     c_hatS[k] = A_hat[k].dot(c_hat[k]) + b_hat[k]
        #     Gamma_hatS[k] = Sigma_hat[k] + A_hat[k].dot(Gamma_hat[k]).dot(A_hat[k].T)
            
        ## Calling from vbem_utils_numba.
        A_hatS, b_hatS, Sigma_hatS,\
            c_hatS, Gamma_hatS = inversion(A_hat, b_hat, Sigma_hat,
                                           c_hat, Gamma_hat)   
        
            
        
        return (c_est, q_z, alpha_stored, sigma_stored, gamma_sb_stored,
                corr_as, beta_stored, delta_qz_stored, elbo_stored, iters,
                A_hat_stored, b_hat_stored, Sigma_hat_stored,
                c_hat_stored, Gamma_hat_stored, Sigma_hatS, A_hatS, b_hatS,
                c_hatS, Gamma_hatS)
    
	
    def inverse_expectation(self, X_hat, k_trunc, gamma_sb, A_hat, b_hat,
                                    c_hat, Gamma_hat):
        
        (nb_data_hat, dim_data_X) = np.shape(X_hat)
        
        dim_data_Y = np.shape(A_hat)[1]
        
        Eyx_hat = np.zeros((nb_data_hat, dim_data_Y), dtype=np.float64)
        
        ## Calculate the gating posteriors E_{q_tau}[pi_k(tau)].
        ## Require gamma_sb = np.zeros((k_trunc, 2), dtype=np.float64)
        gate_weight_post_hat = gate_weight_post(k_trunc, gamma_sb)
        
        
        ## Compute prediction task: 
        # A_hat = np.zeros((k_trunc, dim_data_Y, dim_data_X), dtype=np.float64)
        # b_hat = np.zeros((k_trunc, dim_data_Y), dtype=np.float64)
        # Sigma_hat = np.zeros((k_trunc, dim_data_Y, dim_data_Y), dtype=np.float64)     
        # gk: ([nb_data_hat, k_trunc], np.array)
        # psi_niw = np.zeros((k_trunc, dim_data, dim_data), dtype=np.float64)
        # m_niw = np.zeros((k_trunc, dim_data), dtype=np.float64)
        # lambda_niw = np.zeros(k_trunc, dtype=np.float64)
        # nu_niw = np.zeros(k_trunc, dtype=np.float64)
        gk = posterior_Gaussian_gate(X_hat, gate_weight_post_hat, c_hat, Gamma_hat)[0]
        
        
        for n in range(nb_data_hat):
            for k in range(k_trunc):
                Eyx_hat[n,:,None] += gk[n,k] *  (A_hat[k,:,:]@(X_hat[n,:].T) + b_hat[k,:].T) 
        return Eyx_hat, gk, gate_weight_post_hat
    
    def inverse_density(self, X_hat, Y_hat, k_trunc, gamma_sb, A_hat, b_hat,
                        Sigma_hat, c_hat, Gamma_hat):

        (nb_data_hat, dim_data_X) = np.shape(X_hat)
        
        # dim_data_Y = np.shape(A_hat)[1]
        
        ## Calculate the gating posteriors E_{q_tau}[pi_k(tau)]
        ## and the estimated gating networks.
        gate_weight_post_hat = gate_weight_post(k_trunc, gamma_sb)
        gk = posterior_Gaussian_gate(X_hat, gate_weight_post_hat, c_hat, Gamma_hat)[0]
        
        pyx_hat = np.zeros((nb_data_hat, 1), dtype=np.float64)
        
        for n in range(nb_data_hat):
            for k in range(k_trunc):
                pyx_hat[n,:,None] += gk[n,k] *\
                    multivariate_normal.pdf(Y_hat[n,:,None],
                                               A_hat[k,:,:]@(X_hat[n,:,None].T) + b_hat[k,:].T,
                                               Sigma_hat[k,:,:]) 
        
        return pyx_hat

    ######
    ### Merge-Truncate-Merge Algorithm for BNP-GLLiM. Algorithm 1 from Guha et al. (2021).
    ######
    def MTM(self, pi, N, c, r, quant_MTM, seed, A_hat, b_hat, Sigma_hat, c_hat, Gamma_hat):
        
        import random
        from numpy import linalg as LA
        from numpy import array
        
        # ## uniquement pour le test
        # # N = 1000, omegaN = 0.52894, c < c0, c0 = 1/2*1/9, pi0 = (1/3,1/3,1/3)
        # # c = [0.45, 0.5, 0.55, 1.0]
        # c = 0.05
        # r = 2
        
        # pi = gate_weight_post_hat
        # A_hat = A_hat_stored[-1]
        # b_hat = b_hat_stored[-1]
        # Sigma_hat = Sigma_hat_stored[-1]
        # c_hat = c_hat_stored[-1]
        # Gamma_hat = Gamma_hat_stored[-1]
        
        # numpy.linalg.norm
        # This function is able to return one of eight different matrix norms,
        # or one of an infinite number of vector norms (described below), 
        # depending on the value of the ord parameter.
        # Examples
        # from numpy import linalg as LA
        # a = np.arange(9) - 4
        # b = a.reshape((3, 3))
        # print('LA.norm(a) = ',LA.norm(a))
        # print('LA.norm(b) = ',LA.norm(b))     
        # print('LA.norm(b, fro) = ', LA.norm(b, 'fro'))
        # print('LA.norm(b, 2) = ', LA.norm(b, 2))        
        
        
        ## Calculer posterior contraction rate
        # omegaN = (np.log(np.log(N))/np.log(N))**(1/2)
        
        ## Calculer muk et Vk. 
        K = np.shape(pi)[0]
        dim_data_X = np.shape(c_hat)[1]
        
        dim_data_Y = np.shape(b_hat)[1]
        
        muk = np.zeros((K, dim_data_X+dim_data_Y), dtype=np.float64)
        Vk = np.zeros((K, dim_data_X+dim_data_Y, dim_data_X+dim_data_Y), dtype=np.float64) 
        
        # Uniquement pour le test
        ## vb_N variational error.
        omegaN = (np.log(np.log(N))/np.log(N))**(1/2)
        # vb_N = 5
        ## print("Q2 quantile of arr : ", np.quantile(pi, .50))
        # vb_N = abs((np.quantile(pi, .50)/c)**(1/r)+omegaN)
        
        vb_N = abs((np.quantile(pi, quant_MTM)/c)**(1/r)-omegaN)
        # vb_N = omegaN
        # vb_N = 0
        omegaN += vb_N
        
        for k in range(K):
            # muk[k,0:dim_data_X] = c_hat[k,:]
            # muk[k,dim_data_X:(dim_data_Y+1)] = A_hat[k,:]@c_hat[k,:] + b_hat[k,:]

            muk[k, 0:dim_data_X] = c_hat[k]
            muk[k, dim_data_X:(dim_data_Y+1)] = A_hat[k]@c_hat[k] + b_hat[k]
            
            Vk[k, 0:dim_data_X, 0:dim_data_X] = Gamma_hat[k]
            Vk[k, 0:dim_data_X, dim_data_X:(dim_data_Y+1)] = \
                Gamma_hat[k]@A_hat[k].T
            Vk[k, dim_data_X:(dim_data_Y+1), 0:dim_data_X] = \
                A_hat[k]@Gamma_hat[k]                   
            Vk[k, dim_data_X:(dim_data_Y+1), dim_data_X:(dim_data_Y+1)] = \
                Sigma_hat[k] + A_hat[k]@Gamma_hat[k]@A_hat[k].T          
            
        ## Stage 1: Merge procedure.
        ## Reorder atoms thetak
        # Ligne 1:

        random.seed(seed)
        tau = random.sample(range(pi.shape[0]),pi.shape[0])
        print('tau = ', tau)
        
        tauE = np.copy(tau) 
        piTau = np.copy(pi)
        
                
        ## Ligne 2:
        tauR = []    
        for tauj in range(pi.shape[0]):
            for taui in range(tauj):
                
                ## Calculer distance
                #
                dThetaij = (LA.norm(muk[taui] - muk[tauj])**2 + \
                            LA.norm(Vk[taui] - Vk[tauj])**2)**(1/2)
                
                if (dThetaij < omegaN)&(tauE[taui] != -1):
                    piTau[taui] += piTau[tauj]
                    tauR.append(tauj) 
                    tauE[tauj] = -1 # -1 = nous le retirons de E.
                    piTau[tauj] = 0
        
        print('Merge tauE = ', tauE)
        print('Remove indices tauR = ', tauR)
        print('Remove indices piTau = ', piTau)
                   
        ## Ligne 3: Collect G'.
            
        tauE = np.delete(tauE, tauR)
        # piTau = np.delete(piTau, tauR)
        print('Remove tauE = ', tauE)
        # print('Remove piTau = ', piTau)
        
        ## Calculate qk, q1 > q2 > ...
        tauqk = np.argsort(-piTau,axis=0)[0:tauE.shape[0]]
        piTauqk = -np.sort(-piTau,axis=0)[0:tauE.shape[0]]
        print('Sorted indices tauqk = ', tauqk)
        print('Sorted piTauqk = ', piTauqk)
        
        ## Calculer mukPhi VkPhi
        mukPhi = muk[tauqk]
        VkPhi = Vk[tauqk]
           
        ## 
        ## Stage 2: Truncate-Merge procedure.
        ##
        # Ligne 4:
        condA = piTauqk > (c*omegaN)**r
        indexA = np.where(condA)[0]
        condN = piTauqk <= (c*omegaN)**r
        indexN = np.where(condN)[0]
        print('indexA = ', indexA)
        print('indexN = ', indexN)
        print('omegaN = ', omegaN)
        print('vb_N = ', vb_N)
        print('(c*omegaN)**r = ', (c*omegaN)**r)
        
        ## Ligne 5:
        lenIndexA = indexA.shape[0]    
        for i in range(lenIndexA):
            for j in range(i):
                ## Calculer distance
                #
                qPhij = piTauqk[i]*(LA.norm(mukPhi[i] - mukPhi[j])**r + \
                            LA.norm(VkPhi[i] - VkPhi[i])**r)
                if qPhij <= (c*omegaN)**r:
                    ## Remove i from A and add it to N
                    indexA = np.delete(indexA, i)
                    indexN = np.append(indexN, i)
                    
        ## Ligne 6:
        lenIndexN = indexN.shape[0]
        lenIndexAT = indexA.shape[0]
        for i in range(lenIndexN):
            dPhi = []
            for j in range(lenIndexAT):
                # Calculer la distance entre Phi_j et Phi_i.
                dPhiiPhij = (LA.norm(mukPhi[j] - mukPhi[indexN[i]])**2 + \
                            LA.norm(VkPhi[j] - VkPhi[indexN[i]])**2)**(1/2)
                dPhi.append(dPhiiPhij)
            if len(dPhi):    
                # Trouver un atome Phi_j, j \in \cA, qui est le plus proche de Phi_i.
                min_dPhiiPhij = np.argmin(dPhi)
                # Mise à jour q_j = q_j + q_i
                piTauqk[indexA[min_dPhiiPhij]] += piTauqk[indexN[i]]
            
        ## Ligne 7: Retourner la valeur \widetidle{G}.
        pi_hatT = piTauqk[indexA]
        print('pi_hatT = ', pi_hatT)
        print('indexA = ', indexA)
        print('A_MTM = ', A_hat[indexA,:,:])

        
        KT = pi_hatT.shape[0]
        
            
        return KT, indexA

