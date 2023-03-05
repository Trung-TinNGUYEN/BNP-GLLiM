#####################################################################################
#
#####################################################################################
import numpy as np
from gaussian import Gaussian 
from pitmanyor import PitManYor
from vbem_utils_numba import expectation_z, posterior_Student_gate
    #update_hyper_expert_init, compute_q_z_tilde, normalize_exp_qz
from potts_model import update_beta_z


class VBEM():
    
    """ 
    Implementation of VBEM algorithm for BNP-MRF-GLLiM.
    

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
            #self.py = Gaussian("y", nb_init)
            self.py = Gaussian("x", nb_init) 
        else:
            raise SystemExit("Error: check the sampling model")
        
                  
    def __str__(self):
        return "model " + self.model \
                + "\n sampling model: " + str(self.sampling_model) \
                + "\n truncation k: " + str(self.k_trunc)


    def run(self, data_X, data_Y, neighborhood, threshold, maxi,
            seed, nbinit, spatial_interaction):
        """
        Run VBEM algorithm to estimate (hyper)-parameters and compute variational
        posterior approximation for BNP-MRF-GLLiM models.

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
                m_niw_0, lambda_niw_0, psi_niw_0, nu_niw_0 = self.py.init(data_X, data_Y, nb_data,
                                                         dim_data_X, dim_data_Y, 
                                                         k_trunc, seed, nbinit)
        
        mean_alpha = s1/s2
        mean_log_aps = 0.0
        corr_as = 0.0
        gamma_sb = np.zeros((k_trunc, 2), dtype=np.float64)
        
        # Init A_hat, b_hat, Sigma_hat
        
	   # Init process
        self.process.initialize(s1, s2, mean_sigma) 
           
        nk = q_z.sum(axis=0)
        print("Initial nk:", nk)
    
        # Store the inference results
        # Note that the length of delta_qz_stored is one unit
        # shorter than the others.
        A_hat_stored = [A_hat]
        b_hat_stored = [b_hat]
        Sigma_hat_stored = [Sigma_hat]
        
        gamma_sb_stored = [gamma_sb]
        alpha_stored = [mean_alpha]
        sigma_stored = [mean_sigma]
        beta_stored = [beta_z_0]
        delta_qz_stored = [0.0]
        
        m_hat_stored = [m_niw_0]
        lambda_hat_stored = [lambda_niw_0]
        psi_hat_stored = [psi_niw_0]
        nu_hat_stored = [nu_niw_0]
            
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
            
            m_niw, psi_niw, lambda_niw, nu_niw = self.py.update_posterior(data_X, q_z, 
                                                        nk, nb_data, dim_data_X, k_trunc)
           
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
            
            q_z = expectation_z(q_z, beta_z_0, exp_log, log_tau, 
                                log_one_minus_tau, nb_data, k_trunc, neighborhood)

    
            # cluster counts
            nk = q_z.sum(axis=0)            
            
            #----------------------Check that everything goes right ----------
            # assert(np.min(np.sum(q_z, axis=1)) >= 0.99)
            # _q_z_tilde = compute_q_z_tilde(beta_z_0, q_z, gamma_sb, 
            #                     nb_data, k_trunc, neighborhood)
            # # Normalize q_z_tilde:
            # _q_z_tilde = normalize_exp_qz(_q_z_tilde, nb_data, k_trunc)
            # assert(np.min(np.sum(_q_z_tilde , axis=1)) >= 0.99)
            #-------------------------------------------------------------
           
            ###################################################################                            
            #------------------- VB-M-steps ------------
            ###################################################################                

            ##############################################            
            #----- VB-M-beta step:-----
            ##############################################    
            # Update potts model.
            if spatial_interaction != 0.0:
                beta_z_0 = update_beta_z(q_z, gamma_sb, model, self.sampling_model,
    				 nb_data, k_trunc, beta_z_0, neighborhood, iters)
            
            ##############################################    
            #----- VB-M-(s1,s2,a) step:
            ##############################################    
            # Update process parameters (s1,s2) of alpha variational posterior                
            self.process.update_alpha(model, log_one_minus_tau, mean_alpha,
                                           mean_log_aps, k_trunc)

            mean_alpha, mean_sigma, mean_log_aps = self.process.compute_means(
                                        self.model, log_tau, 
                                        log_one_minus_tau, k_trunc, nb_samples)
            
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
            print('A_hat_test_shape_vb-m-Sigma = ', A_hat.shape)

            ## Second way: directly call A_hat,b_hat, Sigma_hat using self function.          
            # A_hat = self.py.hyper_expert["A_hat"]
            # b_hat = self.py.hyper_expert["b_hat"]
            # Sigma_hat = self.py.hyper_expert["Sigma_hat"]
            
            
            ###################################################################
            # --------------------------------------------------------------
            # cluster counts
            nk = q_z.sum(axis=0)
            
            delta_qz = np.linalg.norm(q_z-q_z_old) / np.linalg.norm(q_z_old)
            delta_beta = beta_z_0 - beta_z_0_old
            
            
            # Display each iteration
            print(iters+1, "nk:", nk)
            print(iters+1, "delta_qz=" + str(delta_qz))
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
            
            gamma_sb_stored.append(gamma_sb)
            alpha_stored.append(mean_alpha)
            sigma_stored.append(mean_sigma)
            beta_stored.append(beta_z_0)
            delta_qz_stored.append(delta_qz)
            
            m_hat_stored.append(m_niw)
            lambda_hat_stored.append(lambda_niw)
            psi_hat_stored.append(psi_niw)
            nu_hat_stored.append(nu_niw)
            
            ## Using append to store data as 
            # A_hat_stored = np.append(A_hat_stored, A_hat.T, axis = 0)
            # b_hat_stored = np.append(b_hat_stored, b_hat.T, axis = 0)
            # Sigma_hat_stored = np.append(Sigma_hat_stored, Sigma_hat.T, axis = 0)

            # alpha_stored = np.append(alpha_stored, mean_alpha)
            # sigma_stored = np.append(sigma_stored, mean_sigma)
            # beta_stored = np.append(beta_stored, beta_z_0)
            # delta_qz_stored = np.append(delta_qz_stored, delta_qz)
            
            # Stopping criterion
            # to do: Should this stopping criterion be changed?
            if self.sampling_model == "gaussian":
                if self.py.convergence(model, delta_beta, delta_qz,
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

  
        
        return (c_est, q_z, alpha_stored, sigma_stored,
                gamma_sb_stored,
                corr_as, beta_stored, delta_qz_stored, 
                A_hat_stored, b_hat_stored, Sigma_hat_stored,
                m_hat_stored, lambda_hat_stored, psi_hat_stored, nu_hat_stored)

	
    def inverse_expectation(self, X_hat, k_trunc, gamma_sb, A_hat, b_hat,
                                    m_hat, lambda_hat, psi_hat, nu_hat):
        
        (nb_data_hat, dim_data_X) = np.shape(X_hat)
        
        dim_data_Y = np.shape(A_hat)[1]
        
        Eyx = np.zeros((nb_data_hat, dim_data_Y), dtype=np.float64)
        
        
        ## TO DO LIST
        # Compare your own mvgaussian_pdf with multivariate_normal_gen. Fait.
        # or scipy.stats import multivariate_t and multivariate_t_gen. Fait.
        
        dofk_hat = np.ones((k_trunc, 1), dtype=np.float64)
        L = np.zeros((k_trunc, dim_data_X, dim_data_X), dtype=np.float64)
        
        for k in range(k_trunc):
            dofk_hat[k] = nu_hat[k] + 1 - dim_data_X
            L[k] = dofk_hat[k]*lambda_hat[k]*psi_hat[k] / (1+lambda_hat[k])
        
        ## Calculate the gating posteriors E_{q_tau}[pi_k(tau)].
        ## Require gamma_sb = np.zeros((k_trunc, 2), dtype=np.float64)
        
        gate_post = np.zeros((k_trunc, 1), dtype=np.float64)
        for k in range(1,k_trunc+1):
            prod_temp = 1.0
            for l in range(1,k):
                prod_temp *= gamma_sb[l-1,1]/(gamma_sb[l-1,0]+gamma_sb[l-1,1])      
            gate_post[k-1,0] = gamma_sb[k-1,0]/(gamma_sb[k-1,0]+gamma_sb[k-1,1])*prod_temp
        
        ## Compute prediction task: 
        #    A_hat = np.zeros((k_trunc, dim_data_Y, dim_data_X), dtype=np.float64)
        #    b_hat = np.zeros((k_trunc, dim_data_Y), dtype=np.float64)
        #    Sigma_hat = np.zeros((k_trunc, dim_data_Y, dim_data_Y), dtype=np.float64)     
        # gk: ([nb_data_hat, k_trunc], np.array)
        # psi_niw = np.zeros((k_trunc, dim_data, dim_data), dtype=np.float64)
        # m_niw = np.zeros((k_trunc, dim_data), dtype=np.float64)
        # lambda_niw = np.zeros(k_trunc, dtype=np.float64)
        # nu_niw = np.zeros(k_trunc, dtype=np.float64)
        gk = posterior_Student_gate(X_hat, gate_post, m_hat, L, dofk_hat)[0]
        
        
        for n in range(nb_data_hat):
            for k in range(k_trunc):
                Eyx[n,:,None] += gk[n,k] *  (A_hat[k,:,:]@(X_hat[n,:].T) + b_hat[k,:].T) 
        return Eyx, gk, gate_post
    
    def inverse_density(self, X_hat, k_trunc, gamma_sb,
                                    A_hat, b_hat, Sigma_hat,
                                    m_hat, lambda_hat, psi_hat, nu_hat):
        pyx_hat = 1
        
        return pyx_hat







