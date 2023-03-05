from distribution import Distribution
from vbem_utils_numba import init_by_kmeans, init_by_gmm, compute_weighted_mean, \
    compute_weighted_cov, compute_niw, expectation_log_gauss, compute_exp_inv_Gamma,\
        compute_normalizing_weighted_input, compute_normalizing_weighted_output,\
            compute_abs, discrete_entropy
            #, update_hyper_expert_init

class Gaussian(Distribution):
    """ 
    Collection of Gaussian emission distributions.
    
    self.parameter corresponds to parameters theta, whose first axis is k_trunc
    """

    def __init__(self, var_name, nbinit, dim=1):
        """
        Parameters
        ----------
        var_name : str
            name of variable in data file? (if any...).
        nbinit : int
            number of random initializations.
        dim : int, optional
            data dimension. The default is 1.

        Returns
        -------
        None.
        """

        # Set nbinit!: number of initializations of the Kmeans algorithm used in order
        # to find an initial partition
        self.var_name = var_name
        self.parameter = {"mean": 0.0, "Sigma": 0.0}
        self.rho = {"nu": 0.0, "psi": 0.0, "m": 0.0, "lambda": 0.0}
        self.hyper_expert = {"A_hat": 0.0, "b_hat": 0.0, "Sigma_hat": 0.0}
        self.hyper_gate = {"c_hat": 0.0, "Gamma_hat": 0.0}
        # self.hyper_expertS = {"A_hatS": 0.0, "b_hatS": 0.0, "Sigma_hatS": 0.0}
        # self.hyper_gateS = {"c_hatS": 0.0, "Gamma_hatS": 0.0}
        # initialization of A_hat, b_hat, Sigma_hat.
        self.dim = dim 	# dim-dimensional normal distribution	
        self.dist_name = "gaussian"
        # The number of initializations of the Kmeans algorithm used in order
        # to find an initial partition
        self.nb_init = nbinit  
        
        
    def __str__(self):
        return "variable " + self.var_name + ": " + self.dist_name +" distribution" \
                + "\nparameters: " + str(self.parameter) \
                + "\n prior parameters: " + str(self.rho)


    def init(self, data_X, data_Y, nb_data, dim_data_X, dim_data_Y, k_trunc,
             seed, nbinit):
        """
        Initialize variational q_z by k-means and update normal 
        inverse-Wishart parameters.

        Parameters
        ----------
        data_X : ([nb_data, dim_data_X] np.array)
            Input sample.
        data_Y : ([nb_data, dim_data_Y] np.array)
            Out sample.
        nb_data : int
            Sample size.
        dim_data_X : int
            data_X dimension.
        k_trunc : int
            truncation values of nb clusters.  
        seed : int
            return the same random numbers multiple times to get predictable, 
            repeatable results.
        nbinit : int
            the number of initializations of the Kmeans algorithm used in order
            to find an initial partition.

        Returns
        -------
        q_z, s1, s2, mean_sigma, beta_z_0.

        """
        import numpy as np
        # seet seed
        np.random.seed(seed)
        # Initialize variational q_z by k-means and update normal 
        # inverse-Wishart parameters

        # TestT
        # Maybe, initialization of q_z is wrong. We have to use the link 
        # between GMM and GLLiM run EMGM (EM algorithm for fitting the
        # joint Gaussian mixture model) on (data_X, data_Y)
        
        data =  np.concatenate((data_X, data_Y), axis=1)
        nb_data, dim_data = data.shape
        
        # m_niw_0, lambda_niw_0, psi_niw_0, nu_niw_0, q_z = init_by_kmeans(data, 
        #                              nb_data, dim_data, k_trunc, nbinit, seed=seed)
        
        q_z, labels, nk = init_by_gmm(data, nb_data, dim_data, data_X, dim_data_X,
                                      k_trunc, nbinit, seed=seed)        
        
        self.dim = dim_data_X
        
        
        ## Just for the general BNP-MRF-GLLiM
        # self.rho["m"] = m_niw_0 
        # self.rho["psi"] = psi_niw_0
        # self.rho["nu"] = nu_niw_0
        # self.rho["lambda"] = lambda_niw_0
        # nk = q_z.sum(axis=0)
        
        # TestT
        # c_est = np.argmax(q_z, axis=1)
        # import matplotlib.pyplot as plt  
        # plt.figure(1)
        # plt.scatter(data[:, 0], data[:, 1], c=c_est, s=40, cmap='viridis')
        # plt.title('Clustering ' + str(nb_data) +' realizations using Joint_GMM for initialization+' +\
        #           str(k_trunc) +' truncated clusters')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        
        # plt.show(1)
        
        # A_hat, b_hat, Sigma_hat = update_hyper_expert_init(data_X, q_z, nk, nb_data, 
        #                                  dim_data_X, k_trunc, data_Y, dim_data_Y)
        # self.hyper_expert["A_hat"]  = A_hat
        # self.hyper_expert["b_hat"]  = b_hat
        # self.hyper_expert["Sigma_hat"] = Sigma_hat
        
        # Better way: Referred to self.fit_predict(X, y) from sklearn/mixture/_base.py
        self.update_hyper_expert(data_X, q_z, nk, nb_data, dim_data_X, k_trunc,
                        data_Y, dim_data_Y)                  
        A_hat_init = self.hyper_expert["A_hat"]
        b_hat_init = self.hyper_expert["b_hat"]
        Sigma_hat_init = self.hyper_expert["Sigma_hat"]
        
        
        self.update_hyper_gate(data_X, q_z, nk, nb_data, dim_data_X, k_trunc)        
        c_hat_init = self.hyper_gate["c_hat"]
        Gamma_hat_init = self.hyper_gate["Gamma_hat"]
        
        ### Showing c_hat, Gamma_hat for each iteration.
        # print('c_hat_test = ', c_hat_init)
        # print('Gamma_hat_test = ', Gamma_hat_init)
        
        # Second way via self.
        # self.update_hyper_expert(data_X, q_z, nk, nb_data, 
        #                                  dim_data_X, k_trunc, data_Y, dim_data_Y)
        # self.hyper_expert["A_hatT"]  = A_hat
        # self.hyper_expert["b_hatT"]  = b_hat
        # self.hyper_expert["Sigma_hatT"] = Sigma_hat
        
	   # Markov Random field parameters initialization
        beta_z_0 = 0.0
        
        s1 = 1.0
        s2 = 200.0/k_trunc
        mean_sigma = 0.0
                
        return q_z, s1, s2, mean_sigma, beta_z_0, A_hat_init, b_hat_init,\
                Sigma_hat_init, c_hat_init, Gamma_hat_init
                    
    
    
    def update_posterior(self, data_X, q_z, nk, nb_data, dim_data_X, k_trunc):
        """
        Calculate variational posterior q_theta* by updating 
        hyperparameters rho = (m, lambda, psi, nu) using for VB-E-theta* step.
        Similar to standard Bayesian finite mixtures.

        Parameters
        ----------
        data_X : ([nb_data, dim_data_X] np.array)
            Sample.   
        q_z : ([nb_data, k_trunc] np.array)
            variational q_z distribution at previous iteration. 
        nk : ([1,k_trunc] np.array)
            Sums of variational posteriors of cluster variables z. 
        nb_data : int
            sample size.
        dim_data_X : int
            data_X dimension.
        k_trunc : int
            truncation values of nb clusters.    

        Remark
        -------
        rho : (m, lambda, psi, nu): 
            hyperparameters rho.
        dim_data_X is known by self and should not be passed as an argument.
        """
        m_niw_0 = self.rho["m"]
        lambda_niw_0 = self.rho["lambda"]
        psi_niw_0 = self.rho["psi"]
        nu_niw_0  = self.rho["nu"]
        weighted_mean = compute_weighted_mean(q_z, data_X, nk, nb_data, 
                                              dim_data_X, k_trunc)
        weighted_cov = compute_weighted_cov(q_z, data_X, weighted_mean, 
                                            nb_data, dim_data_X, k_trunc)
        m_niw, psi_niw, lambda_niw, nu_niw = compute_niw(nk, weighted_mean,
                                                     weighted_cov,
                                                     m_niw_0, lambda_niw_0,
                                                     psi_niw_0, nu_niw_0,
                                                     dim_data_X, k_trunc)
    
        self.rho["m"] = m_niw
        self.rho["lambda"] = lambda_niw
        self.rho["psi"] = psi_niw
        self.rho["nu"] = nu_niw
        
        return m_niw, psi_niw, lambda_niw, nu_niw
 
    # TrungTin Nguyen
    def update_hyper_gate(self, data_X, q_z, nk, nb_data, dim_data_X, k_trunc):
        """
        Updating hyperparameters (c, Gamma) using for VB-M-(c, Gamma) step.
        Similar to M-GMM-step.

        Parameters
        ----------
        data_X : ([nb_data, dim_data_X] np.array)
            Input sample.   
        q_z : ([nb_data, k_trunc] np.array)
            variational q_z distribution at previous iteration. 
        nk : ([1,k_trunc] np.array)
            Sums of variational posteriors of cluster variables z. 
        nb_data : int
            Sample size.
        dim_data_X : int
            Input data dimension.
        k_trunc : int
            Truncation values of nb clusters.
        """
        
        c_hat = compute_weighted_mean(q_z, data_X, nk, nb_data, dim_data_X, k_trunc)
        Gamma_hat = compute_weighted_cov(q_z, data_X, nk,  c_hat, nb_data,
                                         dim_data_X, k_trunc)
        
        self.hyper_gate["c_hat"]  = c_hat
        self.hyper_gate["Gamma_hat"]  = Gamma_hat
        
        return c_hat, Gamma_hat
       
    # TrungTin Nguyen
    def update_hyper_expert(self, data_X, q_z, nk, nb_data, dim_data_X, k_trunc,
                            data_Y, dim_data_Y):
        """
        Updating hyperparameters (A, b, Sigma) using for VB-M-(A, b, Sigma) step.
        Similar to M-mapping-step in GLLiM-EM.

        Parameters
        ----------
        data_X : ([nb_data, dim_data_X] np.array)
            Input sample.   
        q_z : ([nb_data, k_trunc] np.array)
            variational q_z distribution at previous iteration. 
        nk : ([1,k_trunc] np.array)
            Sums of variational posteriors of cluster variables z. 
        nb_data : int
            Sample size.
        dim_data_X : int
            Input data dimension.
        k_trunc : int
            Truncation values of nb clusters.    
        data_Y : ([nb_data_Y, dim_data_Y] np.array)
            Output sample.    
        dim_data_Y : int
            Output data dimension.
        """
        
        # What are the old values of (A_hat, b_hat, Sigma_hat).
        # A_hat = self.hyper_expert["A_hat"] 
        # b_hat = self.hyper_expert["b_hat"] 
        # Sigma_hat = self.hyper_expert["Sigma_hat"] 
        
        # m_niw_0 = self.rho["m"]
        # lambda_niw_0 = self.rho["lambda"]
        # psi_niw_0 = self.rho["psi"]
        # nu_niw_0  = self.rho["nu"]
        # import sys
        # sys.float_info.min
        
        weighted_mean_X = compute_weighted_mean(q_z, data_X, nk, nb_data, 
                                              dim_data_X, k_trunc)
        weighted_mean_Y = compute_weighted_mean(q_z, data_Y, nk, nb_data, 
                                              dim_data_Y, k_trunc)
        normalizing_weighted_input = compute_normalizing_weighted_input(q_z, 
                                        data_X, nk, nb_data, dim_data_X, k_trunc,
                                        weighted_mean_X)
        normalizing_weighted_output = compute_normalizing_weighted_output(q_z, 
                                          data_Y, nk, nb_data, dim_data_Y, k_trunc,                        
                                                  weighted_mean_Y)            
        A_hat, b_hat, Sigma_hat = compute_abs(q_z, data_X, data_Y, nk, normalizing_weighted_input,
                    normalizing_weighted_output, dim_data_X, dim_data_Y, k_trunc, nb_data)

        self.hyper_expert["A_hat"]  = A_hat
        self.hyper_expert["b_hat"]  = b_hat
        self.hyper_expert["Sigma_hat"] = Sigma_hat 
        
        # TestT
        # self.hyper_expert["A_hatT"]  = A_hat
        # self.hyper_expert["b_hatT"]  = b_hat
        # self.hyper_expert["Sigma_hatT"] = Sigma_hat    
        
        return A_hat, b_hat, Sigma_hat

    # # TrungTin Nguyen
    # def normalizing_weighted_input(self, q_z, data_X, nk, nb_data, dim_data_X, k_trunc):
 
    #     weighted_mean_X = compute_weighted_mean(q_z, data_X, nk, nb_data, 
    #                                           dim_data_X, k_trunc)
    #     normalizing_weighted_input = compute_normalizing_weighted_input(q_z, 
    #                                     data_X, nk, nb_data, dim_data_X, k_trunc,
    #                                     weighted_mean_X)
        
    #     return normalizing_weighted_input
    # TrungTin Nguyen
    def normalizing_weighted_input_output(self, data_X, q_z, nk, nb_data, dim_data_X, k_trunc,
                            data_Y, dim_data_Y):
        
        weighted_mean_X = compute_weighted_mean(q_z, data_X, nk, nb_data, 
                                              dim_data_X, k_trunc)
        weighted_mean_Y = compute_weighted_mean(q_z, data_Y, nk, nb_data, 
                                              dim_data_Y, k_trunc)
        normalizing_weighted_input = compute_normalizing_weighted_input(q_z, 
                                        data_X, nk, nb_data, dim_data_X, k_trunc,
                                        weighted_mean_X) 
        normalizing_weighted_output = compute_normalizing_weighted_output(q_z, 
                                          data_Y, nk, nb_data, dim_data_Y, k_trunc,                        
                                                  weighted_mean_Y)             
        
        return normalizing_weighted_input, normalizing_weighted_output
            
# TrungTin Nguyen.  [Eq. (59) Lu et al., 2020)].
    def expectation_likelihood(self, data_X, data_Y, nb_data, dim_data_X, dim_data_Y, k_trunc):
        """
         Compute -0.5 * expectation E_{q_theta^star} wrt:
             - Gaussian gating network, p(x | z, c, Gamma),
             - Gaussian experts, p(y | x, z; A, b, Sigma).
         Constant terms wrt states z may be omitted.

        Parameters
        ----------
        data_X : ([nb_data, dim_data_X] np.array)
            Input sample.    
        data_Y : ([nb_data, dim_data_Y] np.array)
            Output sample.        
        nb_data : int
            sample size.
        dim_data_X : int
            Input data dimension.
        dim_data_Y : int
            Output data dimension.            
        k_trunc : int
            truncation values of nb clusters.

        Returns
        -------
        [1, k_trunc] : np.array
            -0.5*log_gauss : ([nb_data, k_trunc] np.array)
        
        Remark:
        -------
        Constant terms wrt states z may be omitted.
        
        dim_data_X is known by self and should not be passed as an argument.

        """
        ### This is for the general BNP-MRF-GLLiM
        # m_niw = self.rho["m"]
        # lambda_niw = self.rho["lambda"]
        # psi_niw = self.rho["psi"]
        # nu_niw  = self.rho["nu"]
        
        A_hat = self.hyper_expert["A_hat"] 
        b_hat = self.hyper_expert["b_hat"] 
        Sigma_hat = self.hyper_expert["Sigma_hat"] 
        
        c_hat = self.hyper_gate["c_hat"] 
        Gamma_hat = self.hyper_gate["Gamma_hat"] 
        
        
        e = expectation_log_gauss(data_X, c_hat, Gamma_hat, nb_data, dim_data_X, 
                                  k_trunc, data_Y, A_hat, b_hat, Sigma_hat, dim_data_Y)
                                
        
        return e


    def convergence(self, model, delta_beta, min_deltas, threshold, spatial_interaction):
        
        if (model == 'dp-mrf' or model == 'pyp-mrf') and (spatial_interaction != 0.0):
            return min_deltas < threshold or delta_beta == 0.0
        else:
            return min_deltas < threshold
        
    def entropy_GateExperts(self, q_z, data_X, data_Y, nb_data, dim_data_X, dim_data_Y, k_trunc):
        exp_log = self.expectation_likelihood(data_X, data_Y, nb_data,
                                                 dim_data_X, dim_data_Y, k_trunc)
        entropy = 0.
        for n in range(nb_data):
            for k in range(k_trunc):
                if q_z[n, k] > 0:
                    entropy += q_z[n, k] * exp_log[n, k]

        return entropy
    
    ### For general BNP-GLLiM.
    # def entropy_theta(self, data_X, q_z):
    #     """
    #     Contribution of variational q_theta distribution to entropy
    #     (up to constants wrt hyperparameters Phi and variational parameters)

    #     :Parameters:
    #       `data_X` (numpy.array) - data_X
    #       `q_z` (([nb_data, k_trunc] np.array)) - variational approximation 
    #           of state distribution       
    #     """
    #     import math
    #     from scipy import special
    #     import numpy as np
    #     e = 0.
        
    #     k_trunc = self.rho["lambda"].shape[0]
        
    #     nb_data = data_X.shape[0]
    #     dim_data_X = data_X.shape[1]
    #     nk = q_z.sum(axis=0)
    #     m_niw = self.rho["m"]
    #     lambda_niw = self.rho["lambda"]
    #     psi_niw = self.rho["psi"]
    #     nu_niw  = self.rho["nu"]
    #     # Includes + dim_data_X / lambda_niw[k]
    #     inv_psi = compute_exp_inv_psi(data_X, m_niw, lambda_niw, psi_niw, 
    #                                   nu_niw, nb_data, dim_data_X, k_trunc)
    #     inv_psi = np.multiply(inv_psi, q_z)
    #     for k in range(k_trunc):
    #         psi_d = 0. # mutltivariate digamma function
    #         for d in range(self.dim):
    #             psi_d += special.digamma((nu_niw[k] + 1 - d)/2)
    #         e += nk[k] * (-math.log(np.linalg.det(psi_niw[k])) + \
    #                dim_data_X * math.log(2) + psi_d)
    #     e = e - inv_psi.sum()    
    #     e = 0.5 * e
        
    #     return(e)
