#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""vbem_utils_numba module

.. module:: vbem_utils_numba
    :synopsis: A module dedicated to VBEM steps relying on numba


.. topic:: vbem_utils_numba.py summary

    A module dedicated to VBEM steps relying on numba (acceleration by 
    automatic parallelization)

    :Code status: in development
    :Documentation status: to be completed
    :Author: 
    :Revision: $Id: vbem_utils_numba.py$
    :Usage: >>> from vbem_utils_numba import *
 
"""
import numpy as np
import numba as nb
from scipy import stats
from scipy import special
from scipy import optimize
import math
from scipy.stats import multivariate_normal, multivariate_t

#from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

# may work on some images.
# 'safe' 'threadsafe' 'omp' would not

# 'workqueue''forksafe' would work
# nb.config.THREADING_LAYER = 'forksafe' 
# nb.config.THREADING_LAYER = 'workqueue'

#from potts_model import draw_gradient_beta
###############################################################################
####                        Functions for VB-E steps
###############################################################################

# Compute the Potts part
##@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def pairwise_energy(q_z, nb_data, k_trunc, beta_z, neighborhood):

    local_energy = np.zeros((nb_data, k_trunc), dtype=np.float64)
    for n in range(nb_data):
        local_list = neighborhood[n, :]
        neighbors = local_list[local_list > -1]
        for k in range(k_trunc):
            aux = 0.0
            for i in range(len(neighbors)):
                aux += q_z[neighbors[i], k]
            local_energy[n, k] = beta_z * aux

    return local_energy

# TrungTin Nguyen
# Update the hyperparameters for normal inverse wishart distribution in VB-M-rho step.
# Update the hyperparameters for Gaussian experts in VB-M-(A,b,Sigma) step.
#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def compute_weighted_mean(q_z, data, nk, nb_data, dim_data, k_trunc):
    """
    Sample mean with q_z weights.


    Parameters
    ----------
    q_z : ([nb_data, k_trunc] np.array)
        variational q_z distribution at previous iteration.
    data : ([nb_data, dim_data] np.array)
        Sample.    
    nk : ([1,k_trunc] np.array)
        Sums of variational posteriors of cluster variables z. 
    nb_data : int
        sample size.
    dim_data : int
        data dimension.
    k_trunc : int
        truncation values of nb clusters.

    Returns
    -------
    weighted_mean : ([k_trunc, dim_data] np.array)
        Sample mean with q_z weights.

    """
    
    weighted_mean = np.zeros((k_trunc, dim_data), dtype=np.float64)
    for k in nb.prange(k_trunc):
        for d in nb.prange(dim_data):
            for n in nb.prange(nb_data):
                weighted_mean[k, d] += q_z[n, k] * data[n, d]
            # safe division:
            if nk[k] > 0.0:
                weighted_mean[k, d] = weighted_mean[k, d] / nk[k]

    return weighted_mean

# TrungTin Nguyen
#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def compute_normalizing_weighted_input(q_z, data, nk, nb_data, dim_data, k_trunc,
                                         weighted_mean_X):
    """
    Normalizing weighted sample matrix input and output data with q_z weights.

    Parameters
    ----------        
    q_z : ([nb_data, k_trunc] np.array)
        variational q_z distribution at previous iteration. 
    data : ([nb_data, dim_data] np.array)
        Input sample.           
    nk : ([1,k_trunc] np.array)
        Sums of variational posteriors of cluster variables z. 
    nb_data : int
        Sample size.
    dim_data : int
        Input data dimension.
    k_trunc : int
        truncation values of nb clusters.    
    weighted_mean_X: ([k_trunc, dim_data] np.array)
        Input sample mean with q_z weights.   
    Returns
    -------
    normalizing_weighted_input : ([k_trunc, dim_data, nb_data] np.array)
        Normalizing weighted sample matrix input data.
    """
    
    normalizing_weighted_input = np.zeros((k_trunc, dim_data, nb_data), dtype=np.float64)    
    
    for k in nb.prange(k_trunc):
        # Update normalizing_weighted_input
        for d in nb.prange(dim_data):
            for n in nb.prange(nb_data):
                normalizing_weighted_input[k, d, n] = \
                    q_z[n, k]**(1/2)*(data[n, d] - weighted_mean_X[k, d])
            # # safe division:
            # if nk[k] > 0.0:
            #     normalizing_weighted_input[k, d, n] = \
            #         normalizing_weighted_input[k, d, n] * (nk[k]**(-1/2))
                
    return normalizing_weighted_input

# TrungTin Nguyen
#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def compute_normalizing_weighted_output(q_z, data_Y, nk, nb_data, dim_data_Y, k_trunc,                        
                                          weighted_mean_Y):
    """
    Normalizing weighted sample matrix input and output data with q_z weights.

    Parameters
    ----------        
    q_z : ([nb_data, k_trunc] np.array)
        variational q_z distribution at previous iteration.
    data_Y : ([nb_data, dim_data_Y] np.array)
        Output sample.         
    nk : ([1,k_trunc] np.array)
        Sums of variational posteriors of cluster variables z. 
    nb_data : int
        Sample size.
    dim_data_Y : int
        Output data dimension.
    k_trunc : int
        truncation values of nb clusters.    
    weighted_mean_Y: ([k_trunc, dim_data_Y] np.array)
        Output sample mean with q_z weights.   
    Returns
    -------
    normalizing_weighted_output : ([k_trunc, dim_data_Y, nb_data] np.array)
        Normalizing weighted sample matrix output data.
    """
    normalizing_weighted_output = np.zeros((k_trunc, dim_data_Y, nb_data), dtype=np.float64)    
    
    for k in nb.prange(k_trunc):               
        # Update normalizing_weighted_output
        for d in nb.prange(dim_data_Y):
            for n in nb.prange(nb_data):
                normalizing_weighted_output[k, d, n] = \
                    q_z[n, k]**(1/2)*(data_Y[n, d] - weighted_mean_Y[k, d])
            # # # safe division:
            # if nk[k] > 0.0:
            #     normalizing_weighted_output[k, d, n] = \
            #         normalizing_weighted_output[k, d, n] * (nk[k]**(-1/2))
    return normalizing_weighted_output

# TrungTin Nguyen
#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def compute_abs(q_z, data, data_Y, nk, normalizing_weighted_input,
                normalizing_weighted_output, dim_data, dim_data_Y, k_trunc, nb_data):
    """
    VB-M-(A,b,Sigma) step:
    Compute updated hyperparameters in Gaussian experts for BNP-GLLiM

    Parameters
    ----------
    q_z : ([nb_data, k_trunc] np.array)
        variational q_z distribution at previous iteration.
    data : ([nb_data, dim_data] np.array)
        Input sample.          
    data_Y : ([nb_data, dim_data_Y] np.array)
        Output sample.            
    nk : ([1,k_trunc] np.array)
        Sums of variational posteriors of cluster variables z. 
    normalizing_weighted_input : ([k_trunc, dim_data, nb_data] np.array)
        Sample mean with q_z weights.
    normalizing_weighted_output : ([k_trunc, dim_data_Y, nb_data] np.array)
        Normalizing weighted sample matrix output data.
    dim_data : int
        Input data dimension.        
    dim_data_Y : int
        Output data dimension.
    k_trunc : int
        truncation values of nb clusters.
    nb_data : int
        Sample size.        

    Returns
    -------
    Computed parameters A_hat, b_hat, Sigma_hat

    """
    A_hat = np.zeros((k_trunc, dim_data_Y, dim_data), dtype=np.float64)
    b_hat = np.zeros((k_trunc, dim_data_Y), dtype=np.float64)
    Sigma_hat = np.zeros((k_trunc, dim_data_Y, dim_data_Y), dtype=np.float64) 
    
    
    ### Testing singular matrix errors for Sigma_hat.
    # Sigma_hatH = np.zeros((k_trunc, dim_data_Y, dim_data_Y), dtype=np.float64)
    
    # for k in range(k_trunc):
    #     # Update A_hat_k
    #     A_hat[k] = normalizing_weighted_output[k]@(normalizing_weighted_input[k].T)\
    #         @np.linalg.inv(normalizing_weighted_input[k]@(normalizing_weighted_input[k].T))
            
            
    #     # Update b_hat_k
    #     for d in nb.prange(dim_data_Y):
    #         for n in nb.prange(nb_data):
    #             b_hat[k, d] += q_z[n, k] * (data_Y[n, d] - (A_hat[k]@data[n])[d])
    #         # safe division:
    #         if nk[k] > 0.0:
    #             b_hat[k, d] = b_hat[k, d] / nk[k]
    #     # Update Sigma_hatH
    #     for n in range(nb_data):
    #         for d1 in range(dim_data_Y):
    #             for d2 in range(dim_data_Y):
    #                 cov_element = (data_Y[n, d1] - (A_hat[k]@data[n])[d1] - b_hat[k, d1])\
    #                             * (data_Y[n, d2] - (A_hat[k]@data[n])[d2] - b_hat[k, d2])
    #                 Sigma_hatH[k, d1, d2] += q_z[n, k] * cov_element 
    #         # Test:
    #         Sigma_hat[k, 0, 0]  += q_z[n, k] * ((data_Y[n, 0] - (A_hat[k]@data[n])[0] - b_hat[k, 0])**2)

    #     # safe division:
    #     if nk[k] > 0.0:
    #         Sigma_hatH[k, d1, d2] = Sigma_hatH[k, d1, d2] / nk[k]
    #     # safe division:
    #     if nk[k] > 0.0:
    #         Sigma_hat[k, 0, 0] = Sigma_hat[k, 0, 0] / nk[k]

    # print('Sigma_hatH = ', Sigma_hatH)
    # print('Sigma_hat_test = ', Sigma_hat)
    # print('A_hat_test = ', A_hat)
    # print('b_hat_test = ', b_hat)
    # print('nk_test = ', nk)
    
    for k in range(k_trunc):
        # Update A_hat_k
        A_hat[k] = normalizing_weighted_output[k]@(normalizing_weighted_input[k].T)\
            @np.linalg.inv(normalizing_weighted_input[k]@(normalizing_weighted_input[k].T))
            
        # Update b_hat_k
        for d in nb.prange(dim_data_Y):
            for n in nb.prange(nb_data):
                b_hat[k, d] += q_z[n, k] * (data_Y[n, d] - (A_hat[k]@data[n])[d])
            # safe division:
            if nk[k] > 0.0:
                b_hat[k, d] = b_hat[k, d] / nk[k]
        # Update Sigma_hatH
        for n in range(nb_data):
            for d1 in range(dim_data_Y):
                for d2 in range(dim_data_Y):
                    cov_element = (data_Y[n, d1] - (A_hat[k]@data[n])[d1] - b_hat[k, d1])\
                                * (data_Y[n, d2] - (A_hat[k]@data[n])[d2] - b_hat[k, d2])
                    Sigma_hat[k, d1, d2] += q_z[n, k] * cov_element 
        # safe division:
        if nk[k] > 0.0:
            Sigma_hat[k, d1, d2] = Sigma_hat[k, d1, d2] / nk[k]

    print('Sigma_hat_test = ', Sigma_hat)
    print('A_hat_test = ', A_hat)
    print('A_hat_test_shape = ', A_hat.shape)
    print('b_hat_test = ', b_hat)
    print('nk_test = ', nk)
    
    return A_hat, b_hat, Sigma_hat
#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def compute_weighted_cov(q_z, data, nk, weighted_mean, nb_data, dim_data, k_trunc):
    """
    Sample covariance with q_z weights.

    Parameters
    ----------
    q_z : ([nb_data, k_trunc] np.array)
        variational q_z distribution at previous iteration.
    data : ([nb_data, dim_data] np.array)
        Input sample.    
    weighted_mean : ([k_trunc, dim_data] np.array)
        Sample mean with q_z weights.        
    nb_data : int
        sample size.
    dim_data : int
        data dimension.
    k_trunc : int
        truncation values of nb clusters.    

    Returns
    -------
    weighted_cov : ([k_trunc, dim_data, dim_data] np.array)
        Sample covariance with q_z weights.

    """
    
    weighted_cov = np.zeros((k_trunc, dim_data, dim_data), dtype=np.float64)
    for k in range(k_trunc):
        for n in range(nb_data):
            for d1 in range(dim_data):
                for d2 in range(dim_data):
                    cov_element = (data[n, d1] - weighted_mean[k, d1]) \
                                * (data[n, d2] - weighted_mean[k, d2])
                    weighted_cov[k, d1, d2] += q_z[n, k] * cov_element
        # safe division:
        if nk[k] > 0.0:
            weighted_cov[k, d1, d2] = weighted_cov[k, d1, d2] / nk[k]     
    return weighted_cov


#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def compute_niw(nk, weighted_mean, weighted_cov, 
                m_niw_0, lambda_niw_0, psi_niw_0, nu_niw_0, 
                dim_data, k_trunc):
    """
    VE-theta* step:
    Compute updated hyperparameters in Gaussian distributions with
    normal-inverse-Wishart priors

    Parameters
    ----------
    nk : ([1,k_trunc] np.array)
        Sums of variational posteriors of cluster variables z. 
    weighted_mean : ([k_trunc, dim_data] np.array)
        Sample mean with q_z weights.
    weighted_cov : ([k_trunc, dim_data, dim_data] np.array)
        Sample covariance with q_z weights.
    m_niw_0 : numpy.array
        previous location parameters of normal inverse Wishart.
    lambda_niw_0 : numpy.array
        previous precision parameters of normal inverse Wishart.
    psi_niw_0 : numpy.array
        previous scale parameters of inverse Wishart.
    nu_niw_0 : numpy.arra
        previous df parameters of inverse Wishart.
    dim_data : int
        data dimension.
    k_trunc : int
        truncation values of nb clusters.

    Returns
    -------
    Computed parameters m_niw, psi_niw, lambda_niw, nu_niw

    """
    
    psi_niw = np.zeros((k_trunc, dim_data, dim_data), dtype=np.float64)
    m_niw = np.zeros((k_trunc, dim_data), dtype=np.float64)
    lambda_niw = np.zeros(k_trunc, dtype=np.float64)
    nu_niw = np.zeros(k_trunc, dtype=np.float64)

    for k in range(k_trunc):
        lambda_niw[k] = lambda_niw_0[k] + nk[k]
        nu_niw[k] = nu_niw_0[k] + nk[k]
        for d1 in range(dim_data):
            for d2 in range(dim_data):
                psi_element = (m_niw_0[k, d1] - weighted_mean[k, d1]) \
                            * (m_niw_0[k, d2] - weighted_mean[k, d2])
                psi_niw[k, d1, d2] = psi_niw_0[k, d1, d2] \
                                + nk[k]*weighted_cov[k, d1, d2] \
                                + (lambda_niw_0[k]*nk[k] / lambda_niw[k]) \
                                * psi_element
        
            m_niw[k, d1] = (lambda_niw_0[k]*m_niw_0[k, d1]
                        + nk[k]*weighted_mean[k, d1]) / lambda_niw[k]

    return m_niw, psi_niw, lambda_niw, nu_niw


def update_niw(data, q_z, nk, m_niw_0, lambda_niw_0, psi_niw_0, nu_niw_0,
               nb_data, dim_data, k_trunc):
    """
    VE-theta* step:
    Update hyperparameters in Gaussian distributions with
    normal-inverse-Wishart priors    

    Parameters
    ----------
    data : ([nb_data, dim_data] np.array)
        Sample.
    q_z : ([nb_data, k_trunc] np.array)
        variational q_z distribution at previous iteration.
    nk : ([1,k_trunc] np.array)
        Sums of variational posteriors of cluster variables z. 
    m_niw_0 : numpy.array
        previous location parameters of normal inverse Wishart.
    lambda_niw_0 : numpy.array
        previous precision parameters of normal inverse Wishart.
    psi_niw_0 : numpy.array
        previous scale parameters of inverse Wishart.
    nu_niw_0 : numpy.arra
        previous df parameters of inverse Wishart.
    nb_data : int
        sample size.
    dim_data : int
        data dimension.
    k_trunc : int
        truncation values of nb clusters.

    Returns
    -------
    Updated parameters m_niw, psi_niw, lambda_niw, nu_niw


    """
  
    weighted_mean = compute_weighted_mean(q_z, data, nk, 
                                        nb_data, dim_data, k_trunc)
    weighted_cov = compute_weighted_cov(q_z, data, nk, weighted_mean, 
                                        nb_data, dim_data, k_trunc)
    
    m_niw, psi_niw, lambda_niw, nu_niw = compute_niw(nk, weighted_mean,
                                                     weighted_cov,
                                                     m_niw_0, lambda_niw_0,
                                                     psi_niw_0, nu_niw_0,
                                                     dim_data, k_trunc)
    
    return m_niw, psi_niw, lambda_niw, nu_niw

############
# TrungTin Nguyen
# Compute q_Z in VB-E-Z step [Eq. (61) Lu et al., 2020)].
#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def compute_exp_inv_psi(data_X, m_niw, lambda_niw, psi_niw, nu_niw, 
                        nb_data, dim_data_X, k_trunc):
    """
    Compute quadratic part q of expectation E_{q_theta^star} in 
    Gaussian gating network, p(x | z, c, Gamma), with
    normal-inverse-Wishart-type parametrization.

    Parameters
    ----------
    data_X : ([nb_data, dim_data_X] np.array)
        Input sample.    
    m_niw : ([k_trunc, dim_data_X] np.array)
        location parameters of normal inverse Wishart.
    lambda_niw : ([k_trunc, dim_data_X, dim_data] np.array)
        normal concentration parameters of normal inverse Wishart.
    psi_niw : ([k_trunc] np.array)
        scale parameters of inverse Wishart.
    nu_niw : ([k_trunc] np.array)
        df parameters of inverse Wishart.
    nb_data : int
        sample size.
    dim_data_X : int
        data dimension.
    k_trunc : int
        truncation values of nb clusters.

    Returns
    -------
    inv_psi : ([nb_data, k_trunc] np.array)

    Returns
    -------
    - Includes dim_data_X / lamda_niw[k] + nu_niw[k] * e 
    - Does not include * -0.5    

    """

    inv_psi = np.zeros((nb_data, k_trunc), dtype=np.float64)
    
    for k in range(k_trunc):
        A = dim_data_X / lambda_niw[k]
        psi_niw_inv = np.linalg.inv(psi_niw[k])
        for n in range(nb_data):
            iw = 0.0
            for d1 in range(dim_data_X):
                for d2 in range(dim_data_X):
                    iw += (data_X[n, d1] - m_niw[k, d1]) \
                        * psi_niw_inv[d1, d2] \
                        * (data_X[n, d2] - m_niw[k, d2])
            inv_psi[n, k] = A + nu_niw[k] * iw

    return inv_psi

#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def compute_exp_inv_Sigma(data_X, data_Y, A_hat, b_hat, Sigma_hat, nb_data, dim_data_Y, k_trunc):
    """
    Compute quadratic part q of expectation E_{q_theta^star} in 
    Gaussian experts, p(y | x, z; A, b, Sigma).

    Parameters
    ----------
    data_X : ([nb_data, dim_data_X] np.array)
        Input sample.     
    data_Y : ([nb_data, dim_data_Y] np.array)
        Output sample.    
    A_hat : ([k_trunc, dim_data_Y, dim_data_X] np.array)
        Regressor matrix of location parameters of normal distribution.       
    b_hat : ([k_trunc, dim_data_Y] np.array)
        Intercept vector of location parameters of normal distribution.       
    Sigma_hat : ([k_trunc, dim_data_Y, dim_data_Y] np.array)
        Scale parameters (Gaussian experts' covariance matrices) of normal distribution.
    nb_data : int
        Sample size.
    dim_data_Y : int
        Output data dimension.
    k_trunc : int
        Truncation values of nb clusters.

    Returns
    -------
    inv_psi : ([nb_data, k_trunc] np.array)
        - Does not include * -0.5    

    """
    
    inv_Sigma = np.zeros((nb_data, k_trunc), dtype=np.float64)
    
    for k in range(k_trunc):
        Sigma_hat_inv = np.linalg.inv(Sigma_hat[k])
        for n in range(nb_data):
            for d1 in range(dim_data_Y):
                for d2 in range(dim_data_Y):
                    inv_Sigma[n, k] += (data_Y[n, d1] - (A_hat[k]@data_X[n])[d1] - b_hat[k, d1])\
                        * Sigma_hat_inv[d1, d2] \
                        * (data_Y[n, d2] - (A_hat[k]@data_X[n])[d2] - b_hat[k, d2])

    return inv_Sigma

#@nb.jit(nopython=False, nogil=True, fastmath=True, parallel=True)
def expectation_log_gauss(data_X, m_niw, lambda_niw, psi_niw, nu_niw,
                          nb_data, dim_data_X, k_trunc, data_Y, A_hat,
                          b_hat, Sigma_hat, dim_data_Y):
    """
    Compute -0.5 * expectation E_{q_theta^star} wrt:
        - Gaussian gating network, p(x | z, c, Gamma), with normal-inverse-Wishart-type parametrization,
        - Gaussian experts, p(y | x, z; A, b, Sigma).
    Constant terms wrt states z may be omitted.

    Parameters
    ----------
    data_X : ([nb_data, dim_data_X] np.array)
        Input sample.    
    m_niw : ([k_trunc, dim_data_X] np.array)
        Location parameters of normal inverse Wishart.
    lambda_niw : ([k_trunc] np.array)
        Normal concentration parameters of normal inverse Wishart.
    psi_niw : ([k_trunc] np.array)
        Scale parameters of inverse Wishart.
    nu_niw : ([k_trunc] np.array)
        df parameters of inverse Wishart.
    nb_data : int
        Sample size.
    dim_data_X : int
        Input data dimension.
    k_trunc : int
        Truncation values of nb clusters.
    data_Y : ([nb_data, dim_data_Y] np.array)
        Output sample.    
    A_hat : ([k_trunc, dim_data_Y, dim_data_X] np.array)
        Regressor matrix of location parameters of normal distribution.       
    b_hat : ([k_trunc, dim_data_Y] np.array)
        Intercept vector of location parameters of normal distribution.       
    Sigma_hat : ([k_trunc, dim_data_Y, dim_data_Y] np.array)
        Scale parameters (Gaussian experts' covariance matrices) of normal distribution.    
    dim_data_Y : int
        Output data dimension.
    Returns
    -------
    -0.5*log_gauss : ([nb_data, k_trunc] np.array)

    """

    inv_psi = compute_exp_inv_psi(data_X, m_niw, lambda_niw, psi_niw, nu_niw, 
                                    nb_data, dim_data_X, k_trunc)
    inv_Sigma = compute_exp_inv_Sigma(data_X, data_Y, A_hat, b_hat,
                                      Sigma_hat, nb_data, dim_data_Y, k_trunc)
    
    log_cov_psi = np.zeros((1, k_trunc), dtype=np.float64)
    log_cov_Sigma = np.zeros((1, k_trunc), dtype=np.float64)   
    for k in range(k_trunc):   
        det_psi = max(np.linalg.det(psi_niw[k]*0.5), 1e-300)
        ld = math.log(det_psi)
        # det_Sigma = max(np.linalg.det(Sigma_hat[k]*0.5), 1e-300)
        det_Sigma = max(np.linalg.det(Sigma_hat[k]), 1e-300)
        aux = 0.0
        for d in range(dim_data_X):
            aux += special.digamma((nu_niw[k]-d)*0.5)
        log_cov_psi[0, k] = ld - aux
        log_cov_Sigma[0, k] = math.log(det_Sigma)

    log_gauss = log_cov_psi + inv_psi + log_cov_Sigma + inv_Sigma

    return -0.5*log_gauss

############

#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def normalize_exp_qz(q_z, nb_data, k_trunc):
    
    exp_q_z_norm = np.zeros((nb_data, k_trunc), dtype=np.float64)
    for n in range(nb_data):
        exp_q_z_norm[n, :] = q_z[n, :] - np.max(q_z[n, :])
        sum_aux = 0.0
        for k in range(k_trunc):
            sum_aux += math.exp(exp_q_z_norm[n, k])
        for k in range(k_trunc):
            exp_q_z_norm[n, k] = math.exp(exp_q_z_norm[n, k]) / sum_aux
        
    return exp_q_z_norm


#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
## ?? Why we take the normalization here.
## Old definition.
# def normalize_exp_qzj(q_zj, k_trunc):
#     """
#     Normalize number according to exponential function from qzj for VB-E-Z step.

#     Parameters
#     ----------
#     q_zj : ([1, k_trunc] np.array)
#         Unnormalized exp of variational q_z distribution at current iteration.
#     k_trunc : int
#         Truncation values of nb clusters.

#     Returns
#     -------
#     exp_q_zj_norm : ([1, k_trunc] np.array)
#         Normalized exp of variational q_z distribution at current iteration.

#     """
    
#     exp_q_zj_norm = q_zj - np.max(q_zj)
#     sum_aux = 0.0
#     for k in range(k_trunc):
#         sum_aux += math.exp(exp_q_zj_norm[k])
#     for k in range(k_trunc):
#         exp_q_zj_norm[k] = math.exp(exp_q_zj_norm[k]) / sum_aux
        
#     return exp_q_zj_norm

def normalize_exp_qzj(q_zj, k_trunc):
    """
    Normalize number according to exponential function from qzj for VB-E-Z step.

    Parameters
    ----------
    q_zj : ([1, k_trunc] np.array)
        Unnormalized exp of variational q_z distribution at current iteration.
    k_trunc : int
        Truncation values of nb clusters.

    Returns
    -------
    exp_q_zj_norm : ([1, k_trunc] np.array)
        Normalized exp of variational q_z distribution at current iteration.

    """
    exp_q_zj_norm = q_zj - np.max(q_zj)
    # exp_q_zj_norm = q_zj
    sum_aux = 0.0
    for k in range(k_trunc):
        sum_aux += math.exp(exp_q_zj_norm[k])
    for k in range(k_trunc):
        exp_q_zj_norm[k] = math.exp(exp_q_zj_norm[k]) / sum_aux
        
    return exp_q_zj_norm

'''
#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def compute_local_energy(q_z, n, k_trunc, beta_z_0, neighborhood):
    
    local_energy = np.zeros(k_trunc, dtype=np.float64)
    if beta_z_0 > 0.0:
        local_list = neighborhood[n, :]
        neighbors = local_list[local_list > -1]
        for k in range(k_trunc):    
            for n in range(len(neighbors)):
                local_energy[k] += q_z[neighbors[n], k]

    return local_energy * beta_z_0


#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def compute_log_one_minus_tau(log_one_minus_tau, k_trunc):
    
    log_one_minus_tau_sum = np.zeros(k_trunc, dtype=np.float64)
    for k in range(k_trunc):
        for l in range(k):
            log_one_minus_tau_sum[k] += log_one_minus_tau[l]

    return log_one_minus_tau_sum
'''

# TrungTin Nguyen
# Update q_z for BNP-MRF-GLLiM models. [Eq. (38) Lu et al., 2020)].
#@nb.jit(nogil=True, fastmath=True)
def expectation_z(q_z, beta_z_0, exp_log, log_tau, log_one_minus_tau, 
                    nb_data, k_trunc, neighborhood):
    """
    VB-E-Z-step:
        Update clusters's labels 

    Parameters
    ----------
    q_z : ([nb_data, k_trunc] np.array)
        Variational q_z distribution at previous iteration.
    beta_z_0 : positive scalar
        Initial interaction parameter.
    exp_log : [nb_data, k_trunc] np.array
        Compute -0.5 * expectation of log covariance determinant 
        for Gaussian distributions under variational theta* distribution. 
    log_tau : float
        Value of E[log(tau)].
    log_one_minus_tau : ([k_trunc] np.array)
        Value of E[log(1-tau)].
    nb_data : int
        Sample size.
    k_trunc : int
        Truncation values of nb clusters.
    neighborhood : Array of int64 ([nb_data, nb_neighbors] np.array)
        Neighborhood.

    Returns
    -------
    q_z_new : ([nb_data, k_trunc] np.array)
        variational q_z distribution at current iteration.

    """
    
    q_z_new = np.copy(q_z)
    
    log_one_minus_tau_aux = np.zeros(k_trunc, dtype=np.float64)
    for k in range(k_trunc):
        log_one_minus_tau_aux[k] = np.sum(log_one_minus_tau[0:k])

    for n in range(nb_data):
        
        local_energy = np.zeros(k_trunc, dtype=np.float64)
        
        if beta_z_0 != 0.0:
            local_list = neighborhood[n, :]
            neighbors = local_list[local_list > -1]
            tmp = np.sum(q_z_new[neighbors, :], axis=0)
            local_energy = beta_z_0 * tmp
            
        q_z_new[n, :] = exp_log[n, :] + log_tau + log_one_minus_tau_aux + local_energy
                    
        # normalization of q_zj:
        q_z_new[n, :] = normalize_exp_qzj(q_z_new[n, :], k_trunc)

    return q_z_new


def compute_log_taus(gamma_sb):
    
    second_term = special.digamma(np.sum(gamma_sb, axis=1))
    # Calculate E[log(tau)]
    log_tau = special.digamma(gamma_sb.T[0]) - second_term
    # Calculate E[log(1-tau)]
    log_one_minus_tau = special.digamma(gamma_sb.T[1]) - second_term
    
    return log_tau, log_one_minus_tau

###############################################################################
#                        Functions for VB-M steps
###############################################################################
# Update the hyperparameters gamma_k
#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def update_gamma_sb(nk, mean_alpha, mean_sigma, k_trunc):

    gamma_sb = np.zeros((k_trunc, 2), dtype=np.float64)
    for k in range(k_trunc):
        sum_aux = 0.0
        for l in range(k+1, k_trunc):
            sum_aux += nk[l]
        gamma_sb[k, 0] = 1.0 - mean_sigma + nk[k]
        #gamma_sb[k, 1] = mean_alpha + (k+1)*mean_sigma + sum_aux
        # Tin VB-E-tau update: 2022-09-24
        gamma_sb[k, 1] = mean_alpha + k*mean_sigma + sum_aux 

    return gamma_sb


def compute_dp_means(log_tau, log_one_minus_tau, s1_dp, s2_dp, k_trunc):
    
    mean_alpha = 0.0
    mean_sigma = 0.0
    if s1_dp > 0.0 and s2_dp > 0.0:
        mean_alpha = s1_dp/s2_dp
    else:
        mean_alpha = 0.0
    return mean_alpha, mean_sigma


# Importance sampling     
def py_samples(log_tau, log_one_minus_tau, s1_py, s2_py, k_trunc, nb_samples):
    """
    Importance sampling scheme for PY case
    to approximate variational posteriors mean_sigma, mean_alpha

    Parameters
    ----------
    log_tau : TYPE
        DESCRIPTION.
    log_one_minus_tau : TYPE
        DESCRIPTION.
    s1_py : TYPE
        DESCRIPTION.
    s2_py : TYPE
        DESCRIPTION.
    k_trunc : int
        truncation values of nb clusters.
    nb_samples : TYPE
        DESCRIPTION.

    Returns
    -------
    alpha_samples : TYPE
        DESCRIPTION.
    sigma_samples : TYPE
        DESCRIPTION.
    weights : TYPE
        Importance weights.

    """
    
    ksi = 0.0
    for k in range(k_trunc-1):
        ksi += log_tau[k] - k*log_one_minus_tau[k]
    
    # Generate samples for alpha
    scale = 1.0/s2_py
    alpha_samples = stats.gamma.rvs(s1_py, scale=scale, size=nb_samples)
    
    # Generate samples for sigma
    sigma_samples = stats.uniform.rvs(0.0, 1.0, size=nb_samples)

    alpha_samples = alpha_samples - sigma_samples

    weights = np.array([], dtype=np.float64) 
    for l in range(nb_samples):
        arg1 = 1.0 - sigma_samples[l]
        arg2 = alpha_samples[l]+(k_trunc-1)*sigma_samples[l]
        fact = special.gamma(alpha_samples[l]) \
            / (special.gamma(arg2)*special.gamma(arg1)**(k_trunc-1))
        aux = 1.0
        for k in range(k_trunc-1):
            tot = alpha_samples[l] + sigma_samples[l]
            aux *= (alpha_samples[l] + k*sigma_samples[l])/tot
        arg3 = max(ksi*sigma_samples[l], -700.0)
        weights = np.append(weights, aux*fact*math.exp(-arg3))

    weights = weights/np.sum(weights)

    return alpha_samples, sigma_samples, weights


#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def compute_py_means(alpha_samples, sigma_samples, weights):
    
    mean_alpha = 0.0
    mean_sigma = 0.0
    mean_log_aps = 0.0
    for l in range(len(weights)):
        mean_alpha += weights[l]*alpha_samples[l]
        mean_sigma += weights[l]*sigma_samples[l]
        mean_log_aps += weights[l]*math.log(alpha_samples[l]+sigma_samples[l])
    
    return mean_alpha, mean_sigma, mean_log_aps


#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def corr_alpha_sigma(mean_alpha, mean_sigma, alpha_samples, sigma_samples, 
                    weights):
    corr_as = 0.0
    m_a2 = 0.0
    m_s2 = 0.0
    m_as = 0.0
    for l in range(len(weights)):
        m_a2 += weights[l]*alpha_samples[l]*alpha_samples[l]
        m_s2 += weights[l]*sigma_samples[l]*sigma_samples[l]
        m_as += weights[l]*alpha_samples[l]*sigma_samples[l]

    corr_as = (m_as-mean_alpha*mean_sigma) \
            / np.sqrt((m_a2-mean_alpha**2)*(m_s2-mean_sigma**2))
    
    if corr_as < -1.0:
        corr_as = -1.0
    if corr_as > 1.0:
        corr_as = 1.0

    return corr_as


# Update the hyperparameters s1 and s2
#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def update_s_dp(log_one_minus_tau, s1, s2, k_trunc):
    """
    
    Parameters
    ----------
    log_one_minus_tau : TYPE
        E[log(1-tau)].
    s1, s2 : TYPE
        Gamma prior over alpha with two hyperparameters s1, s2: 
        p_alpha = Gamma(s1, s2).
    s2 : TYPE
        DESCRIPTION.
    k_trunc : int
        truncation values of nb clusters.

    Returns
    -------
    hat_s1, hat_s2 : alpha variational posterior, q_alpha = Gamma(hat_s1, hat_s2).

    """
    hat_s1 = s1 + k_trunc - 1.0
    aux = 0.0
    for l in range(k_trunc-1):
        aux += log_one_minus_tau[l]
    hat_s2 = s2 - aux

    return hat_s1, hat_s2


# Compute the mean-field-like Qz
#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def compute_q_z_tilde(beta_z, q_z, gamma_sb, nb_data, k_trunc, neighborhood):

    log_mrf = pairwise_energy(q_z, nb_data, k_trunc, beta_z, neighborhood)
    
    tau_tilde = np.zeros(k_trunc, dtype=np.float64)
    log_pi_tilde = np.zeros(k_trunc, dtype=np.float64)
    
    for k in range(k_trunc):
        tau_tilde[k] = gamma_sb[k, 0] / (gamma_sb[k, 0] + gamma_sb[k, 1])
        log_one_minus_tau_tilde = 0.0
        for l in range(k): 
            log_one_minus_tau_tilde += math.log(1.0-tau_tilde[l])
        log_pi_tilde[k] = math.log(tau_tilde[k]) + log_one_minus_tau_tilde
    
    q_z_tilde = np.zeros((nb_data, k_trunc), dtype=np.float64)
    for n in range(nb_data):
        for k in range(k_trunc):
            q_z_tilde[n, k] = log_pi_tilde[k] + log_mrf[n, k]

    return q_z_tilde


#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def gradient_beta(beta_z, q_z, gamma_sb, nb_data, k_trunc, neighborhood):
    
    q_z_tilde = compute_q_z_tilde(beta_z, q_z, gamma_sb, 
                                nb_data, k_trunc, neighborhood)
    
    # Mormalize q_z_tilde:
    q_z_tilde = normalize_exp_qz(q_z_tilde, nb_data, k_trunc)

    grad_interaction = 0.0
    grad_normalization = 0.0
    
    for n in range(nb_data):
        local_list = neighborhood[n, :]
        neighbors = local_list[local_list > -1]
        for k in range(k_trunc):
            aux1 = 0.0
            aux2 = 0.0
            for i in range(len(neighbors)):
                aux1 += q_z[neighbors[i], k]
                aux2 += q_z_tilde[neighbors[i], k]
            grad_interaction += aux1 * q_z[n, k]
            grad_normalization += aux2 * q_z_tilde[n, k]

    return (grad_interaction - grad_normalization) * 0.5
# to do: check if there is a factor 1/2


#@nb.jit(nopython=True, nogil=True, parallel=True)
def part1_gradient_beta(q_z, gamma_sb, nb_data, k_trunc, neighborhood):
    
    grad_interaction = np.float64(0.0)
    
    for n in nb.prange(nb_data):
        local_list = neighborhood[n, :]
        neighbors = local_list[local_list > -1]
        for k in nb.prange(k_trunc):
            aux1 = np.float64(0.0)
            for i in nb.prange(len(neighbors)):
                aux1 += q_z[neighbors[i], k]
            grad_interaction += aux1 * q_z[n, k]

    return grad_interaction 


#@nb.jit(nopython=True, nogil=True, parallel=True)
def part2_gradient_beta(beta_z, q_z, gamma_sb, nb_data, k_trunc, neighborhood):
    
    q_z_tilde = compute_q_z_tilde(beta_z, q_z, gamma_sb, 
                                nb_data, k_trunc, neighborhood)
    
    # Mormalize q_z_tilde:
    q_z_tilde = normalize_exp_qz(q_z_tilde, nb_data, k_trunc)

    grad_normalization = np.float64(0.0)
    
    for n in nb.prange(nb_data):
        local_list = neighborhood[n, :]
        neighbors = local_list[local_list > -1]
        for k in nb.prange(k_trunc):
            aux2 = np.float64(0.0)
            for i in nb.prange(len(neighbors)):
                aux2 += q_z_tilde[neighbors[i], k]
            grad_normalization += aux2 * q_z_tilde[n, k]

    return grad_normalization


def gradient_s1(s1, mean_alpha, mean_sigma, mean_log_aps):
    
    f = math.log(s1) - special.digamma(s1) \
        + mean_log_aps - math.log(mean_alpha+mean_sigma)

    return f


# Solve the equations for s1 and s2
def update_s_py(mean_alpha, mean_sigma, mean_log_aps, s1, s2):
    """
    VB-M-(s1, s2, a) step where we use for sigma a uniform prior so a does not have
    to taken into account.
    
    Parameters
    ----------
    mean_alpha, mean_sigma :
    s1, s2 : TYPE
        Gamma prior over alpha with two hyperparameters s1, s2: 
        p_alpha = Gamma(s1, s2).


    Returns
    -------
    hat_s1, hat_s2 = argmax_{s1,s2} E{q_{sigma,alpha}}{log p(alpha|sigma; s1,s2)}.

    """
    
    hat_s1 = 0.0
    hat_s2 = 0.0
    try:
        s1_ast = optimize.brentq(gradient_s1, 1e-10, 1e10, 
                                args=(mean_alpha, mean_sigma, mean_log_aps), 
                                maxiter=300, full_output=False)
        hat_s1 = s1_ast
    except (ValueError, RuntimeError):
        print("Warning: No solution for the M-s1 step.")
        hat_s1 = s1
    
    if mean_alpha + mean_sigma == 0.0:
        hat_s2 = s2
    else:
        hat_s2 = hat_s1 / (mean_alpha+mean_sigma)
    
    return hat_s1, hat_s2


def update_beta_z(q_z, gamma_sb, model, nb_data, 
                    k_trunc, beta_z_0, neighborhood):

    beta_z = 0.0
    if (model == 'dp-mrf') or (model == 'py-mrf'):
        try:
            beta_ast = optimize.brentq(gradient_beta, 0.0, 10.0, 
                                    args=(q_z, gamma_sb, nb_data, 
                                        k_trunc, neighborhood), 
                                        maxiter=300, full_output=False)
            beta_z = beta_ast
        except (ValueError, RuntimeError):
            print("Warning: No solution for the M-beta step.")
            beta_z = beta_z_0
    
    return beta_z

###############################################################################
#      Pairwise probability matrix for finding optimal clustering (attempts)
###############################################################################
# Memory problem since the matrix dimensions are too huge to handle
#@nb.jit(nopython=True, nogil=True, parallel=True)
def estimate_psm(q_z, nb_data, k_trunc):

    psmat = np.zeros((nb_data, nb_data), dtype=np.float64)
    for i in nb.prange(nb_data):
        for j in nb.prange(nb_data):
            if i == j:
                psmat[i, j] = 1.0
            else:
                for k in nb.prange(k_trunc):
                    psmat[i, j] += q_z[i, k] * q_z[j, k]
    
    return psmat


# Memory problem since the matrix dimensions are too huge to handle
#@nb.jit(nopython=True, nogil=True, parallel=True)
def indicator_mu(m_niw, k1, k2, threshold):
    diff = 0.0
    for d in nb.prange(m_niw.shape[1]):
        diff += (m_niw[k1, d] - m_niw[k2, d]) ** 2
    if np.sqrt(diff) < threshold:
        return 1.0
    else:
        return 0.0


#@nb.jit(nopython=True, nogil=True, parallel=True)
def estimate_psm_with_mu(q_z, m_niw, nb_data, k_trunc, threshold):

    psmat = np.zeros((nb_data, nb_data), dtype=np.float64)
    for i in nb.prange(nb_data):
        for j in nb.prange(nb_data):
            if i == j:
                psmat[i, j] = 1.0
            else:
                for k1 in nb.prange(k_trunc):
                    for k2 in nb.prange(k1+1, k_trunc):
                        if indicator_mu(m_niw, k1, k2, threshold) == 1.0:
                            psmat[i, j] += q_z[i, k1] * q_z[j, k2]
    
    return psmat

###############################################################################
#               Label initialization with Kmeans++
###############################################################################
'''
#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def init_lambda(data_X, q_z, nk, nb_data, dim_data_X, k_trunc):
    
    weighted_mean = compute_weighted_mean(q_z, data_X, nk, 
                                        nb_data, dim_data_X, k_trunc)
    weighted_cov = compute_weighted_cov(q_z, data_X, weighted_mean, 
                                        nb_data, dim_data_X, k_trunc)
    
    lambda_niw_0 = np.ones(k_trunc, dtype=np.float64)
    for k in range(k_trunc):
        # safe division:
        if nk[k] > 0.0:
            lambda_niw_0[k] = np.trace(weighted_cov[k]) / nk[k] / dim_data_X

    return lambda_niw_0
'''

#@nb.jit(nopython=True, nogil=True, fastmath=True, parallel=False)
def init_q_z(labels, nb_data, k_trunc):
    
    q_z = np.zeros((nb_data, k_trunc), dtype=np.float64)
    for i in range(nb_data):
        # q_z[i, np.random.choice(k_trunc, 1)] = 1.0
        q_z[i, labels[i]] = 1.0
    
    return q_z


def init_by_kmeans(data, nb_data, dim_data, k_trunc, nb_init, seed=100):
    """
    Initialize variational q_z by k-means and update normal inverse-Wishart
    parameters
    """
    m_niw_0 = np.zeros((k_trunc, dim_data), dtype=np.float64)
    psi_niw_0 = np.zeros((k_trunc, dim_data, dim_data), dtype=np.float64)
    psi_niw_0[:] = np.eye(dim_data, dtype=np.float64) * 1000.0
    nu_niw_0 = np.ones(k_trunc, dtype=np.float64) * dim_data
    lambda_niw_0 = np.ones(k_trunc, dtype=np.float64)
    
    import cv2
    # convert to np.float32
    arr = np.float32(data)
    # Define criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    # Use kmeans++ center initialization by Arthur 2007
    flags = cv2.KMEANS_PP_CENTERS
    # flags = cv2.KMEANS_RANDOM_CENTERS
    cv2.setRNGSeed(seed)
        
    _, labels, _ = cv2.kmeans(data=arr, K=k_trunc, bestLabels=None, 
                            criteria=criteria, attempts=nb_init, flags=flags)
    labels = labels.T[0]
    
    q_z = init_q_z(labels, nb_data, k_trunc)
    '''
    from sklearn.mixture import BayesianGaussianMixture    
    
    bgmm = BayesianGaussianMixture(n_components=k_trunc, init_params="kmeans", 
                                    random_state=0, n_init=nb_init).fit(data)
    
    q_z = bgmm.predict_proba(data)
    '''

    nk = q_z.sum(axis=0) + 10 * np.finfo(q_z.dtype).eps
    
    m_niw, psi_niw, lambda_niw, nu_niw = update_niw(data, q_z, nk,
                                                    m_niw_0, lambda_niw_0,
                                                    psi_niw_0, nu_niw_0,
                                                    nb_data, dim_data,
                                                    k_trunc)
    
    return m_niw, lambda_niw, psi_niw, nu_niw, q_z

# TestT
def init_by_gmm(data, nb_data, dim_data, data_X, dim_data_X, k_trunc, nb_init, seed=100):
    """
    Initialize variational q_z by k-means and update normal inverse-Wishart
    parameters
    """
    m_niw_0 = np.zeros((k_trunc, dim_data), dtype=np.float64)
    psi_niw_0 = np.zeros((k_trunc, dim_data, dim_data), dtype=np.float64)
    psi_niw_0[:] = np.eye(dim_data, dtype=np.float64) * 1000.0
    nu_niw_0 = np.ones(k_trunc, dtype=np.float64) * dim_data
    lambda_niw_0 = np.ones(k_trunc, dtype=np.float64)
    
    # import cv2
    # # convert to np.float32
    # arr = np.float32(data)
    # # Define criteria
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    # # Use kmeans++ center initialization by Arthur 2007
    # flags = cv2.KMEANS_PP_CENTERS
    # # flags = cv2.KMEANS_RANDOM_CENTERS
    # cv2.setRNGSeed(seed)
        
     # _, labels, _ = cv2.kmeans(data=arr, K=k_trunc, bestLabels=None, 
    #                         criteria=criteria, attempts=nb_init, flags=flags)
    # labels = labels.T[0]
    
    # q_z = init_q_z(labels, nb_data, k_trunc)
    
    from sklearn.mixture import GaussianMixture
    # data =  np.concatenate((data_X, data_Y), axis=1)
    gmm = GaussianMixture(n_components=k_trunc, random_state = seed).fit(data)##
    #gmm = GaussianMixture(random_state = seed).fit(data)
    labels = gmm.fit_predict(data)

    gmm.fit(data)
    q_z = gmm.predict_proba(data) 
    '''
    from sklearn.mixture import BayesianGaussianMixture    
    
    bgmm = BayesianGaussianMixture(n_components=k_trunc, init_params="kmeans", 
                                    random_state=0, n_init=nb_init).fit(data)
    
    q_z = bgmm.predict_proba(data)
    '''

    nk = q_z.sum(axis=0) + 10 * np.finfo(q_z.dtype).eps
    
    # m_niw, psi_niw, lambda_niw, nu_niw = update_niw(data, q_z, nk,
    #                                                 m_niw_0, lambda_niw_0,
    #                                                 psi_niw_0, nu_niw_0,
    #                                                 nb_data, dim_data,
    #                                                 k_trunc)
    m_niw, psi_niw, lambda_niw, nu_niw = update_niw(data[:, None, 0], q_z, nk,
                                                    m_niw_0, lambda_niw_0,
                                                    psi_niw_0, nu_niw_0,
                                                    nb_data, dim_data_X,
                                                    k_trunc)
    
    return m_niw, lambda_niw, psi_niw, nu_niw, q_z, labels

#######################################################################
## Conditional density estimation BNP-GLLiM by TrungTin Nguyen
#######################################################################

def Gaussian_ndata(data, mean, cov):
    """
    Calculate pdf of a Gaussian distribution over nb_data.

    Parameters
    ----------
    data : ([nb_data, dim_data] np.array)
        Input sample.
    mean : ([1, dim_data], np.array)
        Means of Gaussian.
    cov : ([dim_data, dim_data], np.array)
        Covariance matrices of Gaussian.
        
    Returns
    -------
    log_Ni : ([nb_data, 1], np.array)
        Log pdf of a Gaussian distribution.
    Ni : ([nb_data, 1], np.array)
        Pdf of a Gaussian distribution.
    """
    
    # ## Code from scratch  without using multivariate_normal.
    # (nb_data, dim_data) = np.shape(data)
    # det_cov = np.linalg.inv(cov)
    # z = ((data - np.ones((nb_data, 1))@mean)@det_cov)*(data - np.ones((nb_data, 1))@mean)
    # mahalanobis = np.sum(z, axis=1, keepdims=True)
    # log_Ni = -(dim_data/2)*np.log(2*np.pi) - 0.5*np.log(det_cov) - 0.5*mahalanobis
    # Ni = np.exp(log_Ni)
    
    (nb_data, dim_data) = np.shape(data)
    log_Ni = np.zeros((nb_data, 1))
    Ni = np.ones((nb_data, 1))
    
    for n in range(nb_data):
        log_Ni[n, :, None] = multivariate_normal.logpdf(data[n, :, None], mean, cov)
        Ni[n, :, None] = multivariate_normal.pdf(data[n, :, None], mean, cov)
    
    return log_Ni, Ni

def Student_ndata(data, mean, cov, dof):
    """
    Calculate pdf of a Gaussian distribution over nb_data.

    Parameters
    ----------
    data : ([nb_data, dim_data] np.array)
        Input sample.
    mean : ([1, dim_data], np.array)
        Means of Gaussian.
    cov : ([dim_data, dim_data], np.array)
        Covariance matrices of Gaussian.
        
    Returns
    -------
    log_Ni : ([nb_data, 1], np.array)
        Log pdf of a multivariate Student distribution.
    Ni : ([nb_data, 1], np.array)
        Pdf of a multivariate Student distribution.
    """
    
    # ## Code from scratch  without using multivariate_normal.
    # (nb_data, dim_data) = np.shape(data)
    # det_cov = np.linalg.inv(cov)
    # z = ((data - np.ones((nb_data, 1))@mean)@det_cov)*(data - np.ones((nb_data, 1))@mean)
    # mahalanobis = np.sum(z, axis=1, keepdims=True)
    # log_Ni = -(dim_data/2)*np.log(2*np.pi) - 0.5*np.log(det_cov) - 0.5*mahalanobis
    # Ni = np.exp(log_Ni)
    
    (nb_data, dim_data) = np.shape(data)
    log_Ni = np.zeros((nb_data, 1))
    Ni = np.ones((nb_data, 1))
 
    for n in range(nb_data):
        log_Ni[n, :, None] = multivariate_t.logpdf(data[n, :, None], mean, cov, dof)
        Ni[n, :, None] = multivariate_t.pdf(data[n, :, None], mean, cov, dof)
    
    return log_Ni, Ni


def Gaussian_weight_pdf(data, weight, mean, cov):
    """
    Calculate the log product between the weights and the PDF 
    of a Gaussian distribution.

    Parameters
    ----------
    data : ([nb_data, dim_data] np.array)
        Input sample.    
    weight : ([nb_mixture], np.array)
        Mixing proportion.
    mean : ([nb_mixture, dim_data], np.array)
        Means of Gaussian components.
    cov : ([nb_mixture, dim_data, dim_data], np.array)
        Covariance matrices of Gaussian components.
    Returns
    -------
    log_Nik : ([nb_data, nb_mixture], np.array)
        Log Pdf of a Gaussian distribution PDF.
        
    log_pik_Nik : ([nb_data, nb_mixture], np.array)
        Log of product between the weights 
        and the PDF of a Gaussian distribution.  
    """
    
    (nb_data, dim_data) = np.shape(data)
    nb_mixture = len(weight)
    log_Nik = np.zeros((nb_data, nb_mixture))
    log_pik_Nik = np.zeros((nb_data, nb_mixture))
    
    for k in range(nb_mixture):
        log_Nik[:, k, None] = Gaussian_ndata(data, mean[None,k,:], cov[k])[0]
        log_pik_Nik[:, k, None] = np.ones((nb_data, 1))* np.log(weight[k]) + log_Nik[:, k, None]
    
    return log_pik_Nik, log_Nik


def Student_weight_pdf(data, weight, mean, cov, dof):
    """
    Calculate the log product between the posterior weights and the PDF 
    of a multivariate Student's-t mixture model (SMM).

    Parameters
    ----------
    data : ([nb_data, dim_data] np.array)
        Input sample.    
    weight : ([nb_mixture], np.array)
        Mixing proportion.
    mean : ([nb_mixture, dim_data], np.array)
        Means of Gaussian components.
    cov : ([nb_mixture, dim_data, dim_data], np.array)
        Covariance matrices of Gaussian components.
    dof : ([nb_mixture], np.array)
        Degrees of freedom for each components.  
    Returns
    -------
    log_Nik : ([nb_data, nb_mixture], np.array)
        Log Pdf of a multivariate SMM.
        
    log_pik_Nik : ([nb_data, nb_mixture], np.array)
        Log of product between the weights 
        and the PDF of a multivariate SMM.  
    """
    
    (nb_data, dim_data) = np.shape(data)
    nb_mixture = len(weight)
    log_Nik = np.zeros((nb_data, nb_mixture))
    log_pik_Nik = np.zeros((nb_data, nb_mixture))
    
    for k in range(nb_mixture):
        log_Nik[:, k, None] = Student_ndata(data, mean[None,k,:], cov[k], dof[k])[0]
        log_pik_Nik[:, k, None] = np.ones((nb_data, 1))* np.log(weight[k]) + log_Nik[:, k, None]
    
    print('Nik_Student_ndata  = ',np.exp(log_Nik[0:50]))
    print('pik_Nik_Student_weight_pdf  = ',np.exp(log_pik_Nik[0:50]))
    return log_pik_Nik, log_Nik


def logsumexp(x, dimension):
    """
    Compute log(sum(exp(x),dim)) while avoiding numerical underflow.
    By default dim = 1 (row).

    Parameters
    ----------
    x : np.array
        Input data.
    dimension : int
        0: Column sum 1: Row sum.

    Returns
    -------
    log_sum_exp: np.float64
        Value of log(sum(exp(x),dim)).

    """
    # Subtract the largest value in each row
    x_max = np.amax(x, dimension, keepdims=True)
    x = x - x_max
    x_log_sum_exp = x_max + np.log(np.sum(np.exp(x), dimension, keepdims=True))
    
    x_max_check_inf = np.isinf(x_max)
    if np.sum(x_max_check_inf) > 0:
        x_log_sum_exp[x_max_check_inf] = x_max[x_max_check_inf]
    
    return x_log_sum_exp

def posterior_Student_gate(data, weight, mean, cov, dof):
    """
    Compute responsibilities in BNP-GLLiM general.

    Parameters
    ----------
    data : ([nb_data, dim_data] np.array)
        Input sample.    
    weight : ([nb_mixture], np.array)
        Mixing proportion.
    mean : ([nb_mixture, dim_data], np.array)
        Means of Gaussian components.
    cov : ([nb_mixture, dim_data, dim_data], np.array)
        Covariance matrices of Gaussian components.
    dof : ([nb_mixture], np.array)
        Degrees of freedom for all components.      
        
    Returns
    -------
    respons: ([nb_data, nb_mixture], np.array)
        Responsibilities.
    log_pik_Nik : ([nb_data, nb_mixture], np.array)
          Log of product between the weights and the PDF of a multivariate GMM.  
    """
    # loglik : np.float32
    #     Log-likelihood of GMM.    
    
    nb_mixture = len(weight)
    log_pik_Nik = Student_weight_pdf(data, weight, mean, cov, dof)[0]
    print("value Student_weight_pdf = ", log_pik_Nik[0:50])
    log_sum_exp_pik_Nik = logsumexp(log_pik_Nik, 1)
    log_responsik = log_pik_Nik - log_sum_exp_pik_Nik@np.ones((1, nb_mixture))
    respons = np.exp(log_responsik)
    print("value posterior_Student_gate = ", respons)

    # loglik = np.sum(log_sum_exp_pik_Nik)
    
    return respons, log_pik_Nik    

def posterior_Gaussian_gate(data, weight, mean, cov):
    """
    Compute responsibilities in a Gaussian Mixture Model.

    Parameters
    ----------
    data : ([nb_data, dim_data] np.array)
        Input sample.    
    weight : ([nb_mixture], np.array)
        Mixing proportion.
    mean : ([nb_mixture, dim_data], np.array)
        Means of Gaussian components.
    cov : ([nb_mixture, dim_data, dim_data], np.array)
        Covariance matrices of Gaussian components.
        
    Returns
    -------
    respons: ([nb_data, nb_mixture], np.array)
        Responsibilities.
    log_pik_Nik : ([nb_data, nb_mixture], np.array)
          Log of product between the weights and the PDF of a multivariate GMM.  
    """
    # loglik : np.float32
    #     Log-likelihood of GMM.    
    
    nb_mixture = len(weight)
    log_pik_Nik = Gaussian_weight_pdf(data, weight, mean, cov)[0]
    log_sum_exp_pik_Nik = logsumexp(log_pik_Nik, 1)
    log_responsik = log_pik_Nik - log_sum_exp_pik_Nik@np.ones((1, nb_mixture))
    respons = np.exp(log_responsik)
    # loglik = np.sum(log_sum_exp_pik_Nik)
    
    return respons, log_pik_Nik


def sample_GLLiM(pi_true, c_true, Gamma_true, A_true, b_true, Sigma_true, nb_data, seed):
    """
    Draw nb_data samples (Xi, Yi), i = 1,...,nb_data, from a supervised Gaussian locally-linear mapping (GLLiM).

    Parameters
    ----------
    pi_true : ([nb_mixture] np.array)
        Mixing proportion.
    c_true : ([nb_mixture, dim_data_X], np.array)
        Means of Gaussian components.
    Gamma_true : ([nb_mixture, dim_data_X, dim_data_X], np.array)
        Covariance matrices of Gaussian components.
    A_true : ([k_trunc, dim_data_Y, dim_data_X] np.array)
        Regressor matrix of location parameters of normal distribution.       
    b_true : ([k_trunc, dim_data_Y] np.array)
        Intercept vector of location parameters of normal distribution.       
    Sigma_true : ([k_trunc, dim_data_Y, dim_data_Y] np.array)
        Scale parameters (Gaussian experts' covariance matrices) of normal distribution.            
    nb_data : int
        Sample size.    
    seed : int
        Starting number for the random number generator.    
                    
    Returns
    -------
    data_X : ([nb_data, dim_data_X] np.array) 
        Input random sample.
    data_Y : ([nb_data, dim_data_Y] np.array)
        Output random sample.

    """
    ############################################    
    # Sample the input data: data_X
    ############################################   
     
    # Generate nb_data samples from Gaussian mixture model (GMM).

    # Draw nb_data samples from a multinomial distribution.
    rng = np.random.default_rng(seed)
    sample_multinomial = rng.multinomial(1, pi_true, size = nb_data)
    ## Test
    # N = 10
    # rvs = rng.multinomial(1, [0.2, 0.3, 0.5], size = N); rvs
    # Return the categories.
    kclass_X = sample_multinomial.argmax(axis = -1)
    #kclass

    # Draw nb_data samples from a multivariate normal distribution based on kclass.
    dim_data_X = c_true.shape[1]
    data_X = np.zeros((nb_data, dim_data_X))
    
    for n in range(nb_data):
        data_X[n, None] = rng.multivariate_normal(mean = c_true[kclass_X[n]], cov = Gamma_true[kclass_X[n]])
    
    # # Plot sample data_X. 
    # import matplotlib.pyplot as plt
    # plt.plot(data_X, data_X, 'x')
    # plt.axis('equal')
    # plt.show()

    ############################################    
    # Sample the output data: data_Y
    ############################################ 
    dim_data_Y = b_true.shape[1]
    data_Y = np.zeros((nb_data, dim_data_Y))

    # Calculate the gating network probabilites
    # Equivalent to calculation of a posteriori probas in a GMM.

    gating_prob = posterior_Gaussian_gate(data_X, pi_true, c_true, Gamma_true)[0]
    
    nb_mixture = len(pi_true)
    latent_Z =  np.zeros((nb_data, nb_mixture))
    kclass_Y =  np.zeros((nb_data, 1))
    
    for n in range(nb_data):
        Znk = rng.multinomial(1, gating_prob[n], size = 1)[0]
        latent_Z[n] = Znk
        zn = np.where(Znk == 1)[0]
        kclass_Y[n] = zn[0]
        # Sample Y
        data_Y[n, None] = b_true[zn] + (A_true[zn[0], :, :]@data_X[n, None].T).reshape(1, dim_data_Y)\
                            + rng.multivariate_normal(mean = np.zeros((dim_data_Y)),
                                                      cov = Sigma_true[zn[0], :, :])
        
    return data_X, data_Y, latent_Z, kclass_Y


# # TestT
# # initialization of A_hat, b_hat, Sigma_hat.
# def update_hyper_expert_init(data, q_z, nk, nb_data, dim_data, k_trunc,
#                         data_Y, dim_data_Y):
#     """
#     Updating hyperparameters (A, b, Sigma) using for VB-M-(A, b, Sigma) step.
#     Similar to M-mapping-step in GLLiM-EM.

#     Parameters
#     ----------
#     data : ([nb_data, dim_data] np.array)
#         Input sample.   
#     q_z : ([nb_data, k_trunc] np.array)
#         variational q_z distribution at previous iteration. 
#     nk : ([1,k_trunc] np.array)
#         Sums of variational posteriors of cluster variables z. 
#     nb_data : int
#         Sample size.
#     dim_data : int
#         Input data dimension.
#     k_trunc : int
#         truncation values of nb clusters.    
#     data_Y : ([nb_data_Y, dim_data_Y] np.array)
#         Output sample.    
#     dim_data_Y : int
#         Output data dimension.
#     """
    
#     # What are the old values of (A_hat, b_hat, Sigma_hat).
#     # A_hat = self.hyper_expert["A_hat"] 
#     # b_hat = self.hyper_expert["b_hat"] 
#     # Sigma_hat = self.hyper_expert["Sigma_hat"] 
    
#     # m_niw_0 = self.rho["m"]
#     # lambda_niw_0 = self.rho["lambda"]
#     # psi_niw_0 = self.rho["psi"]
#     # nu_niw_0  = self.rho["nu"]
    
#     weighted_mean_X = compute_weighted_mean(q_z, data, nk, nb_data, 
#                                           dim_data, k_trunc)
#     weighted_mean_Y = compute_weighted_mean(q_z, data_Y, nk, nb_data, 
#                                           dim_data_Y, k_trunc)
#     normalizing_weighted_input = compute_normalizing_weighted_input(q_z, 
#                                     data, nk, nb_data, dim_data, k_trunc,
#                                     weighted_mean_X)
#     normalizing_weighted_output = compute_normalizing_weighted_output(q_z, 
#                                       data_Y, nk, nb_data, dim_data_Y, k_trunc,                        
#                                               weighted_mean_Y)            
#     A_hat, b_hat, Sigma_hat = compute_abs(q_z, data, data_Y, nk, normalizing_weighted_input,
#                 normalizing_weighted_output, dim_data, dim_data_Y, k_trunc, nb_data)
  
#     return A_hat, b_hat, Sigma_hat


import os
import matplotlib.pyplot as plt

from inspect import getsourcefile
from os.path import abspath
#https://stackoverflow.com/questions/2632199/how-do-i-get-the-path-of-the-current-executed-file-in-python?lq=1
def get_current_path():
    current_path = abspath(getsourcefile(lambda:0)) # fullname of current file
    #current_path = os.path.dirname(__file__)
    current_dir = os.path.dirname(current_path)
    return current_dir

def save_fig(fname, *args, **kwargs):
    #figdir = '../figures' # default directory one above where code lives
    current_dir = get_current_path()
    figdir = os.path.join(current_dir, "..", "figures")

    if not os.path.exists(figdir):
        print('making directory {}'.format(figdir))
        os.mkdir(figdir)

    fname_full = os.path.join(figdir, fname)
    print('saving image to {}'.format(fname_full))
    #plt.tight_layout()

    # use TrueType fonts so they are embedded
    # https://stackoverflow.com/questions/9054884/how-to-embed-fonts-in-pdfs-produced-by-matplotlib
    # https://jdhao.github.io/2018/01/18/mpl-plotting-notes-201801/
    plt.rcParams['pdf.fonttype'] = 42

    # Font sizes
    SIZE_SMALL = 12
    #SIZE_MEDIUM = 14
    SIZE_LARGE = 24
    # https://stackoverflow.com/a/39566040
    plt.rc('font', size=SIZE_SMALL)  # controls default text sizes
    plt.rc('axes', titlesize=SIZE_SMALL)  # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE_SMALL)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SIZE_SMALL)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE_SMALL)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE_SMALL)  # legend fontsize
    plt.rc('figure', titlesize=SIZE_LARGE)  # fontsize of the figure title

    plt.savefig(fname_full, *args, **kwargs)
    
    
def savefig(fname, *args, **kwargs):
    save_fig(fname, *args, **kwargs)