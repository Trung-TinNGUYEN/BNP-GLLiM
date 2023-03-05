import numpy as np
import scipy.optimize as optimize 
from vbem_utils_numba import  gradient_beta, part1_gradient_beta, part2_gradient_beta 
             

def update_beta_z(q_z, gamma_sb, model, sampling_model, nb_data, 
                    k_trunc, beta_z_0, neighborhood, iters):

    #draw_gradient_beta(-10., 10., 1000, q_z, gamma_sb, nb_data, k_trunc,
    #                       beta_z_0, neighborhood, iters)
    
    if sampling_model == "gaussian":
        
        beta_z = update_beta_z_gaus(q_z, gamma_sb, model, nb_data,
                                    k_trunc, beta_z_0, neighborhood)
    elif sampling_model == "poisson" or sampling_model == "binomial":

        beta_z =  update_beta_z_disease_mapping(q_z, gamma_sb, model, nb_data,
                            k_trunc, beta_z_0, neighborhood)
    else:
        print("Warning: Please check the model name.")
        raise SystemExit

    return beta_z


def update_beta_z_gaus(q_z, gamma_sb, model, nb_data, 
                    k_trunc, beta_z_0, neighborhood):

    beta_z = 0.0
    if (model == 'dp-mrf') or (model == 'pyp-mrf'):
        try:
            beta_ast = optimize.brentq(gradient_beta, -10.0, 10.0,
                                args=(q_z, gamma_sb, nb_data, 
                                    k_trunc, neighborhood), 
                                    maxiter=200, full_output=False)
            beta_z = beta_ast
        except:
            print("Warning: No solution for the M-beta step.")
            beta_z = beta_z_0

    return beta_z


# Optimize beta in VB-M-beta step through gradient descente algorithm
def update_beta_z_disease_mapping(q_z, gamma_sb, model, nb_data, k_trunc, beta_z_0, 
                                                   neighborhood):
    """
    Optimize beta in VB-M-beta step through gradient descente algorithm.

    """
    if (model == 'dp-mrf') or (model == 'pyp-mrf'):
        cur_b = beta_z_0
        gf = 1e100
        # Learning rate
        rate = 1e-4
        precision = 1e-15
        delta_b = 1.
        max_iters = 1e5
        iters = 0 
    
        while delta_b > precision and abs(gf) > 1e-6 and iters < max_iters:
            
            gf = gradient_beta(cur_b, q_z, gamma_sb, nb_data, k_trunc, neighborhood)
            assert(not np.isnan(gf))
            #Gradient descent
            prev_b = cur_b
            cur_b = cur_b + rate * gf  
        
            delta_b = abs(cur_b - prev_b)
            
            iters = iters + 1
            
        gf = gradient_beta(cur_b, q_z, gamma_sb, nb_data, k_trunc, neighborhood)   
        print("nb_iters={} beta={} gradient_beta={}".format(iters, cur_b, gf))
        
        if abs(gf) < 1e-3:
            return cur_b 
        else:
            return beta_z_0

    else:
        return 0.0


def draw_gradient_beta(a, b, n, q_z, gamma_sb, nb_data, k_trunc, 
                           beta_z_0, neighborhood, iters):

    import matplotlib.pyplot as plt    
        
    betas = np.arange(a, b, (b-a)/n)
    g2_beta = np.zeros(betas.shape[0], dtype=np.float64)
    
    g1_beta = part1_gradient_beta(q_z, gamma_sb, nb_data, k_trunc, neighborhood)*0.5
    
    for i,e in enumerate(betas):
        g2_beta[i] = part2_gradient_beta(e, q_z, gamma_sb, nb_data, k_trunc, neighborhood)*0.5 
    
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    fig.suptitle('part2_gradient_beta part1={}'.format(g1_beta), fontsize=16)
    ax.plot(betas, g2_beta, "--", betas, np.repeat(g1_beta, betas.shape[0]), "r--")
    ax.plot([0])
    plt.savefig("part2_fig_{}".format(iters))
    plt.close(fig) 
    
