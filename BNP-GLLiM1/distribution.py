#####################################################################################
#
#####################################################################################


class Distribution():
    """ """

    # Class attributes
        
        
    def __init__(self):
        self.var_name = ""
        self.parameter = {}
        self.prior = {}
        self.post_param = {}
                        
    def __str__(self):
        return self.var_name +" distribution" + "\nparameters:" + str(self.param) \
                + "\n prior parameters: " + str(self.rho) \
                + "\n posterior distibution: " + str(self.post_param)
        
    """
    to be implemented by children classes
    """
    def expectation_likelihood(self, data, nb_data, dim_data, k_trunc):
        print("not implemented yet")
        
    def up_date_posterior(self, data, q_z, nk, nb_data, dim_data, k_trunc):
        print("not implemented yet")
        
    def up_date_prior(self, data, q_z, nk, nb_data, dim_data, k_trunc):
        print("not implemented yet")


   
