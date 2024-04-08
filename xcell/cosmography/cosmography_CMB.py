from .cosmography_base import CosmographyBase
import numpy as np

class CosmograghyCMB(CosmographyBase)
    self.data_name = 'CMB'
    def __init__(self):
        self.z = np.array([1089.95]) 
        self.data = np.array([1.04109])
        self.cov = np.array([[0.00030**2]])

    def get_redshift(self):
        return self.z
    
    def get_data(self):
        return self.data
    
    def get_cov(self):
        return self.cov
    
        
