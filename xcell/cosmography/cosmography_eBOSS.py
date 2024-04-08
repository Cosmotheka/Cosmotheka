from .cosmography_base import CosmographyBase
import numpy as np

class CosmograghyeBOSS(CosmographyBase)
    self.data_name = 'eBOSS'
    def __init__(self):
        self.rd = 147.3
        self.z = np.array([1.48])
        para = 13.23
        perp = 30.21
        fs8 = 0.462
        self.data =  np.array([para, perp, fs8])
        self.cov = np.array([[0.7709, -0.05656, 0.01750],
                             [-0.05656, 0.2640, -0.006204],
                             [0.01750, -0.006204, 0.002308]])

    def get_redshift(self):
        return self.z
    
    def get_data(self):
        return self.data
    
    def get_cov(self):
        return self.cov
    
    def get_rd(self):
    """
    Returns the sound horizon at the drag epoch
    Returns:
        rd (float): sound horizon at the drag epoch
    """
        return self.rd
    
        
