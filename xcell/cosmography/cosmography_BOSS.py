from .cosmography_base import CosmographyBase
import numpy as np

class CosmograghyBOSS(CosmographyBase)
    self.data_name = 'BOSS'
    def __init__(self):
        self.rd = 147.78
        self.z = np.array([0.38, 0.51, 0.61])
        perp = np.array([1512.39, 1975.22, 2306.68])
        para = np.array([81.2087, 90.9029, 98.9647])
        fs8 = np.array([0.49749, 0.457523, 0.436148])
        self.data = np.concatenate([para, perp, fs8])
        self.cov = np.array([
                   [3.63049e+00, 1.80306e+00, 9.19842e-01, 9.71342e+00, 7.75546e+00,
                    5.97897e+00, 2.79185e-02, 1.24050e-02, 4.75548e-03],
                   [1.80306e+00, 3.77146e+00, 2.21471e+00, 4.85105e+00, 1.19729e+01,
                    9.73184e+00, 9.28354e-03, 2.22588e-02, 1.05956e-02],
                   [9.19842e-01, 2.21471e+00, 4.37982e+00, 2.43394e+00, 6.71715e+00,
                    1.60709e+01, 1.01870e-03, 9.71991e-03, 2.14133e-02],
                   [9.71342e+00, 4.85105e+00, 2.43394e+00, 5.00049e+02, 2.94536e+02,
                    1.42011e+02, 3.91498e-01, 1.51597e-01, 4.36366e-02],
                   [7.75546e+00, 1.19729e+01, 6.71715e+00, 2.94536e+02, 7.02299e+02,
                    4.32750e+02, 1.95890e-01, 3.88996e-01, 1.81786e-01],
                   [5.97897e+00, 9.73184e+00, 1.60709e+01, 1.42011e+02, 4.32750e+02,
                    1.01718e+03, 3.40803e-02, 2.46111e-01, 4.78570e-01],
                   [2.79185e-02, 9.28354e-03, 1.01870e-03, 3.91498e-01, 1.95890e-01,
                    3.40803e-02, 2.03355e-03, 8.11829e-04, 2.64615e-04],
                   [1.24050e-02, 2.22588e-02, 9.71991e-03, 1.51597e-01, 3.88996e-01,
                    2.46111e-01, 8.11829e-04, 1.42289e-03, 6.62824e-04],
                   [4.75548e-03, 1.05956e-02, 2.14133e-02, 4.36366e-02, 1.81786e-01,
                    4.78570e-01, 2.64615e-04, 6.62824e-04, 1.18576e-03]])

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
        
