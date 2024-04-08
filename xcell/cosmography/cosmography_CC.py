from .cosmography_base import CosmographyBase
import numpy as np

class CosmograghyCC(CosmographyCC)
    self.data_name = 'CC'
    def __init__(self):
        self.z = np.array([0.07  , 0.09  , 0.12  , 0.17  , 0.179 , 0.199 , 0.2   ,
                   0.27  , 0.28  , 0.352 , 0.38  , 0.3802, 0.4   , 0.4004, 0.4247,
                   0.44  , 0.4497, 0.47  , 0.4783, 0.48  , 0.51  , 0.593 , 0.6   ,
                   0.61  , 0.68  , 0.73  , 0.781 , 0.875 , 0.88  , 0.9   , 1.037 ,
                   1.3   , 1.363 , 1.43  , 1.53  , 1.75  , 1.965])
        self.data =  np.array([69. ,  69. ,  68.6,  83. ,  75. ,  75. ,  72.9,  77. ,
                    88.8,  83. ,  81.5,  83. ,  95. ,  77. ,  87.1,  82.6,  92.8,
                    89. ,  80.9,  97. ,  90.4, 104. ,  87.9,  97.3,  92. ,  97.3,
                   105. , 125. ,  90. , 117. , 154. , 168. , 160. , 177. , 140. ,
                   202. , 186.5])
        err = np.array([19.6,  12. ,  26.2,   8. ,   4. ,   5. ,  29.6,  14. ,
                    36.6,  14. ,   1.9,  13.5,  17. ,  10.2,  11.2,   7.8,  12.9,
                    23. ,   9. ,  62. ,   1.9,  13. ,   6.1,   2.1,   8. ,   7. ,
                    12. ,  17. ,  40. ,  23. ,  20. ,  17. ,  33.6,  18. ,  14. ,
                    40. ,  50.4])
        cov = np.zeros([len(self.z),len(self.z)])
        for i in np.arange(len(self.z)):
            cov[i,i] = err[i]**2
        self.cov = cov

    def get_redshift(self):
        return self.z
    
    def get_data(self):
        return self.data
    
    def get_cov(self):
        return self.cov
    
        
