from mapper_base import MapperBase
from astropy.io import fits
from astropy.table import Table

import mapper_DESwl
import pyccl as ccl
import numpy as np
import pylab as plt
import pymaster as nmt
import os

nside = 512

path_data = '/mnt/extraspace/damonge/S8z_data/DES_data/shear_catalog/'

config =  {'zbin_cat': path_data + 'y1_source_redshift_binning_v1.fits', 
        'data_cat':  path_data + 'mcal-y1a1-combined-riz-unblind-v4-matched.fits',
         'file_nz': path_data + 'y1_redshift_distributions_v1.fits',
         'nside': nside,
         'bin': bin,
         'mask_name': 'name'}

mapper_DESwl.MapperDESwl(config)


            
        
        
