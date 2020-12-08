from mapper_base import MapperBase
from astropy.io import fits
from astropy.table import Table


import mapper_KV450
import pyccl as ccl
import numpy as np
import pylab as plt
import pymaster as nmt
import os

nside = 4096

ells = [0, 30, 60, 90, 120, 150, 180, 210, 240, 272, 309, 351, 398, 452, 513, 582, 661, 750, 852, 967, 1098, 1247, 1416, 1608, 1826, 2073, 2354, 2673, 3035, 3446, 3914, 4444, 5047, 5731, 6508, 7390, 8392, 9529, 10821, 12288]
bands = nmt.NmtBin.from_edges(ells[:-1], ells[1:])
ell_arr = bands.get_effective_ells()

path_data = '/mnt/extraspace/damonge/S8z_data/DES_data/'
cats = [path+'KV450_G12_reweight_3x4x4_v2_good.cat', 
        path+'KV450_G23_reweight_3x4x4_v2_good.cat',
        path+'KV450_GS_reweight_3x4x4_v2_good.cat',
        path+'KV450_G15_reweight_3x4x4_v2_good.cat',
        path+'KV450_G9_reweight_3x4x4_v2_good.cat'
       ]

config1 = {'data_catalogs': cats , 
          'file_nz':path + 'REDSHIFT_DISTRIBUTIONS/Nz_DIR/Nz_DIR_Mean/Nz_DIR_z0.1t0.3.asc',
          'zbin':1,
          'nside':nside}

config2 = {'data_catalogs': cats , 
          'file_nz':path + 'REDSHIFT_DISTRIBUTIONS/Nz_DIR/Nz_DIR_Mean/Nz_DIR_z0.1t0.3.asc',
          'zbin':2,
          'nside':nside}

config3 = {'data_catalogs': cats , 
          'file_nz':path + 'REDSHIFT_DISTRIBUTIONS/Nz_DIR/Nz_DIR_Mean/Nz_DIR_z0.1t0.3.asc',
          'zbin':3,
          'nside':nside}

config4 = {'data_catalogs': cats , 
          'file_nz':path + 'REDSHIFT_DISTRIBUTIONS/Nz_DIR/Nz_DIR_Mean/Nz_DIR_z0.1t0.3.asc',
          'zbin':4,
          'nside':nside}

config5 = {'data_catalogs': cats , 
          'file_nz':path + 'REDSHIFT_DISTRIBUTIONS/Nz_DIR/Nz_DIR_Mean/Nz_DIR_z0.1t0.3.asc',
          'zbin':5,
          'nside':nside}

K0 = mapper_KV450.MapperKV450(config1)
K1 = mapper_KV450.MapperKV450(config2)
K2 = mapper_KV450.MapperKV450(config3)
K3 = mapper_KV450.MapperKV450(config4)
K4 = mapper_KV450.MapperKV450(config5)

fields = {
       'K_f_0':  K0.get_nmt_field(),
       'K_f_1':  K1.get_nmt_field(),
       'K_f_2':  K2.get_nmt_field(),
       'K_f_3':  K3.get_nmt_field(),
       'K_f_4':  K4.get_nmt_field()}

nls = {
    'nl_0': K0.get_nl_coupled(),
    'nl_1': K1.get_nl_coupled(),
    'nl_2': K2.get_nl_coupled(),
    'nl_3': K3.get_nl_coupled(),
    'nl_4': K4.get_nl_coupled()}

for i in range(5):
    for j in range(5):
        wsp = nmt.NmtWorkspace()
        wsp.compute_coupling_matrix(fields['K_f_{}'.format(i)], fields['K_f_{}'.format(j)], bands)
        
        cl_coupled = nmt.compute_coupled_cell(fields['K_f_{}'.format(i)], fields['K_f_{}'.format(j)])
        cl_decoupled = wsp.decouple_cell(cl_coupled)
        
        if i == j:
            nl_decoupled = wsp.decouple_cell(nls['nl_{}'.format(i)])
            np.savetxt('KV450_nl_{}{}.txt'.format(i,j), nl_decoupled)
        
        np.savetxt('KV450_cl_{}{}.txt'.format(i,j), cl_decoupled)
            
        
        
