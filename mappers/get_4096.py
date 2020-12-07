from mapper_base import MapperBase
from astropy.io import fits
from astropy.table import Table

import mapper_DES
import pyccl as ccl
import numpy as np
import pylab as plt
import pymaster as nmt
import os

nside = 4096

ells = []
i = 11.5
while i <= 3 * nside:
    ells.append(round(i))
    i = i+20*(1+i/500)
bands = nmt.NmtBin.from_edges(ells[:-1], ells[1:])
ell_arr = bands.get_effective_ells()

path_data = '/mnt/extraspace/damonge/S8z_data/DES_data/'

config1 = {'data_catalogs':path_data + 'redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_zerr_CATALOG.fits', 
                'file_mask':path_data + 'redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_MASK_HPIX4096RING.fits',
                'file_nz':path_data + 'data_vector/2pt_NG_mcal_1110.fits',
          'bin':1,
          'nside':nside }

config2 = {'data_catalogs':path_data + 'redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_zerr_CATALOG.fits', 
                'file_mask':path_data + 'redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_MASK_HPIX4096RING.fits',
                'file_nz':path_data + 'data_vector/2pt_NG_mcal_1110.fits',
          'bin':2,
          'nside':nside }

config3 = {'data_catalogs':path_data + 'redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_zerr_CATALOG.fits', 
                'file_mask':path_data + 'redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_MASK_HPIX4096RING.fits',
                'file_nz':path_data + 'data_vector/2pt_NG_mcal_1110.fits',
          'bin':3,
          'nside':nside}

config4 = {'data_catalogs':path_data + 'redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_zerr_CATALOG.fits', 
                'file_mask':path_data + 'redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_MASK_HPIX4096RING.fits',
                'file_nz':path_data + 'data_vector/2pt_NG_mcal_1110.fits',
          'bin':4,
          'nside':nside }

config5 = {'data_catalogs':path_data + 'redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_zerr_CATALOG.fits', 
                'file_mask':path_data + 'redmagic_catalog/DES_Y1A1_3x2pt_redMaGiC_MASK_HPIX4096RING.fits',
                'file_nz':path_data + 'data_vector/2pt_NG_mcal_1110.fits',
          'bin':5,
          'nside':nside }
          

D0 = mapper_DES.MapperDES(config1)
D1 = mapper_DES.MapperDES(config2)
D2 = mapper_DES.MapperDES(config3)
D3 = mapper_DES.MapperDES(config4)
D4 = mapper_DES.MapperDES(config5)

fields = {
       'D_f_0':  D0.get_nmt_field(),
       'D_f_1':  D1.get_nmt_field(),
       'D_f_2':  D2.get_nmt_field(),
       'D_f_3':  D3.get_nmt_field(),
       'D_f_4':  D4.get_nmt_field()}

nls = {
    'nl_0': D0.get_nl_coupled(),
    'nl_1': D1.get_nl_coupled(),
    'nl_2': D2.get_nl_coupled(),
    'nl_3': D3.get_nl_coupled(),
    'nl_4': D4.get_nl_coupled()}

for i in range(5):
    for j in range(5):
        wsp = nmt.NmtWorkspace()
        wsp.compute_coupling_matrix(fields['D_f_{}'.format(i)], fields['D_f_{}'.format(j)], bands)
        
        cl_coupled = nmt.compute_coupled_cell(fields['D_f_{}'.format(i)], fields['D_f_{}'.format(j)])
        cl_decoupled = wsp.decouple_cell(cl_coupled)
        
        if i == j:
            nl = wsp.decouple_cell(nls['nl_{}'.format(i)])
            cl_decoupled = cl_decoupled - nl
        
        np.savetxt('DES_cl_{}{}.txt'.format(i,j), cl_decoupled)
            
        
        
