import sys
sys.path.insert(1, '/mnt/zfsusers/jaimerz/PhD/xCell')
from xcell.mappers import MapperDELS
from xcell.mappers import MapperCatWISE
from xcell.mappers import MapperNVSS
from xcell.mappers import utils
from astropy.io import fits
from astropy.table import Table
import pyccl as ccl
import numpy as np
import pylab as plt
import pymaster as nmt
import healpy as hp
import os

nside = 1024
#data_WIxSC = '/mnt/extraspace/damonge/Data/2MPZ_WIxSC/'
#WIxSC_config = {'data_catalog': data_WIxSC+'WIxSC.fits',
#               'mask': data_WIxSC+'WISExSCOSmask_galactic.fits.gz',
#               'star_map': data_WIxSC+'stars.fits',
#               'spec_sample': data_WIxSC+'zSpec-comp-WIxSC.csv',
#               'bin_name': '0',
#               'coordinates': 'C',
#               'z_edges': [0, 0.5],
#               'n_jk_dir': 100,
#                'nside': nside,
#               'mask_name': 'mask_WIxSC'}
data_DELS = '/mnt/extraspace/damonge/Datasets/DELS/'
DELS_config = {'data_catalogs':[data_DELS+'Legacy_Survey_BASS-MZLS_galaxies-selection.fits'],
               'zbin': 1,
               'z_name': 'PHOTOZ_3DINFER',
               'num_z_bins': 500,
               'binary_mask': data_DELS+'Legacy_footprint_final_mask.fits',
               'completeness_map': data_DELS+'Legacy_footprint_completeness_mask_128.fits',
               'star_map': data_DELS+'allwise_total_rot_1024.fits',
               'nside': nside,
               'mask_name': 'mask_DELS'}
data_CATWISE = '/mnt/extraspace/damonge/Datasets/CatWISE/'
CATWISE_config = {'data_catalog': data_CATWISE+'catwise_agns_masked_final_w1lt16p5_alpha.fits',
                   'mask': 'xcell/tests/data/MASKS_exclude_master_final.fits',
                   'mask_name': 'mask_CatWISE',
                   'nside': nside}
data_NVSS = '/mnt/extraspace/damonge/Datasets/NVSS/'
NVSS_config = {'data_catalog': data_NVSS+'nvss.fits',
               'mask': data_CATWISE+'mask.fits',
               'mask_name': 'mask_NVSS',
               'redshift_catalog':data_CATWISE+'100sqdeg_1uJy_s1400.fits',
               'nside': nside}

WIxSC = MapperWIxSC(WIxSC_config)
DELS = MapperDELS(DELS_config)
NVSS = MapperNVSS(NVSS_config)
CATWISE = MapperCatWISE(CATWISE_config)

DELS_cat = DELS.get_catalog()
WIxSC_cat = WIxSC.get_catalog()bands = nmt.NmtBin(nside, nlb=60)
ell_arr = bands.get_effective_ells()

DELS_f = DELS.get_nmt_field()
NVSS_f = NVSS.get_nmt_field()
CATWISE_f = CATWISE.get_nmt_field()

DELSxNVSS_wsp = nmt.NmtWorkspace()
DELSxCATWISE_wsp = nmt.NmtWorkspace()
NVSSxCATWISE_wsp = nmt.NmtWorkspace()
DELSxNVSS_wsp.compute_coupling_matrix(DELS_f, NVSS_f, bands)
DELSxCATWISE_wsp.compute_coupling_matrix(DELS_f, CATWISE_f, bands)
NVSSxCATWISE_wsp.compute_coupling_matrix(NVSS_f, CATWISE_f, bands)

DELSxNVSS_cl_coupled = nmt.compute_coupled_cell(DELS_f, NVSS_f)
DELSxCATWISE_cl_coupled = nmt.compute_coupled_cell(DELS_f, CATWISE_f)
NVSSxCATWIS_cl_coupled = nmt.compute_coupled_cell(NVSS_f, CATWISE_f)

DELSxNVSS_cl = DELSxNVSS_wsp.decouple_cell(DELSxNVSS_cl_coupled)
DELSxCATWISE_cl = DELSxNVSS_wsp.decouple_cell(DELSxCATWISE_cl_coupled)
NVSSxCATWISE_cl = DELSxNVSS_wsp.decouple_cell(NVSSxCATWIS_cl_coupled)

np.savez('DELSxNVSS_{}.npz'.format(nside),
         ell_arr = ell_arr,
         DELSxNVSS_cl_coupled = DELSxNVSS_cl_coupled,
         DELSxNVSS_cl = DELSxNVSS_cl)

np.savez('DELSxCATWISE_{}.npz'.format(nside),
         ell_arr = ell_arr,
         DELSxCATWISE_cl_coupled = DELSxCATWISE_cl_coupled,
         DELSxCATWISE_cl = DELSxCATWISE_cl)

np.savez('NVSSxCATWISE_{}.npz'.format(nside),
         ell_arr = ell_arr,
         NVSSxCATWISE_cl_coupled = NVSSxCATWISE_cl_coupled,
         NVSSxCATWISE_cl = NVSSxCATWISE_cl)