# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '/mnt/zfsusers/jaimerz/PhD/xCell')

from xcell.mappers import MapperSPT

from xcell.mappers import utils
from astropy.io import fits
from astropy.table import Table
import pyccl as ccl
import numpy as np
import pylab as plt
import pymaster as nmt
import healpy as hp
import numpy as np

nside = 4096
bands = nmt.NmtBin(nside, nlb=60)
ell_arr = bands.get_effective_ells()


path_SPT = '/mnt/zfsusers/jaimerz/PhD/xCell/data/SPT/sptsz_planck_ymap_healpix/'
SPT_c =  {'file_map': path_SPT+'SPTSZ_Planck_min_variance_ymap.fits', 
          'file_hm1': path_SPT+'SPTSZ_Planck_min_variance_ymap_half1.fits',
          'file_hm2': path_SPT+'SPTSZ_Planck_min_variance_ymap_half2.fits',
          'file_mask':None,
          'file_gp_mask': path_SPT+'SPTSZ_dust_mask_top_5percent.fits',
          'file_ps_mask':path_SPT+'SPTSZ_point_source_mask_nside_8192_binary_mask.fits',
          'mask_name': 'mask_SPT',
          'nside':nside}
#####
SPT_mapper = MapperSPT(SPT_c)
SPT_full = SPT_mapper.get_signal_map()
SPT_full_f = SPT_mapper.get_nmt_field()
SPT_hm1, SPT_hm2 = SPT_mapper._get_hm_maps()
SPT_hm1_f = SPT_mapper._get_nmt_field(signal=SPT_hm1)
SPT_hm2_f = SPT_mapper._get_nmt_field(signal=SPT_hm2)
SPT_diff = SPT_mapper._get_diff_map()
SPT_diff_f = SPT_mapper.get_nmt_field(signal=SPT_diff)
SPT_mask= SPT_mapper.get_mask()

#####
SPT_ws = nmt.NmtWorkspace()
SPT_ws.compute_coupling_matrix(SPT_full_f, SPT_full_f, bands)

# Noise
SPT_nl_c = SPT_mapper.get_nl_coupled()
SPT_nl_dc = SPT_ws.decouple_cell(SPT_nl_c)
SPT_cl_full_c = nmt.compute_coupled_cell(SPT_full_f, SPT_full_f)
SPT_cl_full_dc = SPT_ws.decouple_cell(SPT_cl_full_c) 

# Already without noise
SPT_cl_fl_c = nmt.compute_coupled_cell(SPT_hm1_f, SPT_hm2_f)
SPT_cl_fl_dc = SPT_ws.decouple_cell(SPT_cl_fl_c) 

####
SPT_dl_full_dc = SPT_cl_full_dc*(10**12*ell_arr*(ell_arr+1))/(2*np.pi)
SPT_dl_fl_dc = SPT_cl_fl_dc*(10**12*ell_arr*(ell_arr+1))/(2*np.pi)
SPT_dnl_dc = SPT_nl_dc*(10**12*ell_arr*(ell_arr+1))/(2*np.pi)


np.savez('SPT_{}.npz'.format(nside),
        ell_arr = ell_arr,
        SPT_nl_c = SPT_nl_c,
        SPT_nl_dc = SPT_nl_dc,
        SPT_cl_full_c = SPT_cl_full_c,
        SPT_cl_full_dc = SPT_cl_full_dc,
        SPT_cl_fl_c = SPT_cl_fl_c,
        SPT_cl_fl_dc = SPT_cl_fl_dc,
        CIB_dl_fl_dc = CIB_dl_fl_dc,
        SPT_dnl_dc = SPT_dnl_dc,
        SPT_dl_full_dc = SPT_dl_full_dc,
        SPT_dl_fl_dc = SPT_dl_fl_dc)