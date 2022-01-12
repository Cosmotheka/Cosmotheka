# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '/mnt/zfsusers/jaimerz/PhD/xCell')

from xcell.mappers import MapperP15tSZ
from xcell.mappers import MapperP15CIB
from xcell.mappers import MapperLenzCIB
from xcell.mappers import MapperP18SMICA
from xcell.mappers import MapperP18CMBK

from xcell.mappers import utils
from astropy.io import fits
from astropy.table import Table
import pyccl as ccl
import numpy as np
import pylab as plt
import pymaster as nmt
import healpy as hp
import numpy as np

nside = 2048
bands = nmt.NmtBin(nside, nlb=60)
ell_arr = bands.get_effective_ells()

path_CIB = '/mnt/zfsusers/jaimerz/PhD/xCell/data/P18/CIB/'
path_CIB_Lenz = '/mnt/zfsusers/jaimerz/PhD/xCell/data/P18/CIB_Lenz/'
CIB_c = {'file_map': path_CIB+'COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits',
         'file_hm1': path_CIB+'COM_CMB_IQU-nilc_2048_R3.00_hm1.fits', 
         'file_hm2': path_CIB+'COM_CMB_IQU-nilc_2048_R3.00_hm2.fits',
         'file_mask':  path_CIB_Lenz+'mask_apod.hpx.fits', #None,
         'file_gp_mask': None, #path_SMICA+'HFI_Mask_GalPlane-apo2_2048_R2.00.fits',
         'file_sp_mask': None, #path_SMICA+'HFI_Mask_PointSrc_2048_R2.00.fits',
         'gal_mask_mode': '0.2',
         'sp_mask_mode': '545',
    'nside': nside}
CIB_Lenz_c = {'file_map': path_CIB_Lenz+'cib_fullmission.hpx.fits',
         'file_hm1': path_CIB_Lenz+'cib_evenring.hpx.fits', 
         'file_hm2': path_CIB_Lenz+'cib_oddring.hpx.fits',
         'file_mask': path_CIB_Lenz+'mask_apod.hpx.fits',
         'file_gp_mask': None, #path_CIB_Lenz+'HFI_Mask_GalPlane-apo2_2048_R2.00.fits',
         'file_sp_mask': None, #path_CIB_Lenz+'HFI_Mask_PointSrc_2048_R2.00.fits',
         'gal_mask_mode': '0.2',
         'sp_mask_mode': '545',
         'nside': nside, 
         'beam_info': None}
CIB_Lenz_b_c = {'file_map': path_CIB_Lenz+'cib_fullmission.hpx.fits',
         'file_hm1': path_CIB_Lenz+'cib_evenring.hpx.fits', 
         'file_hm2': path_CIB_Lenz+'cib_oddring.hpx.fits',
         'file_mask': path_CIB_Lenz+'mask_apod.hpx.fits',
         'file_gp_mask': None, #path_CIB_Lenz+'HFI_Mask_GalPlane-apo2_2048_R2.00.fits',
         'file_sp_mask': None, #path_CIB_Lenz+'HFI_Mask_PointSrc_2048_R2.00.fits',
         'gal_mask_mode': '0.2',
         'sp_mask_mode': '545',
         'nside': nside, 
         'beam_info': {'type': 'Gaussian', 
                       'FWHM_arcmin': 5.0}}
path = '/mnt/zfsusers/jaimerz/PhD/xCell/'
c_test = {'file_map': path+'xcell/tests/data/map.fits',
        'file_mask': path+'xcell/tests/data/map.fits',
        'nside': 32}

CIBlenz_mapper = MapperP15CIB(CIB_Lenz_c)
CIBlenz_map = CIBlenz_mapper.get_signal_map() 
CIBlenz_mask = CIBlenz_mapper.get_mask()
CIBlenz_f = CIBlenz_mapper.get_nmt_field()
CIBlenz_hm1, CIBlenz_hm2 = CIBlenz_mapper._get_hm_maps()
CIBlenz_hm1_f =  CIBlenz_mapper._get_nmt_field(signal=CIBlenz_hm1)
CIBlenz_hm2_f = CIBlenz_mapper._get_nmt_field(signal=CIBlenz_hm2)
######
CIBlenz_b_mapper = MapperP15CIB(CIB_Lenz_b_c)
CIBlenz_b_f = CIBlenz_b_mapper.get_nmt_field()
CIBlenz_b_hm1, CIBlenz_b_hm2 = CIBlenz_b_mapper._get_hm_maps()
CIBlenz_b_hm1_f =  CIBlenz_b_mapper._get_nmt_field(signal=CIBlenz_b_hm1)
CIBlenz_b_hm2_f = CIBlenz_b_mapper._get_nmt_field(signal=CIBlenz_b_hm2)

CIB_ws = nmt.NmtWorkspace()
CIB_ws.compute_coupling_matrix(CIBlenz_f, CIBlenz_f, bands)
####
CIB_b_ws = nmt.NmtWorkspace()
CIB_b_ws.compute_coupling_matrix(CIBlenz_b_f, CIBlenz_b_f, bands)

CIB_nl_c = CIBlenz_mapper.get_nl_coupled()
CIB_nl_dc = CIB_ws.decouple_cell(CIB_nl_c)
####
CIB_b_nl_c = CIBlenz_b_mapper.get_nl_coupled()
CIB_b_nl_dc = CIB_b_ws.decouple_cell(CIB_b_nl_c)

CIB_cl_fl_c = nmt.compute_coupled_cell(CIBlenz_hm1_f, CIBlenz_hm2_f)
CIB_cl_fl_dc = CIB_ws.decouple_cell(CIB_cl_fl_c) 
CIB_cl_c = nmt.compute_coupled_cell(CIBlenz_f, CIBlenz_f)
CIB_cl_dc = CIB_ws.decouple_cell(CIB_cl_c) 
####
CIB_b_cl_fl_c = nmt.compute_coupled_cell(CIBlenz_b_hm1_f, CIBlenz_b_hm2_f)
CIB_b_cl_fl_dc = CIB_ws_b.decouple_cell(CIB_b_cl_fl_c) 
CIB_b_cl_c = nmt.compute_coupled_cell(CIBlenz_b_f, CIBlenz_b_f)
CIB_b_cl_dc = CIB_ws_b.decouple_cell(CIB_b_cl_c) 

CIB_dl_fl_dc = ell_arr*CIB_cl_fl_dc
CIB_dl_fl_dc *= (10**6)**2 
CIB_dl_dc = ell_arr*CIB_cl_dc
CIB_dl_dc *= (10**6)**2 
CIB_dnl_dc = ell_arr*CIB_nl_dc
CIB_dnl_dc *= (10**6)**2 
####
CIB_b_dl_fl_dc = ell_arr*CIB_b_cl_fl_dc
CIB_b_dl_fl_dc *= (10**6)**2 
CIB_b_dl_dc = ell_arr*CIB_b_cl_dc
CIB_b_dl_dc *= (10**6)**2 
CIB_b_dnl_dc = ell_arr*CIB_b_nl_dc
CIB_b_dnl_dc *= (10**6)**2 


np.savez('numbers.npz',
        ell_arr = ell_arr,
        CIB_dl_dc = CIB_cl_dc,
        CIB_dl_fl_dc = CIB_dl_fl_dc,
        CIB_dnl_dc = CIB_nl_dc,
        CIB_b_dl_dc = CIB_b_cl_dc,
        CIB_b_dl_fl_dc = CIB_b_dl_fl_dc,
        CIB_b_dnl_dc = CIB_b_nl_dc)