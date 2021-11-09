# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '/mnt/zfsusers/jaimerz/PhD/xCell')

from xcell.mappers import MapperP15tSZ
from xcell.mappers import MapperP15CIB
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

nside = 512
#ells = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 272, 309,
#        351, 398, 452, 513, 582, 661, 750, 852, 967, 1098,
#        1247, 1416, 1536, 1608, 1826, 2073, 2354, 2673, 3072])
tSZ_tab = np.array([[10, 5.0496e-03, 1.3919e-01, 5.7234e-03],
                [13.5, 8.7080e-03, 1.1643e-01, 7.5606e-03],
                [18, 1.3430e-02, 9.9155e-02, 9.9280e-03],
                [23.5, 2.9325e-02, 8.6276e-02, 1.2834e-02],
                [30.5, 2.1520e-02, 7.4827e-02, 1.6544e-02],
                [40, 2.6605e-02, 6.4972e-02, 2.1591e-02],
                [52.5, 3.9626e-02, 5.6584e-02, 2.8250e-02],
                [68.5, 3.9820e-02, 4.9513e-02, 3.6721e-02],
                [89.5, 6.0241e-02, 4.3516e-02, 4.7775e-02],
                [117, 9.9878e-02, 3.8669e-02, 6.2081e-02],
                [152.5, 1.1375e-01, 3.4993e-02, 8.0185e-02],
                [198, 1.3429e-01, 3.2630e-02, 1.0271e-01],
                [257.5, 1.7920e-01, 3.3977e-02, 1.3093e-01],
                [335.5, 2.2076e-01, 4.3272e-02, 1.6575e-01],
                [436.5, 2.6166e-01, 6.0483e-02, 2.0728e-01],
                [567.5, 2.7879e-01, 7.9377e-02, 2.5553e-01],
                [738, 3.3226e-01, 1.1034e-01, 3.0982e-01],
                [959.5, 4.3979e-01, 1.4774e-01, 3.6810e-01],
                [1247.5, 0, 0, 4.2699e-01],
                [1622, 0, 0, 4.8154e-01],
                [2109, 0, 0, 5.2553e-01],
                [2742, 0, 0, 5.5249e-01]])
CIB_tab = np.array([[33.5, 0.06117613548624482, 0.04258439361767112],
            [97.5, 0.15318377317476986, 0.01889072396960295],
            [161.5, 0.09990097140944328, 0.011211039004065013],
            [225.5, 0.08374607570003902, 0.008501732945214564],
            [289.5, 0.07458065659363647, 0.006907185391668387],
            [353.5, 0.04427605390362208, 0.005622426063339678],
            [417.5, 0.03577421628740376, 0.004604899713339137],
            [481.5, 0.024529655647115622, 0.0038939061476286918],
            [545.5, 0.02271158518311891, 0.0034490936380393604],
            [609.5, 0.020279010962954352, 0.0031492516396896797],
            [673.5, 0.017714350200909033, 0.0028842521312894575],
            [737.5, 0.01588524872635897, 0.0026349438123689822],
            [801.5, 0.013408593653920123, 0.0024381847537097642],
            [865.5, 0.013753794398618673, 0.002301008565117063],
            [929.5, 0.009970670151884489, 0.002205932069184609],
            [993.5, 0.012618297816938042, 0.0021169406993021118],
            [1057.5, 0.006580753973982416, 0.002028920582554497],
            [1121.5, 0.004486733799860173, 0.00196656008458652],
            [1185.5, 0.00838170050949548, 0.001940913480551445],
            [1249.5, 0.00893293464904029, 0.001940848437327486],
            [1313.5, 0.008743169269572516, 0.0019399097786482263],
            [1377.5, 0.004712125103085124, 0.0019296833926719574],
            [1441.5, 0.0034641541127929535, 0.0019224919275331878],
            [1505.5, 0.00664800238822738, 0.0019305907912808382],
            [1569.5, 0.00664929042844447, 0.0019426620989430301],
            [1633.5, 0.00400518624362728, 0.001934979951758305],
            [1697.5, 0.004899282006285914, 0.0019102481055881812],
            [1761.5, 0.004285617602723226, 0.0018835588168846652],
            [1825.5, 0.00676815035622379, 0.0018748844245898108],
            [1889.5, 0.0033332756374055542, 0.0018694199331149373],
            [1953.5, 0.004306622354363802, 0.0018284557516869656]])
ellss = np.transpose(CIB_tab)[0] #np.array([int(x) for x in np.transpose(Tab4)[0]])
#ells = np.array([int(x) for x in np.transpose(Tab4)[0]])
CIBxCMBkcl_ref = np.transpose(CIB_tab)[1]
CIBxCMBkdl_ref = np.transpose(CIB_tab)[0]*(np.transpose(CIB_tab)[1])#-np.transpose(CIB_tab)[2])
CIBcl_ref = [79173.4169839208, 94297.5708827149, 62011.3903711981,
            47372.6697631867, 36103.0039118132, 26946.309996208,
            23714.598042715, 18980.8394675022, 16703.9570462782,
            14051.3340327611, 12749.3113347196, 11293.8664763503,
            10250.2715313782, 9450.11848326413, 8644.55383545777,
            7986.82491748145, 7586.76120184488, 7032.3343042923,
            6594.03156356358, 6351.88472750077, 6012.76923059271,
            5654.98923348836, 5454.42975018716, 5240.10695493978,
            5030.55386291533, 4867.44621812233, 4751.93920326956,
            4559.15275518212, 4503.48473732335, 4390.57355840046,
            4391.76604428468]
CIBdl_ref = np.transpose(CIB_tab)[0]*CIBcl_ref

tSZDl_ref = np.transpose(tSZ_tab)[1]
tSZDl_FG_ref =  [0.00508, 0.00881, 0.01363, 0.02961, 0.02241, 0.02849, 0.04276, 0.04580,
                 0.07104, 0.11914, 0.15150, 0.19390, 0.28175, 0.39837, 0.56743, 0.76866,
                 1.11010, 1.66140, 2.52170, 4.58510, 12.2690, 165.600]
#ells = ells[ells<3*nside]
#bands = nmt.NmtBin(nside, nlb=4, is_Dell=False)
bands = nmt.NmtBin(nside, nlb=60) #nmt.NmtBin.from_edges(ells[:-1], ells[1:])
ell_arr = bands.get_effective_ells()

path_CMBk = '/home/jaimerz/PhD/xCell/data/P15/CMBk/'
CMBk_c =  {'file_klm': path_CMBk+'dat_klm.fits', 
          'file_mask':path_CMBk+'mask.fits.gz',
           'file_noise':path_CMBk+'nlkk.dat',
           'mask_name': 'mask_CMBK',
           'coordinates': 'G',
          'nside':nside}
path_tSZ = '/mnt/zfsusers/jaimerz/PhD/xCell/data/P18/tSZ/'
tSZ_c = {'file_map': path_tSZ+'COM_CompMap_Compton-SZMap-milca-ymaps_2048_R2.00.fits', 
    'file_hm1': path_tSZ+'nilc_ymaps.fits', 
     'file_hm2': path_tSZ+'COM_CompMap_Compton-SZMap-milca-ymaps_2048_R2.00.fits',
     'file_noise': path_tSZ+'COM_CompMap_Compton-SZMap-milca-stddev_2048_R2.00.fits',
     'file_mask': path_tSZ+'COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits',
     'mask_name': 'mask_tSZ',
     'gal_mask_mode': '0.5',
     'nside': nside}
path_SMICA = '/mnt/zfsusers/jaimerz/PhD/xCell/data/P18/SMICA/'
SMICA_c = {'file_map': path_SMICA+'COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits',
     'file_hm1': path_SMICA+'COM_CMB_IQU-smica-nosz_2048_R3.00_hm1.fits', 
     'file_hm2': path_SMICA+'COM_CMB_IQU-smica-nosz_2048_R3.00_hm2.fits',
     'file_mask': None,
     'file_gp_mask': path_SMICA+'HFI_Mask_GalPlane-apo2_2048_R2.00.fits',
     'file_sp_mask': path_SMICA+'HFI_Mask_PointSrc_2048_R2.00.fits',
    'nside': nside}
path_CIB = '/mnt/zfsusers/jaimerz/PhD/xCell/data/P18/CIB/'
path_CIB_Lenz = '/home/jaimerz/PhD/xCell/data/P18/CIB_Lenz/'
CIB_c = {'file_map': path_CIB+'COM_CompMap_CIB-GNILC-F545_2048_R2.00.fits',
         'file_hm1': path_CIB+'COM_CMB_IQU-nilc_2048_R3.00_hm1.fits', 
         'file_hm2': path_CIB+'COM_CMB_IQU-nilc_2048_R3.00_hm2.fits',
         'file_mask':  path_CIB_Lenz+'mask_apod.hpx.fits', #None,
         'file_gp_mask': None, #path_SMICA+'HFI_Mask_GalPlane-apo2_2048_R2.00.fits',
         'file_sp_mask': None, #path_SMICA+'HFI_Mask_PointSrc_2048_R2.00.fits',
         'gal_mask_mode': '0.2',
         'sp_mask_mode': '545',
    'nside': nside}
path_CIB_Lenz = '/mnt/zfsusers/jaimerz/PhD/xCell/data/P18/CIB_Lenz/'
CIB_Lenz_c = {'file_map': path_CIB_Lenz+'cib_fullmission.hpx.fits',
         'file_hm1': path_CIB_Lenz+'cib_evenring.hpx.fits', 
         'file_hm2': path_CIB_Lenz+'cib_oddring.hpx.fits',
         'file_mask': path_CIB_Lenz+'mask_apod.hpx.fits',
         'file_gp_mask': None, #path_CIB_Lenz+'HFI_Mask_GalPlane-apo2_2048_R2.00.fits',
         'file_sp_mask': None, #path_CIB_Lenz+'HFI_Mask_PointSrc_2048_R2.00.fits',
         'gal_mask_mode': '0.2',
         'sp_mask_mode': '545',
         'nside': nside}
path = '/mnt/zfsusers/jaimerz/PhD/xCell/'
c_test = {'file_map': path+'xcell/tests/data/map.fits',
        'file_mask': path+'xcell/tests/data/map.fits',
        'nside': 32}

CIBgnilc_mapper = MapperP15CIB(CIB_c)
CIBlenz_mapper = MapperP15CIB(CIB_Lenz_c)
CMBk_mapper = MapperP18CMBK(CMBk_c)
CIBgnilc_map = CIBgnilc_mapper.get_signal_map() 
CIBgnilc_mask = CIBgnilc_mapper.get_mask()
CIBgnilc_f = CIBgnilc_mapper.get_nmt_field()
CIBgnilc_hm1, CIBgnilc_hm2 = CIBgnilc_mapper._get_hm_maps()
CIBgnilc_hm1_f =  CIBgnilc_mapper._get_nmt_field(signal=CIBgnilc_hm1)
CIBgnilc_hm2_f = CIBgnilc_mapper._get_nmt_field(signal=CIBgnilc_hm2)
CIBlenz_map = CIBlenz_mapper.get_signal_map() 
CIBlenz_mask = CIBlenz_mapper.get_mask()
CIBlenz_f = CIBlenz_mapper.get_nmt_field()
CIBlenz_hm1, CIBlenz_hm2 = CIBlenz_mapper._get_hm_maps()
CIBlenz_hm1_f =  CIBlenz_mapper._get_nmt_field(signal=CIBlenz_hm1)
CIBlenz_hm2_f = CIBlenz_mapper._get_nmt_field(signal=CIBlenz_hm2)
CMBk_map = CMBk_mapper.get_signal_map()
CMBk_mask = CMBk_mapper.get_mask()
CMBk_f = CMBk_mapper.get_nmt_field()

CIBlenzxCMBk_ws = nmt.NmtWorkspace()
CIBlenzxCMBk_ws.compute_coupling_matrix(CIBlenz_f, CMBk_f, bands)
CIBgnilcxCMBk_ws = nmt.NmtWorkspace()
CIBgnilcxCMBk_ws.compute_coupling_matrix(CIBgnilc_f, CMBk_f, bands)
CIBlenzxCMBk_cl_c = nmt.compute_coupled_cell(CIBlenz_f, CMBk_f)
CIBlenzxCMBk_cl_dc = CIBlenzxCMBk_ws.decouple_cell(CIBlenzxCMBk_cl_c)
CIBgnilcxCMBk_cl_c = nmt.compute_coupled_cell(CIBgnilc_f, CMBk_f)
CIBgnilcxCMBk_cl_dc = CIBgnilcxCMBk_ws.decouple_cell(CIBgnilcxCMBk_cl_c) 
CIB_ws = nmt.NmtWorkspace()
CIB_ws.compute_coupling_matrix(CIBlenz_f, CIBlenz_f, bands)
CIB_nl_c = CIBlenz_mapper.get_nl_coupled()
CIB_nl_dc = CIB_ws.decouple_cell(CIB_nl_c)

CIB_cl_fl_c = nmt.compute_coupled_cell(CIBlenz_hm1_f, CIBlenz_hm2_f)
CIB_cl_fl_dc = CIB_ws.decouple_cell(CIB_cl_fl_c) 
CIB_cl_c = nmt.compute_coupled_cell(CIBlenz_f, CIBlenz_f)
CIB_cl_dc = CIB_ws.decouple_cell(CIB_cl_c) 

CIB_dl_fl_dc = ell_arr*CIB_cl_fl_dc
CIB_dl_fl_dc *= (10**6)**2 
CIB_dl_dc = ell_arr*CIB_cl_dc
CIB_dl_dc *= (10**6)**2 
CIB_dnl_dc = ell_arr*CIB_nl_dc
CIB_dnl_dc *= (10**6)**2 

CIBlenzxCMBk_dl_dc = ell_arr*CIBlenzxCMBk_cl_dc
CIBlenzxCMBk_dl_dc *= (10**6) #MJy

CIBgnilcxCMBk_dl_dc = ell_arr*CIBgnilcxCMBk_cl_dc
CIBgnilcxCMBk_dl_dc *= (10**6) #MJy

np.savez('numbers.npz',
        ell_arr = ell_arr,
        CIB_dl_dc = CIB_cl_dc,
        CIB_dl_fl_dc = CIB_dl_fl_dc,
        CIB_dnl_dc = CIB_nl_dc,
        CIBlenzxCMBk_dl_dc = CIBlenzxCMBk_dl_dc,
        CIBgnilcxCMBk_dl_dc = CIBgnilcxCMBk_dl_dc)

plt.plot(ell_arr, CIB_dnl_dc[0], 'bo-', label='CIBlenz nl full')
plt.title('CIB Lenz 545 nl Auto')
plt.legend(loc='lower left', ncol=2, labelspacing=0.1)
plt.savefig('CIB_lenz_nl_545.png')

plt.plot(ell_arr, CIB_dl_dc[0]-CIB_dnl_dc[0], 'bo-', label='CIBlenz full')
plt.plot(ell_arr, CIB_dl_fl_dc[0], 'ro-', label='CIBlenz FXL')
plt.plot(ellss, CIBdl_ref, 'ko-', label='ref')
plt.title('CIB Lenz 545 Auto')
plt.legend(loc='lower left', ncol=2, labelspacing=0.1)
plt.savefig('CIB_lenz_545_auto.png')

plt.plot(ell_arr, CIBlenzxCMBk_dl_dc[0], 'bo-', label='CIBlenzxCMBK')
plt.plot(ell_arr, CIBgnilcxCMBk_dl_dc[0], 'ro-', label='CIBlenzxCMBK')
plt.plot(ellss, CIBxCMBkdl_ref, 'ko-', label='ref')
plt.title('CIB Lenz 545 x CMBk')
plt.xlim(0, 1600)
plt.xlabel('$\\ell$', fontsize=16)
plt.ylabel('$l C_\\ell$[Jy]', fontsize=16)
plt.legend(loc='lower left', ncol=2, labelspacing=0.1)
plt.savefig('CIBxCMBk_545.png')