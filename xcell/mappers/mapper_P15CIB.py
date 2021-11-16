from .mapper_Planck_base import MapperPlanckBase
import healpy as hp
import numpy as np


class MapperP15CIB(MapperPlanckBase):
    def __init__(self, config):
        """
        config - dict
        {'file_map': path+'COM_CompMap_Compton-SZMap-milca_2048_R2.00.fits',
         'file_mask': path+'COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits',
         'mask_name': 'mask_CIB',
         'nside':512}
        """
        self._get_Planck_defaults(config)
        self.beam_info = config.get('beam_fwhm_arcmin', 5.)
        self.gp_mask_mode = config.get('gp_mask_mode', '0.6')
        self.gp_mask_modes = {'0.2': 0,
                              '0.4': 1,
                              '0.6': 2,
                              '0.7': 3,
                              '0.8': 4,
                              '0.9': 5,
                              '0.97': 6,
                              '0.99': 7}
        self.ps_mask_mode = config.get('ps_mask_mode', ['545'])
        self.ps_mask_modes = {'100': 0,
                              '143': 1,
                              '217': 2,
                              '353': 3,
                              '545': 4,
                              '857': 5}

    def _get_hm_maps(self):
        if self.hm1_map is None:
            hm1_map = hp.read_map(self.file_hm1)
            self.hm1_map = [hp.ud_grade(hm1_map,
                            nside_out=self.nside)]
            self.hm1_map[0][self.hm1_map[0] == hp.UNSEEN] = 0.0
            self.hm1_map[0][np.isnan(self.hm1_map[0])] = 0.0
        if self.hm2_map is None:
            hm2_map = hp.read_map(self.file_hm2)
            self.hm2_map = [hp.ud_grade(hm2_map,
                            nside_out=self.nside)]
            self.hm2_map[0][self.hm2_map[0] == hp.UNSEEN] = 0.0
            self.hm2_map[0][np.isnan(self.hm2_map[0])] = 0.0

        return self.hm1_map, self.hm2_map

    def get_dtype(self):
        return 'generic'
