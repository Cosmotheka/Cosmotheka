from .mapper_base_Planck import MapperBasePlanck
import healpy as hp
import numpy as np


class MapperP18SMICA(MapperBasePlanck):
    def __init__(self, config):
        """
        config - dict
        {'file_map': [path+'COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits'],
         '',
         'nside':512}
        """
        self._get_Planck_defaults(config)

    def _get_hm_maps(self):
        if self.hm1_map is None:
            hm1_map = hp.read_map(self.file_hm1)
            self.hm1_map = [hp.ud_grade(hm1_map,
                            nside_out=self.nside)]
        if self.hm2_map is None:
            hm2_map = hp.read_map(self.file_hm2)
            self.hm2_map = [hp.ud_grade(hm2_map,
                            nside_out=self.nside)]
        return self.hm1_map, self.hm2_map

    def get_mask(self):
        if self.mask is None:
            self.mask = np.ones(12*self.nside**2)
            if self.file_gp_mask is not None:
                field = self.gal_mask_modes[self.gal_mask_mode]
                mask = hp.read_map(self.file_gp_mask, field)
                mask = hp.ud_grade(mask,
                                   nside_out=self.nside)
                self.mask *= mask
            if self.file_sp_mask is not None:
                mask = hp.read_map(self.file_sp_mask)
                mask = hp.ud_grade(mask,
                                   nside_out=self.nside)
                self.mask *= mask
        return self.mask

    def get_dtype(self):
        return 'cmb_temperature'
