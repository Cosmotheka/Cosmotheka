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
        self.gal_mask_mode = config.get('gal_mask_mode', '0.6')
        self.gal_mask_modes = {'0.2': 0,
                               '0.4': 1,
                               '0.6': 2,
                               '0.7': 3,
                               '0.8': 4,
                               '0.9': 5,
                               '0.97': 6,
                               '0.99': 7}
        self.sp_mask_mode = config.get('sp_mask_mode', '545')
        self.sp_mask_modes = {'100': 0,
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
        if self.hm2_map is None:
            hm2_map = hp.read_map(self.file_hm2)
            self.hm2_map = [hp.ud_grade(hm2_map,
                            nside_out=self.nside)]
        return self.hm1_map, self.hm2_map

    def get_mask(self):
        if self.mask is None:
            if self.file_mask is not None:
                self.mask = hp.read_map(self.file_mask)
                self.mask = hp.ud_grade(self.mask,
                                        nside_out=self.nside)
            else:
                self.mask = np.ones(12*self.nside**2)
            if self.file_gp_mask is not None:
                field = self.gal_mask_modes[self.gal_mask_mode]
                gal_mask = hp.read_map(self.file_gp_mask, field)
                gal_mask = hp.ud_grade(gal_mask,
                                       nside_out=self.nside)
                self.mask *= gal_mask
            if self.file_sp_mask is not None:
                field = self.sp_mask_modes[self.sp_mask_mode]
                sp_mask = hp.read_map(self.file_sp_mask, field)
                sp_mask = hp.ud_grade(sp_mask,
                                      nside_out=self.nside)
                self.mask *= sp_mask
        return self.mask

    def get_dtype(self):
        return 'generic'
