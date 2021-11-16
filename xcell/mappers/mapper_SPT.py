from .mapper_Planck_base import MapperPlanckBase
import healpy as hp
import numpy as np


class MapperSPT(MapperPlanckBase):
    def __init__(self, config):
        """
        config - dict
        {'file_map': path+'COM_CompMap_Compton-SZMap-ymaps_2048_R2.00.fits',
         'file_mask': path+'COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits',
         'mask_name': 'mask_tSZ',
         'nside':512}
        """
        self._get_Planck_defaults(config)
        # Do we have beam in SPT?
        # self.beam = None
        # self.beam_info = None

    def _get_hm_maps(self):
        if self.hm1_map is None:
            hm1_map = hp.read_map(self.file_hm1_map)
            self.hm1_map = [hp.ud_grade(hm1_map,
                            nside_out=self.nside)]
        if self.hm2_map is None:
            hm2_map = hp.read_map(self.file_hm2_map)
            self.hm2_map = [hp.ud_grade(hm2_map,
                            nside_out=self.nside)]
        return self.hm1_map, self.hm2_map

    def get_dtype(self):
        return 'cmb_tSZ'
