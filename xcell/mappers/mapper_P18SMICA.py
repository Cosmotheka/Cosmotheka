from .mapper_Planck_base import MapperPlanckBase
import healpy as hp


class MapperP18SMICA(MapperPlanckBase):
    def __init__(self, config):
        """
        config - dict
        {'file_map': [path+'COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits'],
         '',
         'nside':512}
        """
        self._get_Planck_defaults(config)
        self.beam_info = config.get('beam_info',
                                    [{'type': 'Gaussian',
                                     'FWHM_arcmin': 5.0}])
        self.gp_mask_mode = config.get('gp_mask_mode', '0.6')
        self.gp_mask_modes = {'0.2': 0,
                              '0.4': 1,
                              '0.6': 2,
                              '0.7': 3,
                              '0.8': 4,
                              '0.9': 5,
                              '0.97': 6,
                              '0.99': 7}

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

    def get_dtype(self):
        return 'cmb_temperature'
