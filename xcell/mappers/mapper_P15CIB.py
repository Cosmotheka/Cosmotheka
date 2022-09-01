from .mapper_Planck_base import MapperPlanckBase
from .utils import rotate_map
import healpy as hp
import numpy as np


class MapperP15CIB(MapperPlanckBase):
    def __init__(self, config):
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
        self.ps_mask_mode = config.get('ps_mask_mode', ['545'])
        self.ps_mask_modes = {'100': 0,
                              '143': 1,
                              '217': 2,
                              '353': 3,
                              '545': 4,
                              '857': 5}

    def _get_hm_maps(self):
        # Returns the half mission maps of the mapper \
        # after masking NaN values and applying the \
        # neccesary coordinate rotations.

        # Returns:
        #    hm1_map (Array)
        #    hm2_map (Array)

        if self.hm1_map is None:
            hm1_map = hp.read_map(self.file_hm1)
            hm1_map[hm1_map == hp.UNSEEN] = 0.0
            hm1_map[np.isnan(hm1_map)] = 0.0
            hm1_map = rotate_map(hm1_map, self.rot)
            self.hm1_map = [hp.ud_grade(hm1_map,
                            nside_out=self.nside)]
        if self.hm2_map is None:
            hm2_map = hp.read_map(self.file_hm2)
            hm2_map[hm2_map == hp.UNSEEN] = 0.0
            hm2_map[np.isnan(hm2_map)] = 0.0
            hm2_map = rotate_map(hm2_map, self.rot)
            self.hm2_map = [hp.ud_grade(hm2_map,
                            nside_out=self.nside)]

        return self.hm1_map, self.hm2_map

    def get_dtype(self):
        return 'generic'
