from .utils import rotate_map
from .mapper_Planck_base import MapperPlanckBase
import healpy as hp
import numpy as np


class MapperP15tSZ(MapperPlanckBase):
    map_name = "P15tSZ"

    def __init__(self, config):
        """
        config - dict
        {'file_map': path+'COM_CompMap_Compton-SZMap-ymaps_2048_R2.00.fits',
         'file_mask': path+'COM_CompMap_Compton-SZMap-masks_2048_R2.01.fits',
         'mask_name': 'mask_tSZ',
         'nside':512}
        """
        self._get_Planck_defaults(config)
        self.file_hm1 = config.get('file_hm1', self.file_map)
        self.file_hm2 = config.get('file_hm2', self.file_map)
        self.beam_info = config.get('beam_info',
                                    [{'type': 'Gaussian',
                                      'FWHM_arcmin': 10.0}])
        self.gp_mask_mode = config.get('gp_mask_mode', '0.5')
        self.gp_mask_modes = {'0.4': 0,
                              '0.5': 1,
                              '0.6': 2,
                              '0.7': 3}
        self.ps_mask_mode = config.get('ps_mask_mode', ['default'])
        self.ps_mask_modes = {'test': 0,
                              'default': 4}

    def _get_hm_maps(self):
        if self.hm1_map is None:
            def get_hm_maps():
                hm1_map = hp.read_map(self.file_hm1, 1)
                ps_mask = self._get_ps_mask()
                hm1_map *= ps_mask
                hm1_map = rotate_map(hm1_map, self.rot)
                hm1_map = hp.ud_grade(hm1_map, nside_out=self.nside)

                hm2_map = hp.read_map(self.file_hm2, 2)
                ps_mask = self._get_ps_mask()
                hm2_map *= ps_mask
                hm2_map = rotate_map(hm2_map, self.rot)
                hm2_map = hp.ud_grade(hm2_map, nside_out=self.nside)

                return np.array([hm1_map, hm2_map])

            fn = '_'.join([f'{self.map_name}_hm_maps',
                           f'coord{self.coords}',
                           f'ns{self.nside}.fits.gz'])

            hm1_map, hm2_map = self._rerun_read_cycle(fn, 'FITSMap',
                                                      get_hm_maps)

            self.hm1_map = hm1_map.reshape((1, -1))
            self.hm2_map = hm2_map.reshape((1, -1))

        return self.hm1_map, self.hm2_map

    def get_dtype(self):
        return 'cmb_tSZ'
