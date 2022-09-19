from .mapper_Planck_base import MapperPlanckBase
from .utils import rotate_map
import healpy as hp


class MapperP18SMICA(MapperPlanckBase):
    map_name = "P18SMICA"

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
        self.ps_mask_modes = {'F100': 0,
                              'F143': 1,
                              'F217': 2,
                              'F353': 3}
        self.ps_mask_mode = config.get('ps_mask_mode',
                                       ['F100', 'F143', 'F217', 'F353'])

    def _get_hm_maps(self):
        if self.hm1_map is None:
            def get_hm_maps()
                hm1_map = hp.read_map(self.file_hm1)
                ps_mask = self._get_ps_mask()
                hm1_map *= ps_mask
                hm1_map = rotate_map(hm1_map, self.rot)
                hm1_map = hp.ud_grade(hm1_map, nside_out=self.nside)

                hm2_map = hp.read_map(self.file_hm2)
                ps_mask = self._get_ps_mask()
                hm2_map *= ps_mask
                hm2_map = rotate_map(hm2_map, self.rot)
                hm2_map = hp.ud_grade(hm2_map, nside_out=self.nside)

                return np.array([hm1_map, hm2_map])

            self.hm1_map, self.hm2_map = self._rerun_read_cycle(fn, 'FITSMap',
                                                                get_hm_maps)

        return self.hm1_map, self.hm2_map

    def get_dtype(self):
        return 'cmb_temperature'
