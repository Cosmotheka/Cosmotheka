import numpy as np
import pandas as pd
import healpy as hp
from scipy.interpolate import interp1d
from .utils import rotate_map
from .mapper_Planck_base import MapperPlanckBase


class MapperCIBLenz(MapperPlanckBase):
    """
    Note that X in the names of files and directories \
    stands for either of the three available frequency \
    channels `X = [353, 545, 857]`.

    **Config**
        - file_map: \
          `".../Datasets/CIB_Lenz/X/2.5e+20_gp20/cib_fullmission.hpx.fits"`
        - file_hm1: \
          `".../Datasets/CIB_Lenz/X/2.5e+20_gp20/cib_evenring.hpx.fits"`
        - file_hm2: \
          `".../Datasets/CIB_Lenz/X/2.5e+20_gp20/cib_oddring.hpx.fits"`
        - file_mask: \
          `".../Datasets/CIB_Lenz/X/2.5e+20_gp20/mask_apod.hpx.fits"`
        - beam_info:

            - type: `'Custom'`
            - file: \
            `'.../Datasets/CIB_Lenz/X/2.5e+20_gp20/windowfunctions.csv'`
            - field: `'Wl_eff'`
            - type: `'PixWin'`
            - nside_native: `2048`
            - nside_wanted: `4096`

        - mask_name: `"mask_CIB_Lenz_X"`
        - path_rerun: `'.../Datasets/CIB_Lenz_X/xcell_runs'`
    """
    map_name = 'CIBLenz'

    def __init__(self, config):
        self._get_Planck_defaults(config)
        self.beam_info = config.get('beam_info',
                                    [{'type': 'Gaussian',
                                      'FWHM_arcmin': 5.0}])
        self.map_name += f"_{config.get('band')}"

    def _get_custom_beam(self, info):
        # Constructs beam function
        # based on window function
        # from file

        fname = info['file']
        field = info['field']
        windowfuncs = pd.read_csv(fname, comment='#')
        pixwin = interp1d(np.array(windowfuncs['ell']),
                          np.log(np.array(windowfuncs[field])),
                          fill_value='extrapolate')
        ell = np.arange(3*self.nside)
        return np.exp(pixwin(ell))

    def _generate_hm_maps(self):
        hm1_map = hp.read_map(self.file_hm1)
        hm1_map[hm1_map == hp.UNSEEN] = 0.0
        hm1_map[np.isnan(hm1_map)] = 0.0
        hm1_map = rotate_map(hm1_map, self.rot)
        hm1_map = hp.ud_grade(hm1_map, nside_out=self.nside)

        hm2_map = hp.read_map(self.file_hm2)
        hm2_map[hm2_map == hp.UNSEEN] = 0.0
        hm2_map[np.isnan(hm2_map)] = 0.0
        hm2_map = rotate_map(hm2_map, self.rot)
        hm2_map = hp.ud_grade(hm2_map, nside_out=self.nside)

        return np.array([hm1_map, hm2_map])

    def get_dtype(self):
        return 'generic'
