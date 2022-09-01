import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from .mapper_P15CIB import MapperP15CIB


class MapperCIBLenz(MapperP15CIB):
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
    def __init__(self, config):
        self._get_Planck_defaults(config)
        self.beam_info = config.get('beam_info',
                                    [{'type': 'Gaussian',
                                      'FWHM_arcmin': 5.0}])

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
