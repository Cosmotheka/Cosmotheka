from .mapper_SDSS import MapperSDSS
import numpy as np


class MapperBOSS(MapperSDSS):
    """
    Note that X in file names stands for either `'CMASS'` or `'LOWZ'`.

    **Config**

        - num_z_bins: `50`
        - nside_mask: `512`
        - nside_nl_threshold: `4096`
        - lmin_nl_from_data: `2000`
        - data_catalogs: \
          `['.../Datasets/BOSS/galaxy_DR12v5_X_North.fits.gz' \
            '.../Datasets/BOSS/galaxy_DR12v5_X_South.fits.gz']`
        - random_catalogs: \
          `['.../Datasets/BOSS/random0_DR12v5_X_North.fits.gz', \
            '.../Datasets/BOSS/random0_DR12v5_X_South.fits.gz']`

        - z_edges: `[0.4, 0.8]` / `[0.1, 0.43]`
        - path_rerun: `'.../Datasets/BOSS/xcell_runs'`
        - mask_name: `'mask_X'`
        - mapper_class: `'MapperBOSS'`
        - bias: `1.93` / `1.93`
    """
    def __init__(self, config):
        self._get_SDSS_defaults(config)
        self.z_edges = config.get('z_edges', [0, 1])

    def _get_w(self, mod='data'):
        # Returns the weights for the sources of \
        # the mapper's data or randoms catalog.

        if self.ws[mod] is None:
            cat = self.get_catalog(mod=mod)
            if mod == 'data':
                w_systot = np.array(cat['WEIGHT_SYSTOT'])
                w_cp = np.array(cat['WEIGHT_CP'])
                w_noz = np.array(cat['WEIGHT_NOZ'])
                w = w_systot*(w_cp+w_noz-1)  # FKP left out
                # Eqn. 50 of https://arxiv.org/pdf/1509.06529.pdf
            if mod == 'random':
                w = np.ones_like(cat['RA'])
            self.ws[mod] = w
        return self.ws[mod]

    def _get_random_cols(self):
        # Returns the names of the columns \
        # of interest for the mapper.
        return ['RA', 'DEC', 'Z']
