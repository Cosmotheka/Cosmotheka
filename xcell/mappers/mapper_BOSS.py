from .mapper_SDSS import MapperSDSS
import numpy as np


class MapperBOSS(MapperSDSS):
    def __init__(self, config):
        """
        config - dict
        {'data_catalogs':[data_path+'BOSSLOWZ/galaxy_DR12v5_LOWZ_North.fits'],
        'random_catalogs':[data_path+'BOSSLOWZ/random1_DR12v5_LOWZ_North.fits'],
        'mask_path':[data_path+'BOSSLOWZ/mask_DR12v5_LOWZ_North.fits'],
        'z_edges':[0.15, 0.43] / [0.43, 0.75]
        'nside':nside, 'nside_mask': nside_mask,
        'mask_name': 'mask_LOWZ_NGC'}
        """
        self._get_SDSS_defaults(config)
        self.z_edges = config.get('z_edges', [0, 1])

    def _get_w(self, mod='data'):
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
