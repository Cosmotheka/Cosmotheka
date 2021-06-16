from .mapper_base import MapperSDSS
from .utils import get_map_from_points
import numpy as np
import healpy as hp


class MapperBOSSLOWZ(MapperSDSS):
    def __init__(self, config):
        """
        config - dict
          {'data_catalogs':['eBOSS_QSO_clustering_data-NGC-vDR16.fits'],
           'random_catalogs':['eBOSS_QSO_clustering_random-NGC-vDR16.fits'],
           'z_edges':[0, 1.5],
           'nside':nside,
           'nside_mask': nside_mask,
           'mask_name': 'mask_QSO_NGC_1'}
        """
        self._get_defaults(config)

        self.cats = {'data': None, 'random': None}

        self.z_arr_dim = config.get('z_arr_dim', 50)
        self.nside_mask = config.get('nside_mask', 512)
        self.npix = hp.nside2npix(self.nside)
        self.mask_path = config.get('mask_path', None)
        self.z_edges = config.get('z_edges', [0.15, 0.43])

        self.ws = {'data': None, 'random': None}
        self.alpha = None

        self.dndz = None
        self.delta_map = None
        self.nl_coupled = None
        self.mask = None

    def _bin_z(self, cat):
        return cat[(cat['Z'] >= self.z_edges[0]) &
                   (cat['Z'] < self.z_edges[1])]

    def _get_w(self, mod='data'):
        # Could make this more general and pass it to the superclass
        if self.ws[mod] is None:
            cat = self.get_catalog(mod=mod)
            if mod == 'data':
                w_systot = np.array(cat['WEIGHT_SYSTOT'])
                w_cp = np.array(cat['WEIGHT_CP'])
                w_noz = np.array(cat['WEIGHT_NOZ'])
                #w = w_systot*(w_cp+w_noz-1)
                w = w_systot*w_cp*w_noz
                # Eqn. 50 of https://arxiv.org/pdf/1509.06529.pdf
            elif mod == 'random':
                w = np.ones_like(cat['RA'])
            self.ws[mod] = w  # FKP left out
        return self.ws[mod]

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0