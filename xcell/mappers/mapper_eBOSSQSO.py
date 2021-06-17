from .mapper_base import MapperSDSS
from .utils import get_map_from_points
import numpy as np
import healpy as hp


class MappereBOSSQSO(MapperSDSS):
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
        self.z_edges = config.get('z_edges', [0, 3])

        self.ws = {'data': None, 'random': None}
        self.alpha = None

        self.dndz = None
        self.delta_map = None
        self.nl_coupled = None
        self.mask = None
        self.nside_nl_threshold = config.get('nside_nl_threshold',
                                             4096)
        self.lmin_nl_from_data = config.get('lmin_nl_from_data',
                                            2000)

    def _bin_z(self, cat):
        return cat[(cat['Z'] >= self.z_edges[0]) &
                   (cat['Z'] < self.z_edges[1])]

    def _get_w(self, mod='data'):
        if self.ws[mod] is None:
            cat = self.get_catalog(mod=mod)
            cat_SYSTOT = np.array(cat['WEIGHT_SYSTOT'])
            cat_CP = np.array(cat['WEIGHT_CP'])
            cat_NOZ = np.array(cat['WEIGHT_NOZ'])
            self.ws[mod] = cat_SYSTOT*cat_CP*cat_NOZ  # FKP left out
        return self.ws[mod]

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
