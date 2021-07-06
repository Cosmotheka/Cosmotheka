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

        self.w_method = 'BOSS'
        self.ws = {'data': None, 'random': None}
        self.alpha = None

        self.dndz = None
        self.delta_map = None
        self.nl_coupled = None
        self.mask = None
