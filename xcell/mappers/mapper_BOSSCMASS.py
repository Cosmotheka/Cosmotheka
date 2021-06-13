from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.io import fits
from astropy.table import Table, vstack
import numpy as np
import healpy as hp
import pymaster as nmt
import os


class MapperBOSSCMASS(MapperBase):
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
        self.z_edges = config['z_edges']

        self.ws = {'data': None, 'random': None}
        self.alpha = None

        self.dndz = None
        self.delta_map = None
        self.nl_coupled = None
        self.mask = None

    def get_catalog(self, mod='data'):
        if mod == 'data':
            data_file = self.config['data_catalogs']
        else:
            data_file = self.config['random_catalogs']

        if self.cats[mod] is None:
            cats = []
            for file in data_file:
                if not os.path.isfile(file):
                    raise ValueError(f"File {file} not found")
                with fits.open(file) as f:
                    cats.append(self._bin_z(Table.read(f)))
            self.cats[mod] = vstack(cats)
        return self.cats[mod]

    def _bin_z(self, cat):
        return cat[(cat['Z'] >= self.z_edges[0]) &
                   (cat['Z'] < self.z_edges[1])]

    def get_nz(self, dz=0):
        if self.dndz is None:
            cat_data = self.get_catalog(mod='data')
            w_data = self._get_w(mod='data')
            h, b = np.histogram(cat_data['Z'], bins=self.z_arr_dim,
                                weights=w_data, range=[0.5, 2.5])
            self.dndz = np.array([0.5 * (b[:-1] + b[1:]), h])
        z, nz = self.dndz
        z_dz = z + dz
        sel = z_dz >= 0
        return np.array([z_dz[sel], nz[sel]])

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
