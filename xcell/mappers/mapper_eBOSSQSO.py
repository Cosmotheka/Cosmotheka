from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.io import fits
from astropy.table import Table, vstack
import numpy as np
import healpy as hp
import pymaster as nmt
import os


class MappereBOSSQSO(MapperBase):
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

    def _get_w(self, mod='data'):
        if self.ws[mod] is None:
            cat = self.get_catalog(mod=mod)
            cat_SYSTOT = np.array(cat['WEIGHT_SYSTOT'])
            cat_CP = np.array(cat['WEIGHT_CP'])
            cat_NOZ = np.array(cat['WEIGHT_NOZ'])
            self.ws[mod] = cat_SYSTOT*cat_CP*cat_NOZ  # FKP left out
        return self.ws[mod]

    def _get_alpha(self):
        if self.alpha is None:
            w_data = self._get_w(mod='data')
            w_random = self._get_w(mod='random')
            self.alpha = np.sum(w_data)/np.sum(w_random)
        return self.alpha

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

    def get_signal_map(self):
        if self.delta_map is None:
            self.delta_map = np.zeros(self.npix)
            cat_data = self.get_catalog(mod='data')
            cat_random = self.get_catalog(mod='random')
            w_data = self._get_w(mod='data')
            w_random = self._get_w(mod='random')
            alpha = self._get_alpha()
            nmap_data = get_map_from_points(cat_data, self.nside,
                                            w=w_data)
            nmap_random = get_map_from_points(cat_random, self.nside,
                                              w=w_random)
            mask = self.get_mask()
            goodpix = mask > 0
            self.delta_map = (nmap_data - alpha * nmap_random)
            self.delta_map[goodpix] /= mask[goodpix]
        return [self.delta_map]

    def get_mask(self):
        if self.mask is None:
            cat_random = self.get_catalog(mod='random')
            w_random = self._get_w(mod='random')
            alpha = self._get_alpha()
            self.mask = get_map_from_points(cat_random,
                                            self.nside_mask,
                                            w=w_random)
            self.mask *= alpha
            # Account for different pixel areas
            area_ratio = (self.nside_mask/self.nside)**2
            self.mask = area_ratio * hp.ud_grade(self.mask,
                                                 nside_out=self.nside)
        return self.mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            if self.nside < 4096:
                print('calculing nl from weights')
                cat_data = self.get_catalog(mod='data')
                cat_random = self.get_catalog(mod='random')
                w_data = self._get_w(mod='data')
                w_random = self._get_w(mod='random')
                alpha = self._get_alpha()
                pixel_A = 4*np.pi/hp.nside2npix(self.nside)
                mask = self.get_mask()
                w2_data = get_map_from_points(cat_data, self.nside,
                                              w=w_data**2)
                w2_random = get_map_from_points(cat_random, self.nside,
                                                w=w_random**2)
                goodpix = mask > 0
                N_ell = (w2_data[goodpix].sum() +
                         alpha**2*w2_random[goodpix].sum())
                N_ell *= pixel_A**2/(4*np.pi)
                self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
            else:
                print('calculating nl from mean cl values')
                f = self.get_nmt_field()
                cl = nmt.compute_coupled_cell(f, f)[0]
                N_ell = np.mean(cl[2000:2*self.nside])
                self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
