import os
import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.io import fits
from astropy.table import Table, vstack
from .utils import get_map_from_points

class MapperBase(object):
    def __init__(self, config):
        self._get_defaults(config)

    def _get_defaults(self, config):
        self.config = config
        self.mask_name = config.get('mask_name', None)
        self.nside = config['nside']
        self.nmt_field = None

    def get_signal_map(self):
        raise NotImplementedError("Do not use base class")

    def get_contaminants(self):
        return None

    def get_mask(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_coupled(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_covariance(self):
        raise NotImplementedError("Do not use base class")

    def get_nmt_field(self, **kwargs):
        if self.nmt_field is None:
            signal = self.get_signal_map(**kwargs)
            mask = self.get_mask(**kwargs)
            cont = self.get_contaminants(**kwargs)
            n_iter = kwargs.get('n_iter', 0)
            self.nmt_field = nmt.NmtField(mask, signal,
                                          templates=cont, n_iter=n_iter)
        return self.nmt_field

class MapperSDSS(MapperBase):
    def __init__(self):
        return
        
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

    def get_nz(self, dz=0):
        if self.dndz is None:
            cat_data = self.get_catalog(mod='data')
            w_data = self._get_w(mod='data')
            h, b = np.histogram(cat_data['Z'], bins=self.z_arr_dim,
                                weights=w_data) #, range=[0.5, 2.5])
            self.dndz = np.array([0.5 * (b[:-1] + b[1:]), h])
        z, nz = self.dndz
        z_dz = z + dz
        sel = z_dz >= 0
        return np.array([z_dz[sel], nz[sel]])
    
    def _get_alpha(self):
        if self.alpha is None:
            w_data = self._get_w(mod='data')
            w_random = self._get_w(mod='random')
            self.alpha = np.sum(w_data)/np.sum(w_random)
        return self.alpha
    
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