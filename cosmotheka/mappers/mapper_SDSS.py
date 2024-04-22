import os
import numpy as np
import healpy as hp
import pymaster as nmt
import fitsio
from .utils import get_map_from_points
from .mapper_base import MapperBase


class MapperSDSS(MapperBase):
    """
    Base mapper class for all SDSS mappers. \
    
    All SDSS mappers compute their signal map \
    by computing the overdensity of galaxies \
    with respect to a random catalog. \
    
    delta = (n_data - alpha * n_random) / mask \
    mask = Area_diff * n_random * alpha \
    
    where n_data and n_random are the number \
    counts of galaxies per pixel \
    in the data and random catalogs. \
    alpha is the ratio between the sums \
    of the weights of the data and random catalog \
    The mask is computed from the random catalog \
    accounting for the different are of the pixels \
    if the map and the mask havce different \
    resolutions. \
    The noise power spectrum is computed from the \
    data catalog if the chosen resolution is below \
    a 4096, otherwise, the high multipole tail of \
    the signal power spectrum is used to estimate \
    the noise power spectrum.

    """
    # The SDSS_name passed in the configuration file will be added to map_name
    map_name = 'SDSS'

    def __init__(self, config):
        raise NotImplementedError("Do not use base class")

    def _get_SDSS_defaults(self, config):
        self._get_defaults(config)
        self.SDSS_name = config['SDSS_name']
        self.map_name += f'_{self.SDSS_name}'
        self.cats = {'data': None, 'random': None}
        self.num_z_bins = config.get('num_z_bins', 50)
        self.nside_mask = config.get('nside_mask', 512)
        self.ws = {'data': None, 'random': None}
        self.alpha = None
        self.nl_coupled = None
        self.nside_nl_threshold = config.get('nside_nl_threshold',
                                             4096)
        self.lmin_nl_from_data = config.get('lmin_nl_from_data',
                                            2000)
        self.rot = self._get_rotator('C')

    def get_catalog(self, mod='data'):
        """
        Returns the mapper's data or \
        random catalog.
        We only make use of the columns \
        cols = ['RA', 'DEC', 'Z', \
        'WEIGHT_SYSTOT',  'WEIGHT_CP ', \
        'WEIGHT_NOZ'] \
        
        Kwargs:
            mode='data'

        Returns:
            catalog (Array)
        """
        if mod == 'data':
            data_file = self.config['data_catalogs']
            cols = ['RA', 'DEC', 'Z', 'WEIGHT_SYSTOT',
                    'WEIGHT_CP', 'WEIGHT_NOZ']
        else:
            data_file = self.config['random_catalogs']
            cols = self._get_random_cols()
        if self.cats[mod] is None:
            cats = []
            for file in data_file:
                if not os.path.isfile(file):
                    raise ValueError(f"File {file} not found")
                d = fitsio.read(file, columns=cols)
                cats.append(self._bin_z(d))
            self.cats[mod] = np.hstack(cats)
        return self.cats[mod]

    def _get_nz(self):
        cat_data = self.get_catalog(mod='data')
        w_data = self._get_w(mod='data')
        h, b = np.histogram(cat_data['Z'], bins=self.num_z_bins,
                            weights=w_data, range=self.z_edges)
        zm = 0.5 * (b[:-1] + b[1:])
        nz = h
        return {'z_mid': zm, 'nz': nz}

    def get_nz(self, dz=0):
        """
        Computes the redshift distribution of sources. \
        Then, it shifts the distribution by "dz" (default dz=0).

        Kwargs:
            dz=0

        Returns:
            [z, nz] (Array)
        """
        if self.dndz is None:
            fn = f'{self.map_name}_dndz.npz'
            self.dndz = self._rerun_read_cycle(fn, 'NPZ', self._get_nz)
        return self._get_shifted_nz(dz)

    def _get_alpha(self):
        # Returns the ration between the \
        # sum of the weights of the data catalog \
        # and the weights of the random catalog.

        if self.alpha is None:
            w_data = self._get_w(mod='data')
            w_random = self._get_w(mod='random')
            self.alpha = np.sum(w_data)/np.sum(w_random)
        return self.alpha

    def _get_signal_map(self):
        delta_map = np.zeros(self.npix)
        cat_data = self.get_catalog(mod='data')
        cat_random = self.get_catalog(mod='random')
        w_data = self._get_w(mod='data')
        w_random = self._get_w(mod='random')
        alpha = self._get_alpha()
        nmap_data = get_map_from_points(cat_data, self.nside,
                                        w=w_data, rot=self.rot)
        nmap_random = get_map_from_points(cat_random, self.nside,
                                          w=w_random, rot=self.rot)
        mask = self.get_mask()
        goodpix = mask > 0
        delta_map = (nmap_data - alpha * nmap_random)
        delta_map[goodpix] /= mask[goodpix]
        return delta_map

    def _get_mask(self):
        # Calculates the mask of the SDSS \
        # data sets based on their random catalogs.

        cat_random = self.get_catalog(mod='random')
        w_random = self._get_w(mod='random')
        alpha = self._get_alpha()
        mask = get_map_from_points(cat_random, self.nside_mask,
                                   w=w_random, rot=self.rot)
        mask *= alpha
        # Account for different pixel areas
        area_ratio = (self.nside_mask/self.nside)**2
        mask = area_ratio * hp.ud_grade(mask,
                                        nside_out=self.nside)
        return mask

    def _get_nl_coupled(self):
        # Calculates the noise power spectrum \
        # for SDSS mappers. If the chosen resolution \
        # is below a 4096, the random catalog is used \
        # to compute the noise power spectrum. Otherwise, \
        # the high multipole tail of the signal power \
        # spectrum is used to estimate the noise power spectrum.

        if self.nside < self.nside_nl_threshold:
            print('calculing nl from weights')
            cat_data = self.get_catalog(mod='data')
            cat_random = self.get_catalog(mod='random')
            w_data = self._get_w(mod='data')
            w_random = self._get_w(mod='random')
            alpha = self._get_alpha()
            pixel_A = 4*np.pi/hp.nside2npix(self.nside)
            mask = self.get_mask()
            w2_data = get_map_from_points(cat_data, self.nside,
                                          w=w_data**2,
                                          rot=self.rot)
            w2_random = get_map_from_points(cat_random, self.nside,
                                            w=w_random**2,
                                            rot=self.rot)
            goodpix = mask > 0
            N_ell = (w2_data[goodpix].sum() +
                     alpha**2*w2_random[goodpix].sum())
            N_ell *= pixel_A**2/(4*np.pi)
            nl_coupled = N_ell * np.ones((1, 3*self.nside))
        else:
            print('calculating nl from mean cl values')
            f = self.get_nmt_field()
            cl = nmt.compute_coupled_cell(f, f)[0]
            N_ell = np.mean(cl[self.lmin_nl_from_data:2*self.nside])
            nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return {'nls': nl_coupled}

    def get_nl_coupled(self):
        """
        Calculates the noise power spectrum \
        for SDSS mappers. If the chosen resolution \
        is below a 4096, the random catalog is used \
        to compute the noise power spectrum. Otherwise, \
        the high multipole tail of the signal power \
        spectrum is used to estimate the noise power spectrum.

        Returns:
            nl_coupled (Array): coupled noise power spectrum
        """
        if self.nl_coupled is None:
            fn = '_'.join([f'{self.map_name}_Nell',
                           f'coord{self.coords}',
                           f'ns{self.nside}.npz'])
            d = self._rerun_read_cycle(fn, 'NPZ',
                                       self._get_nl_coupled)
            self.nl_coupled = d['nls']
        return self.nl_coupled

    def _bin_z(self, cat):
        # Removes all but the catalog sources
        # inside the chosen redshift bin.
        return cat[(cat['Z'] >= self.z_edges[0]) &
                   (cat['Z'] < self.z_edges[1])]

    def get_dtype(self):
        """
        Returns the data type of the field.
        Returns:
                dtype (str): data type of the field
        """
        return 'galaxy_density'

    def get_spin(self):
        """
        Returns the spin of the field.
        Returns:
                spin (int): spin of the field
        """
        return 0
