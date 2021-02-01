from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.table import Table, Column
import astropy.table
import numpy as np
import healpy as hp
import os


class MapperDESY1wl(MapperBase):
    def __init__(self, config):
        """
        Data source:
        https://des.ncsa.illinois.edu/releases/y1a1/key-catalogs/key-shape
        config - dict
          {'zbin_cat': 'y1_source_redshift_binning_v1.fits',
           'data_cat':  'mcal-y1a1-combined-riz-unblind-v4-matched.fits',
           'file_nz': '/.../.../y1_redshift_distributions_v1.fits'
           'nside': Nside,
           'zbin': zbin,
           'mask_name': name,
           'path_lite': path
           }
        """

        self._get_defaults(config)
        self.config = config
        self.path_lite = config.get('path_lite', None)
        self.mode = config.get('mode', 'shear')
        self.zbin = config['zbin']
        self.npix = hp.nside2npix(self.nside)

        # dn/dz
        self.nz = None 
        
        # load cat
        self.cat_data = self._load_catalog()
        self._remove_additive_bias()

        self.signal_map = None
        self.maps = {'PSF': None, 'shear': None}

        self.mask = None

        self.nl_coupled = None
        self.nls = {'PSF': None, 'shear': None}

    def _load_catalog(self):
        # Read catalogs
        # Columns explained in
        #
        # Galaxy catalog
        columns_data = ['coadd_objects_id', 'e1', 'e2',
                        'psf_e1', 'psf_e2', 'ra', 'dec',
                        'R11', 'R22', 'flags_select']
        # z-bin catalog
        columns_zbin = ['coadd_objects_id', 'zbin_mcal']

        read_lite, fname_lite = self._check_lite_exists(self.zbin)
        
        if read_lite:
            print('Loading lite bin{} cat'.format(self.zbin))
            self.cat_data = Table.read(fname_lite, memmap=True)
        else:
            print('Loading full cat')
            self.cat_data = Table.read(self.config['data_cat'],
                                       format='fits', memmap=True)
            self.cat_data.keep_columns(columns_data)
            cat_zbin = Table.read(self.config['zbin_cat'],
                                  format='fits', memmap=True)
            cat_zbin.keep_columns(columns_zbin)
            self.cat_data = astropy.table.join(self.cat_data, cat_zbin)
            # Note: By default join uses checks the values of the column named
            # the same in both tables
            col_w = Column(name='weight',
                           data=np.ones(len(self.cat_data), dtype=int))
            self.cat_data.add_column(col_w)

            # remove bins which are not the one of interest
            self.cat_data.remove_rows(self.cat_data['zbin_mcal'] != self.zbin)
            # filter for -90<dec<-35
            self.cat_data.remove_rows(self.cat_data['dec'] < -90)
            self.cat_data.remove_rows(self.cat_data['dec'] > -35)
            # remove flagged galaxies
            self.cat_data.remove_rows(self.cat_data['flags_select'] != 0)
            if fname_lite is not None:
                self.cat_data.write(fname_lite)
                
        return self.cat_data
    
    def _check_lite_exists(self, i):
        if self.path_lite is None:
            return False, None
        else:
            fname_lite = self.path_lite + f'DESwlMETACAL_catalog_lite_zbin{i}.fits'
            return os.path.isfile(fname_lite), fname_lite

    def _set_mode(self, mode=None):
        if mode is None:
            mode = self.mode

        if mode == 'shear':
            e1_flag = 'e1'
            e2_flag = 'e2'
        elif mode == 'PSF':
            e1_flag = 'psf_e1'
            e2_flag = 'psf_e2'
        else:
            raise ValueError(f"Unknown mode {mode}")
        return e1_flag, e2_flag, mode

    def _remove_additive_bias(self):
        self.cat_data['e1'] -= np.mean(self.cat_data['e1'])
        self.cat_data['e2'] -= np.mean(self.cat_data['e2'])
        return

    def get_signal_map(self, mode=None):
        e1f, e2f, mod = self._set_mode(mode)
        if self.maps[mod] is None:
            we1 = get_map_from_points(self.cat_data, self.nside,
                                      w=self.cat_data[e1f],
                                      ra_name='ra',
                                      dec_name='dec')
            we2 = get_map_from_points(self.cat_data, self.nside,
                                      w=self.cat_data[e1f],
                                      ra_name='ra',
                                      dec_name='dec')
            mask = self.get_mask()
            goodpix = mask > 0
            we1[goodpix] /= mask[goodpix]
            we2[goodpix] /= mask[goodpix]
            self.maps[mod] = [-we1, we2]

        self.signal_map = self.maps[mod]
        return self.signal_map

    def get_nz(self):
        if self.nz is None:
            self.nz = Table.read(self.config['file_nz'], format='fits',
                                 hdu=1)['Z_MID', 'BIN{}'.format(self.zbin + 1)]
        return self.nz

    def get_mask(self):
        if self.mask is None:
            self.mask = get_map_from_points(self.cat_data, self.nside,
                                            ra_name='ra', dec_name='dec')
        return self.mask

    def get_nl_coupled(self, mode=None):
        e1f, e2f, mod = self._set_mode(mode)
        if self.nls[mod] is None:
            w2s2 = get_map_from_points(self.cat_data, self.nside,
                                       w=0.5*(self.cat_data[e1f]**2 +
                                              self.cat_data[e2f]**2),
                                       ra_name='ra', dec_name='dec')
            N_ell = hp.nside2pixarea(self.nside) * np.sum(w2s2) / self.npix
            nl = N_ell * np.ones(3*self.nside)
            nl[:2] = 0  # Ylm = for l < spin
            self.nls[mod] = np.array([nl, 0*nl, 0*nl, nl])
        self.nl_coupled = self.nls[mod]
        return self.nl_coupled

    def get_dtype(self):
        return 'galaxy_shear'

    def get_spin(self):
        return 2
