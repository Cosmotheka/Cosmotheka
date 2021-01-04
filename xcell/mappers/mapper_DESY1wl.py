from mapper_base import MapperBase
from astropy.table import Table, Column
from utils import get_map_from_points
import astropy.table
import numpy as np
import healpy as hp
import pymaster as nmt
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
           'bin': bin,
           'mask_name': name,
           'path_lite': path
           }
        """

        self.config = config
        self.path_lite = config.get('path_lite', None)
        self.mode = config.get('mode', 'shear')
        self.mask_name = config.get('mask_name', None)
        self.bin = config['bin']
        self.nside = config['nside']
        self.npix = hp.nside2npix(self.nside)
        self.bin_edges = np.array([0.3, 0.43, 0.63, 0.9, 1.3])

        # dn/dz
        self.nz = Table.read(config['file_nz'], format='fits',
                             hdu=1)['Z_MID', 'BIN{}'.format(self.bin + 1)]

        # load cat
        self.cat_data = self._load_catalog()
        self._remove_additive_bias()

        self.signal_map = None
        self.maps = {'PSF': None, 'shear': None}

        self.mask = None

        self.nl_coupled = None
        self.nls = {'PSF': None, 'shear': None}
        
        self.nmt_field = None

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

        fcat_lite = 'DESwlMETACAL_catalog_lite'
        fcat_bin = '{}_zbin{}.fits'.format(fcat_lite, self.bin)
        fcat_lite += '.fits'
        
        #Try with david cats
        #fcat_bin = 'catalog_metacal_bin{}_zbin_mcal.fits'.format(self.bin)
        
        if os.path.isfile(self.path_lite + fcat_bin):
            print('Loading lite bin{} cat'.format(self.bin))
            self.cat_data = Table.read(self.path_lite + fcat_bin, memmap=True)
        elif os.path.isfile(self.path_lite + fcat_lite):
            print('loading full lite cat')
            self.cat_data = Table.read(self.path_lite + fcat_lite, memmap=True)
        else:
            print('loading full cat')
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
            self.cat_data.write(fcat_lite)
            
            #remove bins which are not the one of interest
            self.cat_data.remove_rows(self.cat_data['zbin_mcal'] != self.bin)
            #filter for -90<dec<-35
            self.cat_data.remove_rows(self.cat_data['dec'] < -90)
            self.cat_data.remove_rows(self.cat_data['dec'] > -35)
            #remove flagged galaxies
            self.cat_data.remove_rows(self.cat_data['flags_select'] != 0)
            
            self.cat_data.write(fcat_bin)

        return self.cat_data
    
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
        self.cat_data['e1']  = self.cat_data['e1'] - np.mean(self.cat_data['e1'])
        self.cat_data['e2']  = self.cat_data['e2'] - np.mean(self.cat_data['e2'])
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