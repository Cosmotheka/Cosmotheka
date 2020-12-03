from .mapper_base import MapperBase

from astropy.table import Table, Column
import astropy.table
import fitsio
import pandas as pd
import numpy as np
import healpy as hp
import pymaster as nmt
import os

class MapperDESwlMETACAL(MapperBase):
    def __init__(self, config):
        """
        config - dict
          {'zbin_cat': '/.../.../y1_source_redshift_binning_v1.fits',
           'data_cat':  '/.../.../mcal-y1a1-combined-riz-unblind-v4-matched.fits',
           'file_nz': '/.../.../y1_redshift_distributions_v1.fits'
           'nside': Nside,
           'bin': bin
           }
        """

        self.config = config
        self.cat_data = None

        self.bin = config['bin']
        self.nside = config['nside']
        self.npix = hp.nside2npix(self.nside)

        self.bin_edges = np.array([0.3, 0.43, 0.63, 0.9, 1.3])
        # Read catalogs
        # Columns explained in
        #
        # Galaxy catalog
        self.columns_data = ['coadd_objects_id', 'e1', 'e2', 'psf_e1', 'psf_e2', 'ra', 'dec',
                             'R11', 'R22', 'flags_select']
        # z-bin catalog
        self.columns_zbin = ['coadd_objects_id', 'zbin_mcal']

        # dn/dz
        self.nz = Table.read(config['file_nz'], format='fits',
                             hdu=1)['Z_MID', 'BIN{}'.format(self.bin + 1)]

        self.signal_map  = None
        self.psf_map     = None
        self.shear_map   = None
        self.mask        = None
        self.nmt_field   = None
        self.nl_coupled  = None
        self.shear_nl_coupled = None
        self.psf_nl_coupled   = None

    def _load_catalogs(self):
        if self.cat_data is not None:
            return self.cat_data

        fcat_lite = 'DESwlMETACAL_catalog_lite'
        fcat_bin = '{}_zbin{}.fits'.format(fcat_lite, self.bin)
        fcat_lite += '.fits'
        if os.path.isfile(fcat_bin):
            self.cat_data = Table.read(fcat_bin, memmep=True)
        elif os.path.isfile(fcat_lite):
            self.cat_data = Table.read(fcat_lite, memmep=True)
        else:
            self.cat_data = Table.read(self.config['data_cat'], format='fits', memmep=True)
            self.cat_data.keep_columns(self.columns_data)
            cat_zbin = Table.read(self.config['zbin_cat'], format='fits', memmep=True)
            cat_zbin.keep_columns(self.columns_zbin)
            self.cat_data = astropy.table.join(self.cat_data, cat_zbin)
            col_w = Column(name='weight', data=np.ones(len(self.cat_data), dtype=int))
            self.cat_data.add_column(col_w)
            # Note: By default join uses checks the values of the column named
            # the same in both tables
            self.cat_data.write(fcat_lite)

        self.cat_data.remove_rows(self.cat_data['zbin_mcal'] != self.bin + 1)
        self.cat_data.write(fcat_bin)

        return self.cat_data

    def _bin_z(self, cat):
        z_key = 'zbin_mcal'
        return cat[cat[z_key] == self.bin + 1]

    def _get_counts_map(self, w=None, nside=None):
        if nside is None:
            nside = self.nside

        phi = np.radians(self.cat_data['ra'])
        theta = np.radians(90 - self.cat_data['dec'])
        ipix = hp.ang2pix(self.nside, theta, phi)
        npix = hp.nside2npix(nside)

        numcount = np.bincount(ipix,  weights=w , minlength=npix )

    def get_signal_map(self, mode = None):
        if mode is None:
            mode  = self.mode

        if mode == 'shear':
            print('Calculating shear spin-2 field')
            self.signal_map = self._get_shear_map()
            return self.signal_map
        elif mode == 'PSF':
            print('Calculating PSF spin-2 field')
            self.signal_map = self._get_psf_map()
            return self.signal_map
        else:
            print('Unrecognized mode. Please choose between: shear or PSF')
            return

    def _get_shear_map(self):
        if self.shear_map is None:
            we1 = self._get_counts_map(w=self.cat_data['e1'], nside=None)
            we2 = self._get_counts_map(w=self.cat_data['e2'], nside=None)

            mask = self._get_mask()
            goodpix  = mask > 0
            we1[goodpix] /= self._get_galaxy_mask()[goodpix]
            we2[goodpix] /= self._get_galaxy_mask()[goodpix]

            self.shear_map = [we1, we2]
        return self.shear_map

    def _get_psf_map(self):
        if self.psf_map is None:
            we1 = self._get_counts_map(w=self.cat_data['psf_e1'], nside=None)
            we2 = self._get_counts_map(w=self.cat_data['psf_e2'], nside=None)

            mask = self._get_mask()
            goodpix  = mask > 0
            we1[goodpix] /= self._get_galaxy_mask()[goodpix]
            we2[goodpix] /= self._get_galaxy_mask()[goodpix]

            self.psf_map = [we1, we2]
        return self.psf_map

    def get_mask(self):
        if self.mask is None:
            self.mask = self._get_counts_map()
        return self.mask

    def get_nmt_field(self, mode = None):
        if mode is None:
            mode = self.mode
        signal = self.get_signal_map(mode = mode)
        mask = self.get_mask()
        self.nmt_field = nmt.NmtField(mask, signal, n_iter = 0)
        return self.nmt_field

    def get_nl_coupled(self, mode = None):
        if  mode == 'shear':
            print('Calculating shear nl coupled')
            self.nl_coupled = self._get_shear_nl_coupled()
            return self.nl_coupled
        elif mode == 'PSF':
            print('Calculating psf nl coupled')
            self.nl_coupled = self._get_psf_nl_coupled()
            return self.nl_coupled
        else:
            print('Unrecognized mode. Please choose between: shear or PSF')
            return

    def _get_shear_nl_coupled(self):
        if self.shear_nl_coupled is None:
            w2s2 = self._get_counts_map( w = 0.5 * (self.cat_data['e1']**2 + self.cat_data['e2']**2), nside=None)
            N_ell = hp.nside2pixarea(self.nside) * np.sum(w2s2) / self.npix
            nl = N_ell * np.ones(3*self.nside)
            nl[:2] = 0  # Ylm = for l < spin
            self.shear_nl_coupled = np.array([nl, 0 * nl, 0 * nl, nl])

        return self.shear_nl_coupled

    def _get_psf_nl_coupled(self):
        if self.psf_nl_coupled is None:
            w2s2 = self._get_counts_map(w = 0.5 * (self.cat_data['psf_e1']**2 + self.cat_data['psf_e2']**2), nside=None)
            N_ell = hp.nside2pixarea(self.nside) * np.sum(w2s2) / self.npix
            nl = N_ell * np.ones(3*self.nside)
            nl[:2] = 0  # Ylm = for l < spin
            self.psf_nl_coupled = np.array([nl, 0 * nl, 0 * nl, nl])

        return self.psf_nl_coupled