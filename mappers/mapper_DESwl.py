from .mapper_base import MapperBase

from astropy.table import Table
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

        self.bin = config['bin']
        self.nside = config['nside']
        self.npix = hp.nside2npix(self.nside)

        self.bin_edges = np.array([0.3, 0.43, 0.63, 0.9, 1.3])
        # Read catalogs
        # Columns explained in
        #
        # Galaxy catalog
        self.columns_data = ['e1', 'e2', 'psf_e1', 'psf_e2', 'ra', 'dec',
                             'R11', 'R22', 'flags_select']
        # z-bin catalog
        self.columns_zbin = ['coadd_objects_id', 'zbin_mcal']

        # dn/dz
        self.nz = Table.read(config['file_nz'], format='fits',
                             hdu=1)['Z_MID', 'BIN{}'.format(self.bin + 1)]

        self.cat_data = None
        self.cat_zbin = None
        self.weights = None
        self.mask = None
        self.dndz = None
        self.sh_maps = None
        self.nl_coupled = None
        self.nmt_fields = None

    def _load_catalogs(self):
        if self.cat_data is None:
            self.cat_data = Table.read(self.config['data_cat'], format='fits', memmep=True)
            self.cat_data.keep_columns(self.columns_data)

            self.cat_zbin = Table.read(self.config['zbin_cat'], format='fits', memmep=True)
            self.cat_zbin.keep_columns(self.columns_zbin)

            self.weights = np.ones(len(self.cat_data))
        return self.cat_data, self.cat_zbin, self.weights

    def _get_counts_map(self):
        nside = self.nside
        phi = np.radians(self.cat_data['DEC'])
        theta = np.radians(90 - self.cat_data['RA'])

        ipix = hp.ang2pix(self.nside, theta, phi)

        return numcount

    def get_mask(self):
        goodpix = self.mask > 0.5
        self.mask[~goodpix] = 0
        self.mask = hp.ud_grade(self.mask, nside_out=self.nside)
        return self.mask

    def get_nz(self, num_z=200):
        if self.dndz is None:
            #equivalent to getting columns 1 and 3 in previous code
            z  = self.nz['Z_MID']
            pz = self.nz['BIN%d' % (self.bin)]
            # Calculate z bias
            dz = 0
            z_dz = z - dz
            # Set to 0 points where z_dz < 0:
            sel = z_dz >= 0
            z_dz = z_dz[sel]
            pz = pz[sel]
        return np.array([z_dz, pz])

    def get_signal_map(self):
        if self.delta_map is None:
            self.delta_map = np.zeros(self.npix)
            N_mean = np.sum(self.nmap_data)/np.sum(self.mask)
            goodpix = self.mask > 0
            self.delta_map[goodpix] = (self.nmap_data[goodpix])/(self.mask[goodpix]*N_mean) -1
        return [self.delta_map]

    def get_nmt_field(self):
        if self.nmt_field is None:
            signal = self.get_signal_map()
            mask = self.get_mask()
            self.nmt_field = nmt.NmtField(mask, signal, n_iter = 0)
        return self.nmt_field

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            N_mean = np.sum(self.nmap_data)/np.sum(self.mask)
            N_mean_srad = N_mean / (4 * np.pi) * self.npix
            correction = np.sum(self.w_data**2) / np.sum(self.w_data)
            N_ell = correction * np.sum(self.mask) / self.npix / N_mean_srad
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled
