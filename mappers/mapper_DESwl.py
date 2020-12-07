from mapper_base import MapperBase

from astropy.table import Table, Column
import astropy.table
import pandas as pd
import numpy as np
import healpy as hp
import pymaster as nmt
import os

class MapperDESwl(MapperBase):
    def __init__(self, config):
        """
        config - dict
          {'zbin_cat': '/.../.../y1_source_redshift_binning_v1.fits',
           'data_cat':  '/.../.../mcal-y1a1-combined-riz-unblind-v4-matched.fits',
           'file_nz': '/.../.../y1_redshift_distributions_v1.fits'
           'nside': Nside,
           'bin': bin,
           'mask_name': name
           }
        """

        self.config = config
        self.mask_name = self.config['mask_name']
        
        self.bin = config['bin']
        self.nside = config['nside']
        self.npix = hp.nside2npix(self.nside)

        self.bin_edges = np.array([0.3, 0.43, 0.63, 0.9, 1.3])
        
        # load cat
        self._load_catalog()
        
        #bin data 
        self.cat_data = self._bin_z()
        print('Data binned:' + len(self.cat_data['ra']))

        # dn/dz
        self.nz = Table.read(config['file_nz'], format='fits',hdu=1).to_pandas()
        self.nz = self.nz['Z_MID', 'BIN{}'.format(self.bin + 1)]

        self.signal_map  = None
        self.psf_map     = None
        self.shear_map   = None
        self.mask        = None
        self.nmt_field   = None
        self.nl_coupled  = None
        self.shear_nl_coupled = None
        self.psf_nl_coupled  = None
        

    def _load_catalog(self):
        # Read catalogs
        # Columns explained in
        #
        # Galaxy catalog
        columns_data = ['coadd_objects_id', 'e1', 'e2', 'psf_e1', 'psf_e2', 'ra', 'dec',
                             'R11', 'R22', 'flags_select']
        # z-bin catalog
        columns_zbin = ['coadd_objects_id', 'zbin_mcal']

        fcat_lite = 'DESwlMETACAL_catalog_lite'
        fcat_bin = fcat_lite + '_zbin.pkl'
        fcat_lite += '.pkl'
        
        if os.path.isfile(fcat_bin):
            print('Loading lite z_bin')
            self.cat_zbin = Table.read(fcat_bin, memmap=True)
        else:
            print('Lite zbins not found, loading original and turning them into lite')
            self.cat_zbin = Table.read(self.config['zbin_cat'], format='fits', memmap=True).to_pandas()
            self.cat_zbin = self.cat_zbin[columns_zbin]
            print('Loaded zbin cats')
            self.cat_zbin.to_pickle(fcat_bin)
            print('Saved lite zbin cats')
            
        if os.path.isfile(fcat_lite):
            print('Loading lite cat')
            self.cat_data = Table.read(fcat_lite, memmap=True)
        else:
            print('Lite cats not found, loading original and turning them into lite')
            self.cat_data = Table.read(self.config['data_cat'], format='fits', memmap=True).to_pandas()
            self.cat_data = self.cat_data[self.column_data]
            print('Loaded data cats')
            self.cat_data.to_pickle(fcat_lite)
            print('Saved lite data cats')      

        return 
    
    def _bin_z(self):
        self.cat_data = pd.concat([self.cat_data, self.cat_zbin], axis=1, sort=False)
        return self.cat_data[self.cat_data.zbin_mcal != self.bin + 1]

    def _get_counts_map(self, w=None, nside=None):
        if nside is None:
            nside = self.nside

        phi = np.radians(self.cat_data['ra'])
        theta = np.radians(90 - self.cat_data['dec'])
        ipix = hp.ang2pix(self.nside, theta, phi)
        npix = hp.nside2npix(nside)

        numcount = np.bincount(ipix,  weights=w , minlength=npix )
        
        return numcount

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
