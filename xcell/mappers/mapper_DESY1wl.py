from .utils import get_map_from_points
from .mapper_base import MapperBase
from astropy.table import Table, hstack
import numpy as np
import healpy as hp


class MapperDESY1wl(MapperBase):
    """
    Note that last letter of the the mask name stands for the \
    chosen redshdift bin (`i = [1,2,3,4]`).

    path = `'.../Datasets/DES_Y1/shear_catalog/'`

    ***Config***

        - zbin: `0` / `1` / `2` /`3`
        - mode: `shear` / `PSF`
        - zbin_cat: `path+'y1_source_redshift_binning_v1.fits'`
        - data_cat: `path+'mcal-y1a1-combined-riz-unblind-v4-matched.fits'`
        - file_nz: `path+'y1_redshift_distributions_v1.fits'`
        - path_rerun: `'.../Datasets/DES_Y1/xcell_reruns/'`
        - mask_name: `'mask_DESY1wli'`
        - mapper_class: `'MapperDESY1wl'`
    """
    def __init__(self, config):
        self._get_defaults(config)
        self.config = config
        self.rot = self._get_rotator('C')
        self.mode = config.get('mode', 'shear')
        self.zbin = config['zbin']
        self.npix = hp.nside2npix(self.nside)
        # dn/dz
        self.dndz = None
        # load cat
        self.cat_data = None
        # get items for calibration
        self.Rs = None

        self.signal_map = None
        self.maps = {'PSF': None, 'shear': None}

        self.nl_coupled = None
        self.nls = {'PSF': None, 'shear': None}

    def get_catalog(self):
        """
        Returns the chosen redshift bin of the \
        mappers catalog after removing the \
        additive and multiplicative bias.

        Returns:
            cat_data (Array)
        """
        if self.cat_data is None:
            # load cat
            self.cat_data = self._load_catalog()
            # get items for calibration
            self.Rs = self._get_Rs()
            # clean data
            self.cat_data.remove_rows(self.cat_data['zbin_mcal'] != self.zbin)
            # calibrate
            self._remove_additive_bias()
            self._remove_multiplicative_bias()

        return self.cat_data

    def _load_catalog_from_raw(self):
        # Loads the raw DESY1 catalog and \
        # produces a lite version only with the \
        # columns of interest and after applying \
        # a -90 < DEC < 35 filter.

        # Read catalogs
        # Columns explained in
        #
        # Galaxy catalog
        columns_data = ['coadd_objects_id', 'e1', 'e2',
                        'psf_e1', 'psf_e2', 'ra', 'dec',
                        'R11', 'R12', 'R21', 'R22',
                        'flags_select']
        # z-bin catalog
        columns_zbin = ['zbin_mcal', 'zbin_mcal_1p',
                        'zbin_mcal_1m', 'zbin_mcal_2p', 'zbin_mcal_2m']
        print('Loading full cat')
        cat = Table.read(self.config['data_cat'],
                         format='fits', memmap=True)
        cat.keep_columns(columns_data)
        cat_zbin = Table.read(self.config['zbin_cat'],
                              format='fits', memmap=True)
        cat_zbin.keep_columns(columns_zbin)
        cat = hstack([cat, cat_zbin])

        # remove bins which are not the one of interest
        # Logic: If item in one of zbins --> sel = False
        # Thus it is not removed by next line
        sel = (cat['zbin_mcal'] != self.zbin) * \
            (cat['zbin_mcal_1p'] != self.zbin) * \
            (cat['zbin_mcal_1m'] != self.zbin) * \
            (cat['zbin_mcal_2p'] != self.zbin) * \
            (cat['zbin_mcal_2m'] != self.zbin)
        cat.remove_rows(sel)
        # filter for -90<dec<-35
        cat.remove_rows(cat['dec'] < -90)
        cat.remove_rows(cat['dec'] > -35)
        # remove flagged galaxies
        cat.remove_rows(cat['flags_select'] != 0)
        return cat.as_array()

    def _load_catalog(self):
        # Loads the lite DESY1 catalog.

        fn = f'DESY1wl_catalog_rerun_bin{self.zbin}.fits'
        cat = self._rerun_read_cycle(fn, 'FITSTable',
                                     self._load_catalog_from_raw)
        return Table(cat)

    def _set_mode(self, mode=None):
        # Given the chose mapper mode ('shear' or 'PSF'), \
        # it returns the corresponding name of the \
        # ellipticity fields in the catalog.

        # Kwargs:
        #    mode=None

        # Returns:
        #     e1_flag (String), e2_flag (String), mode (String)

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

    def _get_Rs(self):
        # Computes the R factor used to calculate \
        # the multiplicative bias of the maps.

        if self.Rs is None:
            data_1p = self.cat_data[self.cat_data['zbin_mcal_1p'] == self.zbin]
            data_1m = self.cat_data[self.cat_data['zbin_mcal_1m'] == self.zbin]
            data_2p = self.cat_data[self.cat_data['zbin_mcal_2p'] == self.zbin]
            data_2m = self.cat_data[self.cat_data['zbin_mcal_2m'] == self.zbin]

            mean_e1_1p = np.mean(data_1p['e1'])
            mean_e2_1p = np.mean(data_1p['e2'])
            mean_e1_1m = np.mean(data_1m['e1'])
            mean_e2_1m = np.mean(data_1m['e2'])
            mean_e1_2p = np.mean(data_2p['e1'])
            mean_e2_2p = np.mean(data_2p['e2'])
            mean_e1_2m = np.mean(data_2m['e1'])
            mean_e2_2m = np.mean(data_2m['e2'])

            self.Rs = np.array([[(mean_e1_1p-mean_e1_1m)/0.02,
                                 (mean_e1_2p-mean_e1_2m)/0.02],
                                [(mean_e2_1p-mean_e2_1m)/0.02,
                                 (mean_e2_2p-mean_e2_2m)/0.02]])
        return self.Rs

    def _remove_additive_bias(self):
        self.cat_data['e1'] -= np.mean(self.cat_data['e1'])
        self.cat_data['e2'] -= np.mean(self.cat_data['e2'])
        return

    def _remove_multiplicative_bias(self):
        # Should be done only with galaxies truly in zbin
        Rg = np.array([[np.mean(self.cat_data['R11']),
                        np.mean(self.cat_data['R12'])],
                       [np.mean(self.cat_data['R21']),
                        np.mean(self.cat_data['R22'])]])
        Rmat = Rg + self.Rs
        one_plus_m = np.sum(np.diag(Rmat))*0.5

        self.cat_data['e1'] /= one_plus_m
        self.cat_data['e2'] /= one_plus_m
        return

    def _get_ellipticity_maps(self, mode=None):
        # Returns the ellipticity maps of the \
        # chose catalog ('shear' or 'PSF').

        e1f, e2f, mod = self._set_mode(mode)
        print('Computing bin{} signal map'.format(self.zbin))
        cat_data = self.get_catalog()
        we1, we2 = get_map_from_points(cat_data, self.nside,
                                       qu=[-cat_data[e1f], cat_data[e2f]],
                                       ra_name='ra',
                                       dec_name='dec',
                                       rot=self.rot)
        mask = self.get_mask()
        goodpix = mask > 0
        we1[goodpix] /= mask[goodpix]
        we2[goodpix] /= mask[goodpix]
        return we1, we2

    def get_signal_map(self, mode=None):
        e1f, e2f, mod = self._set_mode(mode)
        if self.maps[mod] is not None:
            self.signal_map = self.maps[mod]
            return self.signal_map

        # This will only be computed if self.maps['mod'] is None
        def get_ellip_maps():
            return self._get_ellipticity_maps(mode=mode)

        fn = '_'.join([f'DESY1wl_signal_map_{mod}_bin{self.zbin}',
                       f'coord{self.coords}',
                       f'ns{self.nside}.fits.gz'])
        d = self._rerun_read_cycle(fn, 'FITSMap', get_ellip_maps,
                                   section=[0, 1])
        self.maps[mod] = np.array([d[0], d[1]])
        self.signal_map = self.maps[mod]
        return self.signal_map

    def get_nz(self, dz=0):
        """
        Returns the mappers redshift \
        distribtuion of sources from a file.

        Kwargs:
            dz=0

        Returns:
            [z, nz] (Array)
        """
        if self.dndz is None:
            f = Table.read(self.config['file_nz'], format='fits',
                           hdu=1)['Z_MID', 'BIN{}'.format(self.zbin + 1)]
            self.dndz = {'z_mid': f['Z_MID'],
                         'nz': f['BIN{}'.format(self.zbin + 1)]}
        return self._get_shifted_nz(dz)

    def _get_mask(self):
        # Returns the mapper's mask after applying. \
        # the mapper's threshold.

        cat_data = self.get_catalog()
        msk = get_map_from_points(cat_data, self.nside,
                                  ra_name='ra', dec_name='dec',
                                  rot=self.rot)
        return msk

    def get_nl_coupled(self, mode=None):
        e1f, e2f, mod = self._set_mode(mode)
        if self.nls[mod] is not None:
            self.nl_coupled = self.nls[mod]
            return self.nl_coupled

        # This will only be computed if self.nls['mod'] is None
        def get_w2s2():
            e1f, e2f, mod = self._set_mode(mode)
            cat_data = self.get_catalog()
            mp = get_map_from_points(cat_data, self.nside,
                                     w=0.5*(cat_data[e1f]**2 +
                                            cat_data[e2f]**2),
                                     ra_name='ra', dec_name='dec',
                                     rot=self.rot)
            return mp

        fn = '_'.join([f'DESY1wl_{mod}_w2s2_bin{self.zbin}',
                       f'coord{self.coords}',
                       f'ns{self.nside}.fits.gz'])
        w2s2 = self._rerun_read_cycle(fn, 'FITSMap', get_w2s2)

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
