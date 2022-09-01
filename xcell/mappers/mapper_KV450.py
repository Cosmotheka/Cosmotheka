from .mapper_base import MapperBase
from .utils import get_map_from_points, save_rerun_data
from astropy.table import Table, vstack
import numpy as np
import healpy as hp


class MapperKV450(MapperBase):
    """
    Mapper for the KiDS-1000 weak lensing data set. \
    """
    def __init__(self, config):
        """
        config - dict
          {'data_catalogs': ['KV450_G12_reweight_3x4x4_v2_good.cat',
                             'KV450_G23_reweight_3x4x4_v2_good.cat',
                             'KV450_GS_reweight_3x4x4_v2_good.cat',
                             'KV450_G15_reweight_3x4x4_v2_good.cat',
                             'KV450_G9_reweight_3x4x4_v2_good.cat'] ,
          'file_nz': Nz_DIR_z0.1t0.3.asc,
          'zbin':0,
          'nside':nside,
          'mask_name': 'mask_KV450_0'}
        """

        self._get_defaults(config)
        self.rot = self._get_rotator('C')

        self.column_names = ['SG_FLAG', 'GAAP_Flag_ugriZYJHKs',
                             'Z_B', 'Z_B_MIN', 'Z_B_MAX',
                             'ALPHA_J2000', 'DELTA_J2000', 'PSF_e1', 'PSF_e2',
                             'bias_corrected_e1', 'bias_corrected_e2',
                             'weight']

        self.mode = config.get('mode', 'shear')
        self.zbin_edges = np.array([[0.1, 0.3],
                                    [0.3, 0.5],
                                    [0.5, 0.7],
                                    [0.7, 0.9],
                                    [0.9, 1.2]])
        self.npix = hp.nside2npix(self.nside)
        self.zbin = int(config['zbin'])
        self.z_edges = self.zbin_edges[self.zbin]
        # Multiplicative bias
        # Values from Table 2 of 1812.06076 (KV450 cosmo paper)
        self.m = (-0.017, -0.008, -0.015, 0.010, 0.006)

        self.cat_data = None

        self.w2s2 = None
        self.w2s2s = {'PSF': None, 'shear': None, 'stars': None}

        self.dndz = None
        self.sel = {'galaxies': 1, 'stars': 0}

        self.signal_map = None
        self.maps = {'PSF': None, 'shear': None, 'stars': None}

        self.masks = {'stars': None, 'galaxies': None}

        self.nl_coupled = None
        self.nls = {'PSF': None, 'shear': None, 'stars': None}

    def get_catalog(self):
        """
        Returns the chosen redshift bin of the \
        mappers catalog.

        Returns:
            cat_data (Array)
        """
        if self.cat_data is None:
            fn = f'KV450_cat_bin{self.zbin}.fits'
            self.cat_data = self._rerun_read_cycle(fn, 'FITSTable',
                                                   self._load_catalog,
                                                   saved_by_func=True)
        return self.cat_data

    def _load_catalog(self):
        # Loads the lite DESY1 catalog. \
        # Selects the chosen bin in the catalog. \
        # Removes the additive and multiplicative  biases.

        nzbins = self.zbin_edges.shape[0]
        cat_bins = [Table() for i in range(nzbins)]

        for file_data in self.config['data_catalogs']:
            cat_field = Table.read(file_data, format='fits')[self.column_names]
            # GAAP cut
            goodgaap = cat_field['GAAP_Flag_ugriZYJHKs'] == 0
            cat_field = cat_field[goodgaap]
            # Binning
            for ibin in range(nzbins):
                sel = self._bin_z(cat_field, ibin)
                cat_field_bin = cat_field[sel]
                self._remove_additive_bias(cat_field_bin)
                cat_bins[ibin] = vstack([cat_bins[ibin], cat_field_bin])

        for ibin, cat in enumerate(cat_bins):
            self._remove_multiplicative_bias(cat, ibin)
            fn = f'KV450_cat_bin{ibin}.fits'
            save_rerun_data(self, fn, 'FITSTable', cat.as_array())
        return cat_bins[self.zbin].as_array()

    def _set_mode(self):
        # Given the chosen mapper mode ('shear', 'PSF' and 'stars), \
        # it returns the kind of the map associated \
        # ('shear', 'PSF' --> 'galaxy' and 'stars' --> stars) \
        # with the mode and corresponding name of the \
        # ellipticity fields in the catalog.

        # Returns:
        #     kind (String), e1_flag (String), \
        #     e2_flag (String), mode (String)

        mode = self.mode

        if mode == 'shear':
            kind = 'galaxies'
            e1_flag = 'bias_corrected_e1'
            e2_flag = 'bias_corrected_e2'
        elif mode == 'PSF':
            kind = 'galaxies'
            e1_flag = 'PSF_e1'
            e2_flag = 'PSF_e2'
        elif mode == 'stars':
            kind = 'stars'
            e1_flag = 'bias_corrected_e1'
            e2_flag = 'bias_corrected_e2'
        else:
            raise ValueError(f"Unknown mode {mode}")
        return kind, e1_flag, e2_flag, mode

    def _bin_z(self, cat, zbin):
        # Removes all sources in the catalog \
        # outside the chosen redshift bin.

        z_key = 'Z_B'
        z_edges = self.zbin_edges[zbin]
        return ((cat[z_key] > z_edges[0]) &
                (cat[z_key] <= z_edges[1]))

    def _remove_additive_bias(self, cat):
        sel_gals = cat['SG_FLAG'] == 1
        if np.any(sel_gals):
            e1mean = np.average(cat['bias_corrected_e1'][sel_gals],
                                weights=cat['weight'][sel_gals])
            e2mean = np.average(cat['bias_corrected_e2'][sel_gals],
                                weights=cat['weight'][sel_gals])
            cat['bias_corrected_e1'][sel_gals] -= e1mean
            cat['bias_corrected_e2'][sel_gals] -= e2mean

    def _remove_multiplicative_bias(self, cat_data, zbin):
        sel_gals = cat_data['SG_FLAG'] == 1
        cat_data['bias_corrected_e1'][sel_gals] /= 1 + self.m[zbin]
        cat_data['bias_corrected_e2'][sel_gals] /= 1 + self.m[zbin]

    def _get_gals_or_stars(self, kind='galaxies'):
        # Returns the mapper's chosen catalog.

        cat_data = self.get_catalog()
        sel = cat_data['SG_FLAG'] == self.sel[kind]
        return cat_data[sel]

    def _get_ellip_maps(self):
        # Returns the ellipticity fields of the mapper's catalog.

        kind, e1f, e2f, mod = self._set_mode()
        print('Computing bin{} signal map'.format(self.zbin))
        data = self._get_gals_or_stars(kind)
        we1, we2 = get_map_from_points(data, self.nside,
                                       w=data['weight'],
                                       qu=[-data[e1f], data[e2f]],
                                       ra_name='ALPHA_J2000',
                                       dec_name='DELTA_J2000',
                                       rot=self.rot)
        mask = self.get_mask()
        goodpix = mask > 0
        we1[goodpix] /= mask[goodpix]
        we2[goodpix] /= mask[goodpix]
        return we1, we2

    def get_signal_map(self):
        kind, e1f, e2f, mod = self._set_mode()
        if self.maps[mod] is not None:
            self.signal_map = self.maps[mod]
            return self.signal_map

        fn = '_'.join([f'KV450_signal_{mod}_bin{self.zbin}',
                       f'coord{self.coords}',
                       f'ns{self.nside}.fits.gz'])
        d = self._rerun_read_cycle(fn, 'FITSMap',
                                   self._get_ellip_maps,
                                   section=[0, 1])
        self.maps[mod] = np.array([d[0], d[1]])
        self.signal_map = self.maps[mod]
        return self.signal_map

    def _get_mask(self):
        kind, e1f, e2f, mod = self._set_mode()
        if self.masks[kind] is not None:
            return self.masks[kind]

        data = self._get_gals_or_stars(kind)
        msk = get_map_from_points(data, self.nside,
                                  w=data['weight'],
                                  ra_name='ALPHA_J2000',
                                  dec_name='DELTA_J2000',
                                  rot=self.rot)
        self.masks[kind] = msk
        return msk

    def _get_w2s2(self):
        # Computes the weight-squared map for
        # noise power spectrum estimation.

        kind, e1f, e2f, mod = self._set_mode()
        if self.w2s2s[mod] is not None:
            self.w2s2 = self.w2s2s[mod]
            return self.w2s2

        def get_w2s2():
            data = self._get_gals_or_stars(kind)
            wcol = data['weight']**2*0.5*(data[e1f]**2+data[e2f]**2)
            w2s2 = get_map_from_points(data, self.nside, w=wcol,
                                       ra_name='ALPHA_J2000',
                                       dec_name='DELTA_J2000',
                                       rot=self.rot)
            return w2s2

        fn = '_'.join([f'KV450_w2s2_{kind}_bin{self.zbin}',
                       f'coord{self.coords}',
                       f'ns{self.nside}.fits.gz'])
        self.w2s2s[mod] = self._rerun_read_cycle(fn, 'FITSMap',
                                                 get_w2s2)
        self.w2s2 = self.w2s2s[mod]
        return self.w2s2

    def get_nl_coupled(self):
        kind, e1f, e2f, mod = self._set_mode()
        if self.nls[mod] is None:
            self.w2s2 = self._get_w2s2()
            N_ell = hp.nside2pixarea(self.nside) * np.mean(self.w2s2)
            nl = N_ell * np.ones(3*self.nside)
            nl[:2] = 0  # ylm = 0 for l < spin
            self.nls[mod] = np.array([nl, 0*nl, 0*nl, nl])
        self.nl_coupled = self.nls[mod]
        return self.nl_coupled

    def get_nz(self, dz=0):
        """
        Loads the redshift distribution of sources \
        from the data products.
        Then, it shifts the distribution by "dz" (default dz=0). \
        Finally, it returns the redshift distribtuion.

        Kwargs:
            dz=0

        Returns:
            [z, nz] (Array)
        """
        if self.dndz is None:
            z, nz = np.loadtxt(self.config['file_nz'], unpack=True)
            self.dndz = {'z_mid': z, 'nz': nz}
        return self._get_shifted_nz(dz)

    def get_dtype(self):
        return 'galaxy_shear'

    def get_spin(self):
        return 2
