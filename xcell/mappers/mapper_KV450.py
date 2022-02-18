from .mapper_base import MapperBase
from .utils import get_map_from_points, save_rerun_data
from astropy.table import Table, vstack
import numpy as np
import healpy as hp


class MapperKV450(MapperBase):
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

        self.mask = None
        self.masks = {'stars': None, 'galaxies': None}

        self.nl_coupled = None
        self.nls = {'PSF': None, 'shear': None, 'stars': None}

    def get_catalog(self):
        if self.cat_data is None:
            fn = f'KV450_cat_bin{self.zbin}.fits'
            self.cat_data = self._rerun_read_cycle(fn, 'FITSTable',
                                                   self._load_catalog,
                                                   saved_by_func=True)
        return self.cat_data

    def _load_catalog(self):
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

    def _set_mode(self, mode=None):
        if mode is None:
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
        cat_data = self.get_catalog()
        sel = cat_data['SG_FLAG'] == self.sel[kind]
        return cat_data[sel]

    def _get_ellip_maps(self, mode=None):
        kind, e1f, e2f, mod = self._set_mode(mode)
        print('Computing bin{} signal map'.format(self.zbin))
        data = self._get_gals_or_stars(kind)
        wcol = data['weight']*data[e1f]
        we1 = get_map_from_points(data, self.nside, w=wcol,
                                  ra_name='ALPHA_J2000',
                                  dec_name='DELTA_J2000')
        wcol = data['weight']*data[e2f]
        we2 = get_map_from_points(data, self.nside, w=wcol,
                                  ra_name='ALPHA_J2000',
                                  dec_name='DELTA_J2000')
        mask = self.get_mask(mod)
        goodpix = mask > 0
        we1[goodpix] /= mask[goodpix]
        we2[goodpix] /= mask[goodpix]
        return we1, we2

    def get_signal_map(self, mode=None):
        kind, e1f, e2f, mod = self._set_mode(mode)
        if self.maps[mod] is not None:
            self.signal_map = self.maps[mod]
            return self.signal_map

        # This will only be computed if self.maps['mod'] is None
        def get_ellip_maps_mod():
            return self._get_ellip_maps(mode=mode)

        fn = f'KV450_signal_{mod}_bin{self.zbin}_ns{self.nside}.fits.gz'
        d = self._rerun_read_cycle(fn, 'FITSMap',
                                   get_ellip_maps_mod,
                                   section=[0, 1])
        self.maps[mod] = np.array([-d[0], d[1]])
        self.signal_map = self.maps[mod]
        return self.signal_map

    def get_mask(self, mode=None):
        kind, e1f, e2f, mod = self._set_mode(mode)
        if self.masks[kind] is not None:
            self.mask = self.masks[kind]
            return self.mask

        def get_mask_mod():
            data = self._get_gals_or_stars(kind)
            msk = get_map_from_points(data, self.nside,
                                      w=data['weight'],
                                      ra_name='ALPHA_J2000',
                                      dec_name='DELTA_J2000')
            return msk

        fn = f'KV450_mask_{kind}_bin{self.zbin}_ns{self.nside}.fits.gz'
        self.masks[kind] = self._rerun_read_cycle(fn, 'FITSMap',
                                                  get_mask_mod)
        self.mask = self.masks[kind]
        return self.mask

    def _get_w2s2(self, mode):
        kind, e1f, e2f, mod = self._set_mode(mode)
        if self.w2s2s[mod] is not None:
            self.w2s2 = self.w2s2s[mod]
            return self.w2s2

        def get_w2s2():
            data = self._get_gals_or_stars(kind)
            wcol = data['weight']**2*0.5*(data[e1f]**2+data[e2f]**2)
            w2s2 = get_map_from_points(data, self.nside, w=wcol,
                                       ra_name='ALPHA_J2000',
                                       dec_name='DELTA_J2000')
            return w2s2

        fn = f'KV450_w2s2_{kind}_bin{self.zbin}_ns{self.nside}.fits.gz'
        self.w2s2s[mod] = self._rerun_read_cycle(fn, 'FITSMap',
                                                 get_w2s2)
        self.w2s2 = self.w2s2s[mod]
        return self.w2s2

    def get_nl_coupled(self, mode=None):
        kind, e1f, e2f, mod = self._set_mode(mode)
        if self.nls[mod] is None:
            self.w2s2 = self._get_w2s2(mode)
            N_ell = hp.nside2pixarea(self.nside) * np.mean(self.w2s2)
            nl = N_ell * np.ones(3*self.nside)
            nl[:2] = 0  # ylm = 0 for l < spin
            self.nls[mod] = np.array([nl, 0*nl, 0*nl, nl])
        self.nl_coupled = self.nls[mod]
        return self.nl_coupled

    def get_nz(self, dz=0):
        if self.dndz is None:
            z, nz = np.loadtxt(self.config['file_nz'], unpack=True)
            self.dndz = {'z_mid': z, 'nz': nz}
        return self._get_shifted_nz(dz)

    def get_dtype(self):
        return 'galaxy_shear'

    def get_spin(self):
        return 2
