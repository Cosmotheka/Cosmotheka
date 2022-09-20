from .mapper_base import MapperBase
from .utils import get_map_from_points, save_rerun_data
from astropy.table import Table
import numpy as np
import healpy as hp


class MapperKiDS1000(MapperBase):
    map_name = 'KiDS1000'

    def __init__(self, config):
        """
        config - dict
          {'data_catalog': 'KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits',
           'file_nz': SOM_N_of_Z/K1000_..._TOMO1_Nz.asc
          'zbin':0,
          'nside':nside,
          'mask_name': 'mask_KiDS1000_0'}
        """

        self._get_defaults(config)
        self.rot = self._get_rotator('C')

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
        # Values from Table 1 of 2007.01845
        self.m = (-0.009, -0.011, -0.015, 0.002, 0.007)

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

        self.e1_flag = config.get('e1_flag', 'e1')
        self.e2_flag = config.get('e2_flag', 'e2')
        self.column_names = ['SG_FLAG', 'Z_B', 'Z_B_MIN', 'Z_B_MAX',
                             'ALPHA_J2000', 'DELTA_J2000', 'PSF_e1', 'PSF_e2',
                             self.e1_flag, self.e2_flag, 'weight']

    def get_catalog(self):
        if self.cat_data is None:
            fn = f'{self.map_name}_cat_bin{self.zbin}.fits'
            self.cat_data = self._rerun_read_cycle(fn, 'FITSTable',
                                                   self._load_catalog,
                                                   saved_by_func=True)
        return self.cat_data

    def _load_catalog(self):
        nzbins = self.zbin_edges.shape[0]
        cat_out = None
        cat_full = Table.read(self.config['data_catalog'],
                              format='fits')[self.column_names]
        for ibin in range(nzbins):
            sel = self._bin_z(cat_full, ibin)
            cat = cat_full[sel]
            self._remove_additive_bias(cat)
            self._remove_multiplicative_bias(cat, ibin)
            cat = cat.as_array()
            fn = f'{self.map_name}_cat_bin{ibin}.fits'
            save_rerun_data(self, fn, 'FITSTable', cat)
            if ibin == self.zbin:
                cat_out = cat
        return cat_out

    def _set_mode(self):
        mode = self.mode

        if mode == 'shear':
            kind = 'galaxies'
            e1_flag = self.e1_flag
            e2_flag = self.e2_flag
        elif mode == 'PSF':
            kind = 'galaxies'
            e1_flag = 'PSF_e1'
            e2_flag = 'PSF_e2'
        elif mode == 'stars':
            kind = 'stars'
            e1_flag = self.e1_flag
            e2_flag = self.e2_flag
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
            e1mean = np.average(cat[self.e1_flag][sel_gals],
                                weights=cat['weight'][sel_gals])
            e2mean = np.average(cat[self.e2_flag][sel_gals],
                                weights=cat['weight'][sel_gals])
            cat[self.e1_flag][sel_gals] -= e1mean
            cat[self.e2_flag][sel_gals] -= e2mean

    def _remove_multiplicative_bias(self, cat, zbin):
        sel_gals = cat['SG_FLAG'] == 1
        cat[self.e1_flag][sel_gals] /= 1 + self.m[zbin]
        cat[self.e2_flag][sel_gals] /= 1 + self.m[zbin]

    def _get_gals_or_stars(self, kind='galaxies'):
        cat_data = self.get_catalog()
        sel = cat_data['SG_FLAG'] == self.sel[kind]
        return cat_data[sel]

    def _get_ellip_maps(self):
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
        # Overwrite base method to simplify the handling of the mods
        kind, e1f, e2f, mod = self._set_mode()
        if self.maps[mod] is not None:
            self.signal_map = self.maps[mod]
            return self.signal_map

        # This will only be computed if self.maps['mod'] is None
        fn = '_'.join([f'{self.map_name}_signal_{mod}_bin{self.zbin}',
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

        fn = '_'.join([f'{self.map_name}_w2s2_{kind}_bin{self.zbin}',
                       f'coord{self.coords}',
                       f'ns{self.nside}.fits.gz'])
        self.w2s2s[mod] = self._rerun_read_cycle(fn, 'FITSMap', get_w2s2)
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
        if self.dndz is None:
            z, nz = np.loadtxt(self.config['file_nz'], unpack=True)[:2]
            self.dndz = {'z_mid': z, 'nz': nz}
        return self._get_shifted_nz(dz)

    def get_dtype(self):
        return 'galaxy_shear'

    def get_spin(self):
        return 2
