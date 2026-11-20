from .mapper_base import MapperBase
from .utils import get_map_from_points, save_rerun_data
from astropy.io import fits
import fitsio
import numpy as np
import pymaster as nmt
import healpy as hp


class MapperKiDSLegacy(MapperBase):
    """
    Mapper for the KiDS-Legacy cosmic shear sample. Each mapper
    instance represents a single tomographic bin (`zbin = 0, ..., 5`).

    The released ellipticities are already c-corrected (per tomographic
    bin and hemisphere), but not corrected for multiplicative bias.
    The mapper removes any residual mean ellipticity per bin and patch,
    and divides by (1+m).

    path = `".../Datasets/KiDS_Legacy/"`

    **Config**

        - data_catalog: `path+"KiDS_Legacy_NS_unblind_final.fits"` \
        (the catalog is distributed gzipped. Decompressing it first \
        is recommended, since otherwise the whole file must be \
        decompressed in memory when reading it).
        - file_nz: `path+"KiDS_Legacy_bandpowers.fits"` \
        (any of the data-vector files in the `data` directory of \
        `KiDS_Legacy_cosmic_shear_data_release.tar.gz`, which contain \
        the redshift distributions in the `NZ_SOURCE` extension).
        - mode: `"shear"` / `"PSF"`
        - zbin: `0` / `1` / `2` / `3` / `4` / `5`
        - mask_name: `"mask_KiDSLegacy__i"`
        - path_rerun: `path+'xcell_reruns/'`
    """
    map_name = 'KiDSLegacy'
    dtype = 'galaxy_shear'
    spin = 2
    # Relevant papers:
    # - Cosmic shear analysis: https://arxiv.org/abs/2503.19441
    # - Data release (catalog and column names):
    # https://kids.strw.leidenuniv.nl/DR5/legacy_wl.php
    # Photo-z bin edges in Z_B. Table 1 of 2503.19441.
    zbin_edges = np.array([[0.10, 0.42],
                           [0.42, 0.58],
                           [0.58, 0.71],
                           [0.71, 0.90],
                           [0.90, 1.14],
                           [1.14, 2.00]])
    # Multiplicative bias. Table 1 of 2503.19441.
    m = (-0.023, -0.016, -0.011, 0.020, 0.030, 0.045)

    def __init__(self, config):
        self._get_defaults(config)
        self.rot = self._get_rotator('C')
        self.mode = config.get('mode', 'shear')
        self.nzbins = len(self.zbin_edges)
        self.zbin = int(config['zbin'])
        if (self.zbin < 0) or (self.zbin >= self.nzbins):
            raise ValueError(f"zbin must be between 0 and {self.nzbins-1}")
        # Prefix for the per-bin catalogs, which do not depend on e.g.
        # whether overlap is removed.
        self.cat_prefix = self.map_name.split('_')[0]
        self.map_name += f"_bin{self.zbin}"

        self.cat_data = None
        self.maps = {'PSF': None, 'shear': None}
        self.w2s2s = {'PSF': None, 'shear': None}
        self.nl_coupled = None
        self.nls = {'PSF': None, 'shear': None}

        self.e1_flag = config.get('e1_flag', 'e1')
        self.e2_flag = config.get('e2_flag', 'e2')
        self.column_names = ['RAJ2000', 'DECJ2000', 'Z_B', 'PATCH',
                             'PSF_e1', 'PSF_e2', 'weight',
                             self.e1_flag, self.e2_flag]

    def get_catalog(self):
        """
        Returns the chosen redshift bin of the \
        mappers catalog, with calibrated ellipticities.

        Returns:
            cat_data (Array)
        """
        if self.cat_data is None:
            fn = f'{self.cat_prefix}_bin{self.zbin}_cat.fits'
            self.cat_data = self._rerun_read_cycle(fn, 'FITSTable',
                                                   self._load_catalog,
                                                   saved_by_func=True)
        return self.cat_data

    def _load_catalog(self):
        # Loads the catalog, splits it into tomographic bins,
        # removes the additive and multiplicative biases, and
        # saves all bins to the rerun directory.
        # Returns the catalog of the chosen bin.
        cat_full = fitsio.read(self.config['data_catalog'],
                               columns=self.column_names)
        patch = np.char.strip(cat_full['PATCH'].astype(str))
        cat_out = None
        for ibin in range(self.nzbins):
            sel = self._bin_z(cat_full, ibin)
            c = cat_full[sel]
            cat = np.zeros(len(c),
                           dtype=[('RA', 'f8'), ('DEC', 'f8'),
                                  ('e1', 'f8'), ('e2', 'f8'),
                                  ('PSF_e1', 'f8'), ('PSF_e2', 'f8'),
                                  ('weight', 'f8')])
            cat['RA'] = c['RAJ2000']
            cat['DEC'] = c['DECJ2000']
            cat['e1'] = c[self.e1_flag]
            cat['e2'] = c[self.e2_flag]
            cat['PSF_e1'] = c['PSF_e1']
            cat['PSF_e2'] = c['PSF_e2']
            cat['weight'] = c['weight']
            self._remove_additive_bias(cat, patch[sel])
            self._remove_multiplicative_bias(cat, ibin)
            fn = f'{self.cat_prefix}_bin{ibin}_cat.fits'
            save_rerun_data(self, fn, 'FITSTable', cat)
            if ibin == self.zbin:
                cat_out = cat
        return cat_out

    def _bin_z(self, cat, zbin):
        # Selects all sources in the chosen redshift bin.
        z_edges = self.zbin_edges[zbin]
        return ((cat['Z_B'] > z_edges[0]) &
                (cat['Z_B'] <= z_edges[1]))

    def _remove_additive_bias(self, cat, patch):
        # The additive bias is computed separately for the
        # North and South patches, as in 2503.19441.
        for p in np.unique(patch):
            sel = patch == p
            for ef in ['e1', 'e2']:
                emean = np.average(cat[ef][sel],
                                   weights=cat['weight'][sel])
                cat[ef][sel] -= emean

    def _remove_multiplicative_bias(self, cat, zbin):
        cat['e1'] /= 1 + self.m[zbin]
        cat['e2'] /= 1 + self.m[zbin]

    def _set_mode(self, mode=None):
        # Given the chosen mapper mode ('shear' or 'PSF'), \
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
            e1_flag = 'PSF_e1'
            e2_flag = 'PSF_e2'
        else:
            raise ValueError(f"Unknown mode {mode}")
        return e1_flag, e2_flag, mode

    def _get_ellip_maps(self, mode=None):
        # Returns the ellipticity maps of the chosen catalog ('shear' or 'PSF')
        e1f, e2f, mod = self._set_mode(mode)
        print('Computing bin{} signal map'.format(self.zbin))
        data = self.get_catalog()
        we1, we2 = get_map_from_points(data, self.nside,
                                       w=data['weight'],
                                       qu=[-data[e1f], data[e2f]],
                                       ra_name='RA',
                                       dec_name='DEC',
                                       rot=self.rot)
        mask = self.get_mask()
        goodpix = mask > 0
        we1[goodpix] /= mask[goodpix]
        we2[goodpix] /= mask[goodpix]
        return we1, we2

    def get_signal_map(self, mode=None):
        # Overwrite base method to simplify the handling of the mods
        e1f, e2f, mod = self._set_mode(mode)
        if self.maps[mod] is not None:
            self.signal_map = self.maps[mod]
            return self.signal_map

        # This will only be computed if self.maps['mod'] is None
        def get_ellip_maps():
            return self._get_ellip_maps(mode=mod)

        fn = '_'.join([f'{self.map_name}_signal_map_{mod}',
                       f'coord{self.coords}',
                       f'ns{self.nside}.fits.gz'])
        d = self._rerun_read_cycle(fn, 'FITSMap', get_ellip_maps,
                                   section=[0, 1])
        self.maps[mod] = np.array([d[0], d[1]])
        self.signal_map = self.maps[mod]
        return self.signal_map

    def _get_mask(self):
        # Returns the weights map.
        data = self.get_catalog()
        msk = get_map_from_points(data, self.nside,
                                  w=data['weight'],
                                  ra_name='RA',
                                  dec_name='DEC',
                                  rot=self.rot)
        return msk

    def _get_w2s2(self, mode=None):
        # Computes squared-weights map for
        # noise power spectrum estimation.
        e1f, e2f, mod = self._set_mode(mode)
        if self.w2s2s[mod] is not None:
            return self.w2s2s[mod]

        def get_w2s2():
            data = self.get_catalog()
            wcol = data['weight']**2*0.5*(data[e1f]**2+data[e2f]**2)
            w2s2 = get_map_from_points(data, self.nside, w=wcol,
                                       ra_name='RA',
                                       dec_name='DEC',
                                       rot=self.rot)
            msk = self.get_mask()
            w2s2[msk == 0] = 0
            return w2s2

        fn = '_'.join([f'{self.map_name}_{mod}_w2s2',
                       f'coord{self.coords}',
                       f'ns{self.nside}.fits.gz'])
        self.w2s2s[mod] = self._rerun_read_cycle(fn, 'FITSMap', get_w2s2)
        return self.w2s2s[mod]

    def get_nl_coupled(self, mode=None):
        e1f, e2f, mod = self._set_mode(mode)
        if self.nls[mod] is None:
            w2s2 = self._get_w2s2(mod)
            N_ell = hp.nside2pixarea(self.nside) * np.mean(w2s2)
            nl = N_ell * np.ones(3*self.nside)
            nl[:2] = 0  # Ylm = 0 for l < spin
            self.nls[mod] = np.array([nl, 0*nl, 0*nl, nl])
        self.nl_coupled = self.nls[mod]
        return self.nl_coupled

    def get_nz(self, dz=0):
        """
        Loads the redshift distribution of sources \
        from the data products. \
        Then, it shifts the distribution by "dz" (default dz=0). \
        Finally, it returns the redshift distribtuion.

        Kwargs:
            dz=0

        Returns:
            [z, nz] (Array)
        """
        if self.dndz is None:
            d = fits.open(self.config['file_nz'])['NZ_SOURCE'].data
            self.dndz = {'z_mid': np.array(d['Z_MID']),
                         'nz': np.array(d[f'BIN{self.zbin+1}'])}
        return self._get_shifted_nz(dz)

    def _get_nmt_catalog_field(self, mode=None, **kwargs):
        """
        Build a catalog-based NaMaster field for KiDS-Legacy.
        """
        e1f, e2f, mod = self._set_mode(mode)
        data = self.get_catalog()
        positions = np.array([data['RA'], data['DEC']])
        field = np.array([-data[e1f], data[e2f]])

        # Retrieve the maximum multipole
        lmax = kwargs.get("lmax", 3*self.nside-1)

        return nmt.NmtFieldCatalog(positions, data['weight'], field,
                                   spin=2, lmax=lmax, lonlat=True)
