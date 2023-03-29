from .utils import get_map_from_points
from .mapper_base import MapperBase
import h5py
import numpy as np
import healpy as hp
import fitsio
import os


class MapperDESY3wl(MapperBase):
    """
    Note that last letter of the the mask name stands for the chosen redshdift
    bin (`i = [1,2,3,4]`).

    path_cat = `'/mnt/extraspace/damonge/Datasets/DES_Y3/catalogs/'`
    path_dp = `'/mnt/extraspace/damonge/Datasets/DES_Y3/data_products/'`

    ***Config***

        - zbin: `0` / `1` / `2` /`3`
        - mode: `shear` / `PSF`
        - indexcat: `'DESY3_indexcat.h5'`
        - file_nz: `'2pt_NG_final_2ptunblind_02_24_21_wnz_redmagic_covupdate.fits'`
        - path_rerun: `'/mnt/extraspace/damonge/Datasets/DES_Y3/xcell_reruns/'`
        - mask_name: `'mask_DESY3wli'`
        - mapper_class: `'MapperDESY1wl'`
    """
    map_name = 'DESY3wl'
    # See https://arxiv.org/pdf/2011.03408.pdf
    # We follow some aspects of https://github.com/des-science/DESY3Cats

    # TODO:
    #  - If we are going to combine it with KiDS, we have to remove the
    #  overlapping area!
    #  -

    def __init__(self, config):
        self._get_defaults(config)
        self.config = config
        self.rot = self._get_rotator('C')
        self.mode = config.get('mode', 'shear')
        self.zbin = config['zbin']
        self.map_name += f"_bin{self.zbin}"
        self.npix = hp.nside2npix(self.nside)
        # dn/dz
        # load cat
        self.cat_index = None
        # get items for calibration
        self.Rs = None

        self.maps = {'PSF': None, 'shear': None}

        self.nl_coupled = None
        self.nls = {'PSF': None, 'shear': None}

        self.mask = None
        self.mcal_groups = ['unsheared', 'sheared_1p', 'sheared_1m',
                            'sheared_2p', 'sheared_2m']
        nones = [None] * len(self.mcal_groups)

        # Initialize to nones some usefuld dictionaries
        # Selection cuts
        self.select = dict(zip(self.mcal_groups, nones))
        # Galaxies position after the selection cuts have been applied
        self.position = dict(zip(self.mcal_groups, nones))
        # Ellipticities after the selection cuts have been applied
        self.ellips = {'PSF': dict(zip(self.mcal_groups, nones)),
                       'shear': dict(zip(self.mcal_groups, nones))}
        # Weights after the selection cuts have been applied
        self.weights = dict(zip(self.mcal_groups, nones))
        # Final ellipticities; i.e. the ones to be used for maps
        self.ellips_unbiased = {'PSF': None, 'shear': None}
        self.debug = config.get("debug", False)

    def _check_kind(self, kind):
        if kind not in self.mcal_groups:
            raise ValueError(f"kind={kind} not valid. It needs to be one of " +
                             ", ".join(self.mcal_groups))

    def _get_cat_index(self):
        # Read the index catalog that links to the other ones.
        # Columns explained in
        # https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs
        if self.cat_index is None:
            self.cat_index = h5py.File(self.config['indexcat'], mode='r')

        return self.cat_index

    def _get_select(self, kind='unsheared'):
        self._check_kind(kind)
        if self.select[kind] is None:
            index = self._get_cat_index()
            subgroup = "select"
            subgroup += kind[-3:] if kind != 'unsheared' else ''
            # select is an array of indices
            select = index[f'index/metacal/{subgroup}'][:]
            # For some reason [:][select] is faster than [select]
            selz = index[f'catalog/sompz/{kind}']['bhat'][:][select] == self.zbin
            select = select[selz]
            if self.debug:
                select = select[:10000]
            self.select[kind] = select

        return self.select[kind]

    def _get_ellips(self, kind='unsheared', mode='shear'):
        self._check_kind(kind)
        if self.ellips[mode][kind] is None:
            e1f, e2f, mode = self._set_mode(mode)
            sel = self._get_select(kind)
            index = self._get_cat_index()
            # For some reason [:][sel] is faster than [sel]
            e1 = index[f'catalog/metacal/{kind}'][e1f][:][sel]
            e2 = index[f'catalog/metacal/{kind}'][e2f][:][sel]
            self.ellips[mode][kind] = np.array((e1, e2))

        return self.ellips[mode][kind]

    def get_ellips_unbiased(self, mode):
        if self.ellips_unbiased[mode] is None:
            ellips = self._get_ellips(mode=mode)
            # TODO: Only for shear??
            if mode == 'shear':
                # Remove additive bias
                ellips -= np.mean(ellips, axis=1)[:, None]
                # Remove multiplicative bias
                ellips = self._remove_multiplicative_bias(ellips)
            self.ellips_unbiased[mode] = ellips

        return self.ellips_unbiased[mode]

    def get_positions(self, kind='unsheared'):
        self._check_kind(kind)
        if self.position[kind] is None:
            sel = self._get_select()
            index = self._get_cat_index()
            # For some reason [:][sel] is faster than [sel]
            ra = index[f'catalog/metacal/{kind}']['ra'][:][sel]
            dec = index[f'catalog/metacal/{kind}']['dec'][:][sel]
            self.position[kind] = {'ra': ra, 'dec': dec}

        return self.position[kind]

    def get_weights(self, kind='unsheared'):
        self._check_kind(kind)
        if self.weights[kind] is None:
            sel = self._get_select(kind)
            index = self._get_cat_index()
            # For some reason [:][sel] is faster than [sel]
            w = index[f'catalog/metacal/{kind}']['weight'][:][sel]
            self.weights[kind] = w

        return self.weights[kind]

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
            e1_flag = 'e_1'
            e2_flag = 'e_2'
        elif mode == 'PSF':
            e1_flag = 'psf_e1'
            e2_flag = 'psf_e2'
        else:
            raise ValueError(f"Unknown mode {mode}")
        return e1_flag, e2_flag, mode

    def _get_Rs(self):
        # Computes the sample response matrix Rs used to calculate the
        # multiplicative bias of the maps.
        # See https://arxiv.org/pdf/1702.02601.pdf

        if self.Rs is None:
            data_1p = self._get_ellips("sheared_1p")
            data_1m = self._get_ellips("sheared_1m")
            data_2p = self._get_ellips("sheared_2p")
            data_2m = self._get_ellips("sheared_2m")

            w_1p = self.get_weights("sheared_1p")
            w_1m = self.get_weights("sheared_1m")
            w_2p = self.get_weights("sheared_2p")
            w_2m = self.get_weights("sheared_2m")

            mean_e1_1p = np.mean(data_1p[0] * w_1p)
            mean_e2_1p = np.mean(data_1p[1] * w_1p)
            mean_e1_1m = np.mean(data_1m[0] * w_1m)
            mean_e2_1m = np.mean(data_1m[1] * w_1m)
            mean_e1_2p = np.mean(data_2p[0] * w_2p)
            mean_e2_2p = np.mean(data_2p[1] * w_2p)
            mean_e1_2m = np.mean(data_2m[0] * w_2m)
            mean_e2_2m = np.mean(data_2m[1] * w_2m)

            self.Rs = np.array([[(mean_e1_1p-mean_e1_1m)/0.02,
                                 (mean_e1_2p-mean_e1_2m)/0.02],
                                [(mean_e2_1p-mean_e2_1m)/0.02,
                                 (mean_e2_2p-mean_e2_2m)/0.02]])
        return self.Rs

    def _remove_multiplicative_bias(self, ellips):
        index = self._get_cat_index()
        cat = index['catalog/metacal/unsheared']
        w = self.get_weights()
        sel = self._get_select()
        Rg = np.array([[np.average(cat['R11'][:][sel], weights=w),
                        np.average(cat['R12'][:][sel], weights=w)],
                       [np.average(cat['R21'][:][sel], weights=w),
                        np.average(cat['R22'][:][sel], weights=w)]])
        Rs = self._get_Rs()
        Rmat = Rg + Rs
        one_plus_m = np.sum(np.diag(Rmat))*0.5

        return ellips / one_plus_m

    def _get_ellipticity_maps(self, mode=None):
        # Returns the ellipticity maps of the chosen catalog ('shear' or 'PSF').
        print('Computing bin{} signal map'.format(self.zbin))
        weights = self.get_weights()
        ellips = self.get_ellips_unbiased(mode)
        pos = self.get_positions()
        we1, we2 = get_map_from_points(pos, self.nside,
                                       qu=[-ellips[0], ellips[1]],
                                       w=weights,
                                       ra_name='ra',
                                       dec_name='dec',
                                       rot=self.rot)
        mask = self.get_mask()
        goodpix = mask > 0
        we1[goodpix] /= mask[goodpix]
        we2[goodpix] /= mask[goodpix]
        return we1, we2

    def get_signal_map(self, mode=None):
        # We overwrite the MapperBase method because otherwise it becomes very
        # convoluted
        e1f, e2f, mod = self._set_mode(mode)
        if self.maps[mod] is not None:
            self.signal_map = self.maps[mod]
            return self.signal_map

        # This will only be computed if self.maps['mod'] is None
        def get_ellip_maps():
            return self._get_ellipticity_maps(mode=mod)

        fn = '_'.join([f'{self.map_name}_signal_map_{mod}',
                       f'coord{self.coords}',
                       f'ns{self.nside}.fits.gz'])
        d = self._rerun_read_cycle(fn, 'FITSMap', get_ellip_maps,
                                   section=[0, 1])
        self.maps[mod] = np.array([d[0], d[1]])
        self.signal_map = self.maps[mod]
        return self.signal_map

    def get_nz(self, dz=0):
        """
        Returns the mappers redshift distribtuion of sources from a file.

        Kwargs:
            dz=0

        Returns:
            [z, nz] (Array)
        """
        if self.dndz is None:
            bn = f"BIN{self.zbin + 1}"
            f = fitsio.read(self.config['file_nz'], ext="nz_source",
                            columns=["Z_MID", bn])
            self.dndz = {'z_mid': f['Z_MID'], 'nz': f[bn]}
        return self._get_shifted_nz(dz)

    def _get_mask(self):
        # Returns the  mask.

        pos = self.get_positions()
        msk = get_map_from_points(pos, self.nside, w=self.get_weights(),
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
            pos = self.get_positions()
            weights = self.get_weights()
            ellips = self.get_ellips_unbiased(mod)
            mp = get_map_from_points(pos, self.nside,
                                     w=0.5*np.sum(ellips**2, axis=0) * weights,
                                     ra_name='ra', dec_name='dec',
                                     rot=self.rot)
            return mp

        fn = '_'.join([f'{self.map_name}_{mod}_w2s2',
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

def save_index_short(path):
    index = h5py.File(path, mode='r')

    fname = '_short'.join(os.path.splitext(path))
    with h5py.File(fname, mode='a') as f:
        # Catalog
        columns = ['ra', 'dec', 'weight', 'e_1', 'e_2', 'psf_e1', 'psf_e2',
                   'R11', 'R12', 'R21', 'R22']
        for col in columns:
            ds = f"catalog/metacal/unsheared/{col}"
            print(f"Loading {ds}", flush=True)
            if ds not in f:
                f.update({ds: index[ds][:]})

        groups = ['sheared_1p', 'sheared_1m', 'sheared_2p', 'sheared_2m']
        columns = ['weight', 'e_1', 'e_2']
        for grp in groups:
            for col in columns:
                ds = f"catalog/metacal/{grp}/{col}"
                print(f"Loading {ds}", flush=True)
                if ds not in f:
                    f.update({ds: index[ds][:]})

        # zbins
        for grp in groups + ['unsheared']:
            ds = f'catalog/sompz/{grp}/bhat'
            print(f"Loading {ds}", flush=True)
            if ds not in f:
                f.update({ds: index[ds][:]})

        # Selection
        subgroups = ['select' + ix for ix in ['', '_1p', '_2p', '_1m', '_2m']]
        for subgrp in subgroups:
            ds = f'index/metacal/{subgrp}'
            print(f"Loading {ds}", flush=True)
            if ds not in f:
                f.update({ds: index[ds][:]})

def save_index_short_per_bin(path):
    def append_column(zbin, ds, col):
        fname = f'_zbin{zbin}'.join(os.path.splitext(path))
        print(f"Creating {fname}")
        with h5py.File(fname, mode='a') as f:
            if ds not in f:
                f.update({ds: col})


    index = h5py.File(path, mode='r')

    # Unsheared
    columns = ['ra', 'dec', 'weight', 'e_1', 'e_2', 'psf_e1', 'psf_e2',
               'R11', 'R12', 'R21', 'R22']
    select = index[f'index/metacal/select'][:]
    bhat = index[f'catalog/sompz/unsheared/bhat'][:][select]

    for col in columns:
        ds = f"catalog/metacal/unsheared/{col}"
        print(f"Loading {ds}", flush=True)
        col = index[ds][:]
        for zbin in range(4):
            append_column(zbin, ds, col[select[bhat == zbin]])

    for zbin in range(4):
        nrows = np.sum(bhat == zbin)
        # zbins
        ds = f'catalog/sompz/unsheared/bhat'
        print(f"Loading {ds}", flush=True)
        append_column(zbin, ds, np.ones(nrows) * zbin)

        # Selection
        ds = f'index/metacal/select'
        print(f"Loading {ds}", flush=True)
        append_column(zbin, ds, np.arange(nrows))

    # Sheared
    for grp in ['sheared_1p', 'sheared_1m', 'sheared_2p', 'sheared_2m']:
        suffix = grp.split("_")[-1]
        select = index[f'index/metacal/select_{suffix}']
        bhat = index[f'catalog/sompz/{grp}/bhat'][:][select]

        for col in ['weight', 'e_1', 'e_2']:
            ds = f"catalog/metacal/{grp}/{col}"
            print(f"Loading {ds}", flush=True)
            col = index[ds][:]
            for zbin in range(4):
                append_column(zbin, ds, col[select[bhat == zbin]])

        for zbin in range(4):
            nrows = np.sum(bhat == zbin)
            # zbins
            ds = f'catalog/sompz/{grp}/bhat'
            print(f"Loading {ds}", flush=True)
            append_column(zbin, ds, np.ones(nrows) * zbin)

            # Selection
            ds = f'index/metacal/select_{suffix}'
            print(f"Loading {ds}", flush=True)
            append_column(zbin, ds, np.arange(np.sum(bhat == zbin)))
