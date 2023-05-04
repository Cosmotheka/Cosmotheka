from .utils import get_map_from_points, rotate_mask
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
        - file_nz:
            `'2pt_NG_final_2ptunblind_02_24_21_wnz_redmagic_covupdate.fits'`
        - path_rerun: `'/mnt/extraspace/damonge/Datasets/DES_Y3/xcell_reruns/'`
        - mask_name: `'mask_DESY3wli'`
        - mapper_class: `'MapperDESY1wl'`
    """
    map_name = 'DESY3wl'
    # Relevant papers:
    # - Weak lensing catalog: https://arxiv.org/pdf/2011.03408.pdf
    # - Harmonic space weak lensing: https://arxiv.org/pdf/2203.07128.pdf
    # - Table 1 in https://arxiv.org/pdf/2105.13543.pdf for reference
    # We follow some aspects of https://github.com/des-science/DESY3Cats
    # - Info about catalog columns and files:
    # https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs

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
        # TODO: Consider moving this to MapperBase
        # Remove overlap? You should pass a dictionary with name & mask
        self.remove_overlap = config.get("remove_overlap")
        if self.remove_overlap is not None:
            self.map_name += '_removed_overlap'
            for k in self.remove_overlap.keys():
                self.map_name += f'_{k}'
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
        self.position = None
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
            select = index[f'index/{subgroup}_bin{self.zbin+1}'][:]
            if self.debug:
                select = select[:10000]
            self.select[kind] = select

        return self.select[kind]

    def _get_ellips(self, kind='unsheared', mode=None):
        self._check_kind(kind)
        _, _, mode = self._set_mode(mode)
        if self.ellips[mode][kind] is None:
            e1f, e2f, mode = self._set_mode(mode)
            sel = self._get_select(kind)
            index = self._get_cat_index()
            # For some reason [:][sel] is faster than [sel]
            e1 = index[f'catalog/metacal/{kind}'][e1f][:][sel]
            e2 = index[f'catalog/metacal/{kind}'][e2f][:][sel]
            self.ellips[mode][kind] = np.array((e1, e2))

        return self.ellips[mode][kind]

    def get_ellips_unbiased(self, mode=None):
        _, _, mode = self._set_mode(mode)
        if self.ellips_unbiased[mode] is None:
            ellips = self._get_ellips(mode=mode).copy()
            # TODO: Only for shear??
            if mode == 'shear':
                # Following 2203.07128
                # Note that, prior to eq. (1), observed ellipticities were
                # corrected for additive and multiplicative biases by
                # subtracting the (weighted) mean ellipticity (as done in
                # Gatti, Shel- don et al. 2021c) and dividing by the
                # Metacalibration response, both of which were computed for
                # each bin

                # Remove additive bias
                # ellips -= np.mean(ellips, axis=1)[:, None]
                w = self.get_weights()
                c = np.average(ellips, weights=w, axis=1)[:, None]
                print(f"Additive bias: {c}")
                ellips -= c
                # Remove multiplicative bias
                ellips = self._remove_multiplicative_bias(ellips)
            self.ellips_unbiased[mode] = ellips

        return self.ellips_unbiased[mode]

    def get_positions(self):
        if self.position is None:
            sel = self._get_select()
            index = self._get_cat_index()
            # For some reason [:][sel] is faster than [sel]
            ra = index['catalog/metacal/unsheared']['ra'][:][sel]
            dec = index['catalog/metacal/unsheared']['dec'][:][sel]
            self.position = {'ra': ra, 'dec': dec}

        return self.position

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
            # It is computed with the unsheared ellepticities but with cuts
            # obtained from the sheared ones.
            index = self._get_cat_index()
            data = np.array((index['catalog/metacal/unsheared']['e_1'][:],
                             index['catalog/metacal/unsheared']['e_2'][:]))

            sel_1p = self._get_select("sheared_1p")
            sel_1m = self._get_select("sheared_1m")
            sel_2p = self._get_select("sheared_2p")
            sel_2m = self._get_select("sheared_2m")

            data_1p = data[:, sel_1p]
            data_1m = data[:, sel_1m]
            data_2p = data[:, sel_2p]
            data_2m = data[:, sel_2m]

            # w = index['catalog/metacal/unsheared']['weight'][:]
            # w_1p = w[sel_1p]
            # w_1m = w[sel_1m]
            # w_2p = w[sel_2p]
            # w_2m = w[sel_2m]

            # In order to agree with the values in Table 1 of
            # https://arxiv.org/pdf/2105.13543.pdf one needs to apply the
            # weights in the sheared catalog
            w_1p = self.get_weights("sheared_1p")
            w_1m = self.get_weights("sheared_1m")
            w_2p = self.get_weights("sheared_2p")
            w_2m = self.get_weights("sheared_2m")

            mean_e1_1p, mean_e2_1p = np.average(data_1p, weights=w_1p, axis=1)
            mean_e1_1m, mean_e2_1m = np.average(data_1m, weights=w_1m, axis=1)
            mean_e1_2p, mean_e2_2p = np.average(data_2p, weights=w_2p, axis=1)
            mean_e1_2m, mean_e2_2m = np.average(data_2m, weights=w_2m, axis=1)

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
        print("Rg:", Rg, "Rs:", Rs)

        # Following DESY1 & https://arxiv.org/pdf/2105.13543.pdf:
        one_plus_m = np.sum(np.diag(Rmat))*0.5
        # print("Multiplicative bias:", one_plus_m - 1, "Rg:", Rg, "Rs:", Rs)
        return ellips / one_plus_m

        # Following 2011.03408:
        # "As noted in Sheldon & Huff (2017), the total ensemble response
        # matrix h𝑹i is, to good approximation, diagonal: as a consequence, the
        # response correction reduces to element-wise division"
        # return ellips / np.diag(Rmat)[:, None]

    def _get_ellipticity_maps(self, mode=None):
        # Returns the ellipticity maps of the chosen catalog ('shear' or 'PSF')
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
        # Removing the overlapping area at the mask level. Note that this is an
        # approximation and weights, biases, etc should be recomputed.
        if self.remove_overlap is not None:
            for v in self.remove_overlap.values():
                m = hp.read_map(v)
                m[m == hp.UNSEEN] = 0
                m = rotate_mask(m, self.rot)
                m = hp.ud_grade(m, nside_out=self.nside)
                # Set filled pixels to 0
                msk[m != 0] = 0
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
            w = 0.5*np.sum(ellips**2, axis=0) * weights**2
            mp = get_map_from_points(pos, self.nside, w=w,
                                     ra_name='ra', dec_name='dec',
                                     rot=self.rot)
            msk = self.get_mask()
            mp[msk == 0] = 0
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


# This function was used to shorten the official catalog. It is not used to
# produce the Cells. It was tested apart and writing a test for it complicates
# the test data generartion a bit. I'm going to skip it from coverage for now.
def save_index_short_per_bin(path):  # pragma: no cover
    def append_column(f, ds, col):
        if ds not in f:
            f.update({ds: col})

    index = h5py.File(path, mode='r')

    # Unsheared columns
    columns = ['ra', 'dec', 'weight', 'e_1', 'e_2', 'psf_e1', 'psf_e2',
               'R11', 'R12', 'R21', 'R22']
    for zbin in range(4):
        fname = f'_zbin{zbin}'.join(os.path.splitext(path))
        print(f"Creating {fname}")
        f = h5py.File(fname, mode='w')

        # Selection
        # We need to save also the galaxies selected in the sheared cases for
        # the computation of Rs
        ds = f'index/select_bin{zbin+1}'
        print(f"Loading {ds}", flush=True)
        select = index[ds][:]
        select_unsheared = select.copy()
        for suffix in ["1p", "1m", "2p", "2m"]:
            ds = f'index/select_{suffix}_bin{zbin+1}'
            print(f"Loading {ds}", flush=True)
            select = np.concatenate([select, index[ds][:]])
        select = np.unique(select)
        numbers = np.arange(select.size)

        # Added indices for unsheared galaxies
        ds = f'index/select_bin{zbin+1}'
        print(f"Loading {ds}", flush=True)
        append_column(f, ds, numbers[np.isin(select, select_unsheared)])

        # Add indices for sheared galaxies
        for suffix in ["1p", "1m", "2p", "2m"]:
            ds = f'index/select_{suffix}_bin{zbin+1}'
            print(f"Loading {ds}", flush=True)
            select_sheared = index[ds][:]
            append_column(f, ds, numbers[np.isin(select, select_sheared)])

        # Columns
        # Add unsheared columns
        for col in columns:
            ds = f"catalog/metacal/unsheared/{col}"
            print(f"Loading {ds}", flush=True)
            col = index[ds][:][select]
            append_column(f, ds, col)

        # Add sheared galaxies columns
        for grp in ['sheared_1p', 'sheared_1m', 'sheared_2p', 'sheared_2m']:
            # Columns
            for col in ['weight', 'e_1', 'e_2']:
                ds = f"catalog/metacal/{grp}/{col}"
                print(f"Loading {ds}", flush=True)
                col = index[ds][:][select]
                append_column(f, ds, col)
        f.close()
