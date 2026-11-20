from .mapper_base import MapperBase
from .utils import get_map_from_points, save_rerun_data
import h5py
import numpy as np
import pymaster as nmt
import healpy as hp


class MapperDECADEwl(MapperBase):
    """
    Mapper for the DECADE cosmic shear sample. The catalog covers two
    regions, the Northern (NGC) and Southern (SGC) Galactic caps, which are
    calibrated separately (different responses, multiplicative biases and
    redshift distributions). Each mapper instance represents a single
    tomographic bin in a single region.

    path = `".../Datasets/DECADE/"`

    **Config**

        - data_catalog: `path+"shear_catalog_sparse.hdf5"`
        - file_nz: `path+"NGC_n_of_z.npy"` / `path+"SGC_n_of_z.npy"`
        - file_zgrid: `path+"z_grid.npy"`
        - region: `"NGC"` / `"SGC"`
        - zbin: `0` / `1` / `2` / `3`
        - mode: `"shear"` / `"PSF"`
        - mask_name: `"mask_DECADEwl_NGC_i"`
        - path_rerun: `path+'xcell_reruns/'`
    """
    map_name = 'DECADEwl'
    spin = 2
    dtype = 'galaxy_shear'
    # Relevant papers:
    # - Shape catalog: https://arxiv.org/abs/2502.17674
    # - Redshift calibration: https://arxiv.org/abs/2502.17675
    # - NGC+SGC cosmology: https://arxiv.org/abs/2509.03582
    # - Data release (column names, responses and m values):
    # https://dhayaaanbajagane.github.io/data_release/decade/
    # The NGC/SGC split and the response calculation follow
    # dsigma/scripts/process_decade.py (https://github.com/johannesulf/dsigma)

    # Residual multiplicative biases from the data release page. These are
    # *not* applied to the maps (as in the DESY3 mapper, they should be
    # marginalised over at the likelihood level), but are stored here for
    # reference.
    m_bias = {'NGC': (-0.0092, -0.0190, -0.0400, -0.0373),
              'SGC': (-0.0133, -0.0226, -0.0367, -0.0572)}
    # Coordinates of the North Galactic Pole used to split NGC and SGC.
    radec_ngp = (193., 27.)
    nzbins = 4

    def __init__(self, config):
        self._get_defaults(config)
        self.rot = self._get_rotator('C')
        self.mode = config.get('mode', 'shear')
        self.region = config.get('region', 'NGC')
        if self.region not in ['NGC', 'SGC']:
            raise ValueError(f"Unknown region {self.region}. "
                             "Must be 'NGC' or 'SGC'")
        self.zbin = int(config['zbin'])
        if (self.zbin < 0) or (self.zbin >= self.nzbins):
            raise ValueError(f"zbin must be between 0 and {self.nzbins-1}")
        # Prefix for the per-bin catalogs, which do not depend on e.g.
        # whether overlap is removed.
        self.cat_prefix = f"{self.map_name.split('_')[0]}_{self.region}"
        self.map_name += f"_{self.region}_bin{self.zbin}"

        self.cat_data = None
        self.maps = {'PSF': None, 'shear': None}
        self.w2s2s = {'PSF': None, 'shear': None}
        self.nl_coupled = None
        self.nls = {'PSF': None, 'shear': None}

    def _in_region(self, ra, dec):
        # Returns True for objects in the chosen Galactic cap.
        ra_ngp, dec_ngp = np.radians(self.radec_ngp)
        ra = np.radians(ra)
        dec = np.radians(dec)
        cos_alpha = (np.sin(dec)*np.sin(dec_ngp) +
                     np.cos(dec)*np.cos(dec_ngp)*np.cos(ra-ra_ngp))
        is_ngc = cos_alpha > 0
        if self.region == 'NGC':
            return is_ngc
        return ~is_ngc

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
        # Loads the catalog for the chosen region and computes the
        # Metacalibration response of each tomographic bin. Removes the
        # additive bias (mean shear) and divides by the response.
        # All bins are saved to the rerun directory.
        # Returns the catalog of the chosen bin.
        cat_out = None
        bins = np.arange(self.nzbins)
        with h5py.File(self.config['data_catalog'], mode='r') as f:
            ra = f['RA'][:]
            dec = f['DEC'][:]
            in_reg = self._in_region(ra, dec)
            ra = ra[in_reg]
            dec = dec[in_reg]

            def read(key):
                # For some reason [:][sel] is faster than [sel]
                return f[key][:][in_reg]

            sel = read('MCAL_SEL_NOSHEAR')
            goods = [sel == ib+1 for ib in bins]
            # Note: the DECADE catalog comes with a number of flags:
            # FLAGS_FOOTPRINT, FLAGS_FOREGROUND, MCAL_FLAGS.
            # I've checked that selecting by tomographic bin (as
            # above) guarantees that all of these other flags
            # also pass (i.e. they have the right value for a
            # cosmology shear analysis).

            w = read('MCAL_W_NOSHEAR')
            e = np.array([read('MCAL_G_1_NOSHEAR'),
                          read('MCAL_G_2_NOSHEAR')])

            # Shear response: weighted mean of the per-galaxy diagonal
            # elements, R_ii = (e_i^{i+} - e_i^{i-}) / (2*0.01).
            Rg = np.zeros([self.nzbins, 2])
            for i in [1, 2]:
                r = (read(f'MCAL_G_{i}_{i}P') -
                     read(f'MCAL_G_{i}_{i}M'))/0.02
                for ib in bins:
                    Rg[ib, i-1] = np.average(r[goods[ib]],
                                             weights=w[goods[ib]])
            del r

            # Selection response: R_s,ii = (<e_i>^{S_i+} - <e_i>^{S_i-})/0.02
            # where the averages use the unsheared ellipticities, with
            # the selection and weights of the sheared catalogs.
            Rs = np.zeros([self.nzbins, 2])
            for i in [1, 2]:
                for sign, s in zip([1, -1], ['P', 'M']):
                    sel_s = read(f'MCAL_SEL_{i}{s}')
                    w_s = read(f'MCAL_W_{i}{s}')
                    for ib in bins:
                        good_s = sel_s == ib+1
                        Rs[ib, i-1] += sign*np.average(e[i-1][good_s],
                                                       weights=w_s[good_s])
            Rs /= 0.02
            del sel_s, w_s

            psf = np.array([read('MCAL_PSF_G_1_NOSHEAR'),
                            read('MCAL_PSF_G_2_NOSHEAR')])

        for ib in bins:
            good = goods[ib]
            ee = e[:, good]
            ww = w[good]
            # Remove additive bias
            c = np.average(ee, weights=ww, axis=1)
            # Following the DECADE data release, the response is the
            # mean of the two diagonal elements.
            R = np.mean(Rg[ib] + Rs[ib])
            print(f"{self.region} bin {ib}: Additive bias: {c}, "
                  f"Rg: {Rg[ib]}, Rs: {Rs[ib]}, R: {R}", flush=True)
            ee = (ee - c[:, None]) / R

            cat = np.zeros(np.sum(good),
                           dtype=[('RA', 'f8'), ('DEC', 'f8'),
                                  ('e1', 'f8'), ('e2', 'f8'),
                                  ('PSF_e1', 'f8'), ('PSF_e2', 'f8'),
                                  ('weight', 'f8')])
            cat['RA'] = ra[good]
            cat['DEC'] = dec[good]
            cat['e1'] = ee[0]
            cat['e2'] = ee[1]
            cat['PSF_e1'] = psf[0][good]
            cat['PSF_e2'] = psf[1][good]
            cat['weight'] = ww
            fn = f'{self.cat_prefix}_bin{ib}_cat.fits'
            save_rerun_data(self, fn, 'FITSTable', cat)
            if ib == self.zbin:
                cat_out = cat
        return cat_out

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
            # z_grid.npy contains the bin edges and mid point of each z bin.
            # *_n_of_z.npy contains the n(z) of the 4 tomographic bins.
            z = np.load(self.config['file_zgrid'])['z_MID']
            nz = np.load(self.config['file_nz'])[self.zbin]
            self.dndz = {'z_mid': z, 'nz': nz}
        return self._get_shifted_nz(dz)

    def _get_nmt_catalog_field(self, mode=None, **kwargs):
        """
        Build a catalog-based NaMaster field for DECADE WL.
        """
        e1f, e2f, mod = self._set_mode(mode)
        data = self.get_catalog()
        positions = np.array([data['RA'], data['DEC']])
        field = np.array([-data[e1f], data[e2f]])

        # Retrieve the maximum multipole
        lmax = kwargs.get("lmax", 3*self.nside-1)

        return nmt.NmtFieldCatalog(positions, data['weight'], field,
                                   spin=2, lmax=lmax, lonlat=True)
