from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.table import Table, vstack
import os
import numpy as np
import healpy as hp


class MapperHSCDR1wl(MapperBase):
    """
    **Config**

        - depth_cut: `24.5`
        - z_edges: `[0.3, 0.6]` / `[0.6, 0.9]` / `[0.9, 1.2]` / `[1.2, 1.5]`

        - bin_name: `bin0` / `bin1` / `bin2` / `bin3`
        - data_catalogs: \
        `[['.../Datasets/HSC_DR1/HSC_WIDE_GAMA09H.fits'], \
          ['.../Datasets/HSC_DR1/HSC_WIDE_GAMA15H.fits'], \
          ['.../Datasets/HSC_DR1/HSC_WIDE_HECTOMAP.fits'], \
          ['.../Datasets/HSC_DR1/HSC_WIDE_VVDS_part1.fits', \
           '.../Datasets/HSC_DR1/HSC_WIDE_VVDS_part2.fits'], \
          ['.../Datasets/HSC_DR1/HSC_WIDE_WIDE12H.fits'], \
          ['.../Datasets/HSC_DR1/HSC_WIDE_XMM.fits']]`

        - fname_cosmos: `'.../Datasets/HSC_DR1\
        /Afterburner_reweighted_COSMOS_photoz_FDFC.fits'`
        - fnames_cosmos_ph: \
        `['.../Datasets/HSC_DR1/pdf-s17a_wide-9812.cat.fits', \
          '.../Datasets/HSC_DR1/pdf-s17a_wide-9813.cat.fits']`

        - nbin_nz: `100`
        - zlim_nz: `[0.0, 4.0]`
        - mask_name: `'mask_HSC_wl0'` / `'mask_HSC_wl1'` / `'mask_HSC_wl2'` / \
                   `'mask_HSC_wl3'`
        - mapper_class: `'MapperHSCDR1wl'`
        - path_rerun: `'.../Datasets/HSC_DR1/lite/'`
    """
    def __init__(self, config):
        self._get_defaults(config)
        self.rot = self._get_rotator('C')
        self.icut = config.get('depth_cut', 24.5)
        self.z_edges = config['z_edges']
        self.bn = self.config['bin_name']
        self.w_name = 'ishape_hsm_regauss_derived_shape_weight'
        self.npix = hp.nside2npix(self.nside)

        self.nl_coupled = None
        self.dndz = None
        self.cat = None
        self.signal_map = None

    def _get_catalog_from_raw(self):
        # Loads the mappper raw catalog, cleans it,\
        # applies shear cuts, removes all but the \
        # chosen redshift bin and calibrates elipticities. \
        # Finally, it returns the processed catalog.

        cats = []
        for f in self.config['data_catalogs']:
            cat = self._clean_raw_catalog(f)

            # Shear cut
            shear_mod_thr = self.config.get('shear_mod_thr', 2)
            isn = 'ishape_hsm_regauss'
            ishape_flags_mask = ~cat[f'{isn}_flags']
            ishape_sigma_mask = ~np.isnan(cat[f'{isn}_sigma'])
            ishape_res_mask = cat[f'{isn}_resolution'] >= 0.3
            ishape_shear_mod_mask = (cat[f'{isn}_e1']**2 +
                                     cat[f'{isn}_e2']**2) < shear_mod_thr
            ishape_sigma_mask *= ((cat[f'{isn}_sigma'] >= 0.) *
                                  (cat[f'{isn}_sigma'] <= 0.4))
            # Remove masked objects
            imsk = 'iflags_pixel_bright'
            star_mask = np.logical_not(cat[f'{imsk}_object_center'])
            star_mask *= np.logical_not(cat[f'{imsk}_object_any'])
            fdfc_mask = cat['wl_fulldepth_fullcolor']
            shearmask = ishape_flags_mask *\
                ishape_sigma_mask *\
                ishape_res_mask *\
                ishape_shear_mod_mask*star_mask *\
                fdfc_mask
            cat = cat[shearmask]

            # Redshift bin cut
            zs = cat['pz_best_eab']
            zbinmask = (zs <= self.z_edges[1]) & (zs > self.z_edges[0])
            cat = cat[zbinmask]

            # Calibrate shear
            w = cat[self.w_name]
            mhat = np.average(cat[f'{isn}_derived_shear_bias_m'],
                              weights=w)
            resp = 1. - np.average(cat[f'{isn}_derived_rms_e'] ** 2,
                                   weights=w)
            e1 = (cat[f'{isn}_e1']/(2.*resp) -
                  cat[f'{isn}_derived_shear_bias_c1']) / (1 + mhat)
            e2 = (cat[f'{isn}_e2']/(2.*resp) -
                  cat[f'{isn}_derived_shear_bias_c2']) / (1 + mhat)
            cat['e1'] = e1
            cat['e2'] = e2

            # Remove unnecessary columns
            cat.keep_columns(['ra', 'dec', 'e1', 'e2', self.w_name])
            cats.append(cat)
        return vstack(cats).as_array()

    def get_catalog(self):
        """
        If lite catalog exists, loads it from save file. \
        Otherwise, it uses '_get_catalog_from_raw' \
        to produce a lite catalog.

        Returns:
            cat (Array)
        """
        if self.cat is None:
            fn = f'HSCDR1wl_{self.bn}.fits'
            self.cat = self._rerun_read_cycle(fn, 'FITSTable',
                                              self._get_catalog_from_raw)
        return self.cat

    def _clean_raw_catalog(self, fnames):
        # Cleans the raw HSC DR1 catalog
        # and produces a lite catalog.

        cats = []
        for fname in fnames:
            if not os.path.isfile(fname):
                raise ValueError(f"File {fname} not found")
            c = Table.read(fname)
            sel = np.ones(len(c), dtype=bool)
            isnull_names = []
            for key in c.keys():
                if key.__contains__('isnull'):
                    if not key.startswith('ishape'):
                        sel[c[key]] = 0
                    isnull_names.append(key)
                else:
                    # Keep photo-zs and shapes even if they're NaNs
                    if ((not key.startswith("pz_")) and
                            (not key.startswith('ishape'))):
                        sel[np.isnan(c[key])] = 0
            c.remove_columns(isnull_names)
            c.remove_rows(~sel)

            # Collect sample cuts
            sel_area = c['wl_fulldepth_fullcolor']
            sel_clean = sel_area & c['clean_photometry']
            sel_maglim = np.ones(len(c), dtype=bool)
            sel_maglim[c['icmodel_mag'] -
                       c['a_i'] > self.icut] = 0
            # Blending
            sel_blended = np.ones(len(c), dtype=bool)
            # Shear sample cuts as defined in https://arxiv.org/abs/1705.06745
            # abs_flux<10^-0.375
            sel_blended[c['iblendedness_abs_flux'] >= 0.42169650342] = 0
            # S/N in i
            sel_fluxcut_i = np.ones(len(c), dtype=bool)
            sel_fluxcut_i[c['icmodel_flux'] < 10*c['icmodel_flux_err']] = 0
            # S/N in g
            sel_fluxcut_g = np.ones(len(c), dtype=int)
            sel_fluxcut_g[c['gcmodel_flux'] < 5*c['gcmodel_flux_err']] = 0
            # S/N in r
            sel_fluxcut_r = np.ones(len(c), dtype=int)
            sel_fluxcut_r[c['rcmodel_flux'] < 5*c['rcmodel_flux_err']] = 0
            # S/N in z
            sel_fluxcut_z = np.ones(len(c), dtype=int)
            sel_fluxcut_z[c['zcmodel_flux'] < 5*c['zcmodel_flux_err']] = 0
            # S/N in y
            sel_fluxcut_y = np.ones(len(c), dtype=int)
            sel_fluxcut_y[c['ycmodel_flux'] < 5*c['ycmodel_flux_err']] = 0
            # S/N in grzy (at least 2 pass)
            sel_fluxcut_grzy = (sel_fluxcut_g+sel_fluxcut_r +
                                sel_fluxcut_z+sel_fluxcut_y >= 2)
            # Overall S/N
            sel_fluxcut = sel_fluxcut_i*sel_fluxcut_grzy
            # Stars
            sel_stars = np.ones(len(c), dtype=bool)
            sel_stars[c['iclassification_extendedness'] > 0.99] = 0
            # Galaxies
            sel_gals = np.ones(len(c), dtype=bool)
            sel_gals[c['iclassification_extendedness'] < 0.99] = 0
            sel = ~(sel_clean*sel_maglim*sel_gals*sel_fluxcut*sel_blended)
            c.remove_rows(sel)
            cats.append(c)
        return vstack(cats)

    def _get_ellip_maps(self):
        # Returns the ellipticity fields of the mapper's catalog.

        print(f'Computing bin {self.bn} signal map')
        cat = self.get_catalog()
        we1, we2 = get_map_from_points(cat, self.nside,
                                       w=cat[self.w_name],
                                       qu=[cat['e1'], cat['e2']],
                                       ra_name='ra',
                                       dec_name='dec',
                                       rot=self.rot)
        mask = self.get_mask()
        goodpix = mask > 0
        we1[goodpix] /= mask[goodpix]
        we2[goodpix] /= mask[goodpix]
        return we1, we2

    def get_signal_map(self):
        if self.signal_map is None:
            fn = '_'.join([f'HSCDR1wl_signal_{self.bn}',
                           f'coord{self.coords}',
                           f'ns{self.nside}.fits.gz'])
            d = self._rerun_read_cycle(fn, 'FITSMap',
                                       self._get_ellip_maps,
                                       section=[0, 1])
            self.signal_map = np.array([d[0], d[1]])
        return self.signal_map

    def _get_mask(self):
        # Loads mapper's from file

        print(f'Computing bin {self.bn} mask')
        cat = self.get_catalog()
        msk = get_map_from_points(cat, self.nside,
                                  w=cat[self.w_name],
                                  ra_name='ra',
                                  dec_name='dec',
                                  rot=self.rot)
        return msk

    def _get_w2s2(self):
        # Computes weight-square map for
        # noise power spectrum estimation.

        print('Computing w2s2 map')
        cat = self.get_catalog()
        w2s2 = get_map_from_points(cat, self.nside,
                                   w=(0.5*(cat['e1']**2 + cat['e2']**2) *
                                      cat[self.w_name]**2),
                                   ra_name='ra', dec_name='dec',
                                   rot=self.rot)
        return w2s2

    def get_nl_coupled(self):
        if self.nl_coupled is not None:
            return self.nl_coupled

        fn = '_'.join([f'HSCDR1wl_w2s2_{self.bn}',
                       f'coord{self.coords}',
                       f'ns{self.nside}.fits.gz'])
        w2s2 = self._rerun_read_cycle(fn, 'FITSMap', self._get_w2s2)
        N_ell = hp.nside2pixarea(self.nside) * np.sum(w2s2) / self.npix
        nl = N_ell * np.ones(3*self.nside)
        nl[:2] = 0  # Ylm = for l < spin
        self.nl_coupled = np.array([nl, 0*nl, 0*nl, nl])
        return self.nl_coupled

    def _get_nz(self):
        print('Computing nz')
        cat_cosmos = Table.read(self.config['fname_cosmos'])
        cat_photo = vstack([Table.read(n)
                            for n in self.config['fnames_cosmos_ph']])
        oid, id_ph, id_cs = np.intersect1d(cat_photo['ID'],
                                           cat_cosmos['S17a_objid'],
                                           return_indices=True)
        cat_photo = cat_photo[id_ph]
        cat_cosmos = cat_cosmos[id_cs]

        z0, zf = self.z_edges
        msk = np.where((cat_photo['PHOTOZ_BEST'] <= zf) &
                       (cat_photo['PHOTOZ_BEST'] > z0))[0]
        cosmos_masked = cat_cosmos[msk]
        # We need to reweight cosmos by color space and shape weight
        w = cosmos_masked['SOM_weight']*cosmos_masked['weight_source']
        hz, bz = np.histogram(cosmos_masked['COSMOS_photoz'],
                              bins=self.config.get('nbin_nz', 100),
                              range=self.config.get('zlim_nz',
                                                    [0., 4.]),
                              weights=w, density=True)
        dndz = hz*len(cosmos_masked)
        zm = 0.5*(bz[1:] + bz[:-1])
        return {'z_mid': zm, 'nz': dndz}

    def get_nz(self, dz=0):
        """
        Checks if mapper has precomputed the redshift \
        distribution. If not, it uses "_get_nz()" to obtain it. \
        Then, it shifts the distribution by "dz" (default dz=0).

        Kwargs:
            dz=0

        Returns:
            [z, nz] (Array)
        """
        if self.dndz is None:
            fname = f'HSCDR1wl_nz_{self.bn}.npz'
            self.dndz = self._rerun_read_cycle(fname, 'NPZ', self._get_nz)
        return self._get_shifted_nz(dz)

    def get_dtype(self):
        return 'galaxy_shear'

    def get_spin(self):
        return 2
