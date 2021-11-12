from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.table import Table, vstack
import os
import numpy as np
import healpy as hp


class MapperHSCDR1wl(MapperBase):
    def __init__(self, config):
        """ Inputs:
        {'path_lite': path to liteweight files
         'depth_cut': i-band magnitude cut
         'z_edges': photo-z bin edges
         'bin_name': name for this redshift bin
         'data_catalogs': list of lists of files (one list for each HSC field)
         'shear_mod_thr': shear modulus threshold (2)
         'fname_cosmos': name of the matched COSMOS catalog
         'fname_cosmos_ph': list of names of the photo-z COSMOS files
         'nbin_nz': number of intervals for redshift distribution (100)
         'zlim_nz': redshift range of redshift distribution (0-4)
        }
        """
        self._get_defaults(config)
        self.path_lite = config.get('path_lite', None)
        self.icut = config.get('depth_cut', 24.5)
        self.z_edges = config.get('z_edges', [0, 0.5])
        self.bn = self.config['bin_name']
        self.w_name = 'ishape_hsm_regauss_derived_shape_weight'
        self.npix = hp.nside2npix(self.nside)

        self.nl_coupled = None
        self.dndz = None
        self.cat = None
        self.mask = None
        self.signal_map = None

    def _check_lite_exists(self, file_name):
        if self.path_lite is None:
            return False, None
        else:
            fname_lite = os.path.join(self.path_lite, file_name)
            return os.path.isfile(fname_lite), fname_lite

    def get_catalog(self):
        if self.cat is None:
            file_name = f'HSCDR1wl_{self.bn}.fits'
            read_lite, fname_lite = self._check_lite_exists(file_name)

            if read_lite:
                print(f'Loading lite catalog {file_name}')
                self.cat = Table.read(fname_lite, memmap=True)
                return self.cat

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
            self.cat = vstack(cats)
            if fname_lite is not None:
                self.cat.write(fname_lite)
        return self.cat

    def _clean_raw_catalog(self, fnames):
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

    def get_signal_map(self):
        if self.signal_map is None:
            fname_1 = f'HSCDR1wl_{self.bn}_e1_ns{self.nside}.fits.gz'
            fname_2 = f'HSCDR1wl_{self.bn}_e2_ns{self.nside}.fits.gz'
            read_lite1, fname_lite1 = self._check_lite_exists(fname_1)
            read_lite2, fname_lite2 = self._check_lite_exists(fname_2)
            if read_lite1 and read_lite2:
                print(f'Loading lite maps {fname_1}, {fname_2}')
                e1 = hp.read_map(fname_lite1)
                e2 = hp.read_map(fname_lite2)
                self.signal_map = [e1, e2]
            else:
                print(f'Computing bin {self.bn} signal map')
                cat = self.get_catalog()
                we1 = get_map_from_points(cat, self.nside,
                                          w=cat['e1']*cat[self.w_name],
                                          ra_name='ra',
                                          dec_name='dec')
                we2 = get_map_from_points(cat, self.nside,
                                          w=cat['e2']*cat[self.w_name],
                                          ra_name='ra',
                                          dec_name='dec')
                mask = self.get_mask()
                goodpix = mask > 0
                we1[goodpix] /= mask[goodpix]
                we2[goodpix] /= mask[goodpix]
                self.signal_map = [we1, we2]
                if fname_lite1 is not None:
                    hp.write_map(fname_lite1, we1, overwrite=True)
                if fname_lite2 is not None:
                    hp.write_map(fname_lite2, we2, overwrite=True)

        return self.signal_map

    def get_mask(self):
        if self.mask is not None:
            return self.mask

        fname = f'HSCDR1wl_{self.bn}_mask_ns{self.nside}.fits.gz'
        read_lite, fname_lite = self._check_lite_exists(fname)
        if read_lite:
            print(f'Loading lite mask {fname}')
            self.mask = hp.read_map(fname_lite)
        else:
            print(f'Computing bin {self.bn} mask')
            cat = self.get_catalog()
            self.mask = get_map_from_points(cat, self.nside,
                                            w=cat[self.w_name],
                                            ra_name='ra',
                                            dec_name='dec')
            if fname_lite is not None:
                hp.write_map(fname_lite, self.mask, overwrite=True)
        return self.mask

    def get_nl_coupled(self):
        if self.nl_coupled is not None:
            return self.nl_coupled

        fname = f'HSCDR1wl_{self.bn}_w2s2_ns{self.nside}.fits.gz'
        read_lite, fname_lite = self._check_lite_exists(fname)
        if read_lite:
            print(f'Loading lite w2s2 {fname}')
            w2s2 = hp.read_map(fname_lite)
        else:
            print('Computing noise bias')
            cat = self.get_catalog()
            w2s2 = get_map_from_points(cat, self.nside,
                                       w=(0.5*(cat['e1']**2 + cat['e2']**2) *
                                          cat[self.w_name]**2),
                                       ra_name='ra', dec_name='dec')
            if fname_lite is not None:
                hp.write_map(fname_lite, w2s2, overwrite=True)
        N_ell = hp.nside2pixarea(self.nside) * np.sum(w2s2) / self.npix
        nl = N_ell * np.ones(3*self.nside)
        nl[:2] = 0  # Ylm = for l < spin
        self.nl_coupled = np.array([nl, 0*nl, 0*nl, nl])
        return self.nl_coupled

    def get_nz(self, dz=0):
        if self.dndz is None:
            fname = f'HSCDR1wl_{self.bn}_nz.npz'
            read_lite, fname_lite = self._check_lite_exists(fname)
            if read_lite:
                print(f'Loading lite nz {fname_lite}')
                d = np.load(fname_lite)
                self.dndz = [d['z'], d['nz']]
            else:
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
                if fname_lite is not None:
                    np.savez(fname_lite, z=zm, nz=dndz)
                self.dndz = [zm, dndz]

        z, nz = self.dndz
        z_dz = z + dz
        sel = z_dz >= 0
        return np.array([z_dz[sel], nz[sel]])

    def get_dtype(self):
        return 'galaxy_shear'

    def get_spin(self):
        return 2
