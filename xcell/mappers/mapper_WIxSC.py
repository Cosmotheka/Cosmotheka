from .mapper_base import MapperBase
from .utils import get_map_from_points, get_DIR_Nz
import fitsio
from astropy.table import Table
import pymaster as nmt
import numpy as np
import healpy as hp
import os


class MapperWIxSC(MapperBase):
    """
    **Config**

    - mask_name: `"mask_2MPZWISC"`
    - path_rerun: `'.../Datasets/2MPZ_WIxSC/xcell_runs'`
    - z_edges: `[0.10, 0.15]` / `[0.15, 0.20]` / `[0.20, 0.25]` \
               `[0.25, 0.30]` / `[0.30, 0.35]`
    - bin_name: `'z0p10_0p15'` / `'z0p15_0p20'` / `'z0p20_0p25'` \
                `'z0p25_0p30'` / `'z0p30_0p35'`
    - nside_nl_threshold: `8129`
    - lmin_nl_from_data: `8192`
    - nl_analytic: `True`
    - data_catalog: `'.../Datasets/2MPZ_WIxSC/WIxSC.fits'`
    - spec_sample: `'.../Datasets/2MPZ_WIxSC/zSpec-comp-WIxSC.csv'`
    - n_jk_dir: `100`
    - apply_galactic_correction: `True`
    - mask_G: `'.../Datasets/2MPZ_WIxSC/WISExSCOSmask_galactic.fits.gz'`
    - mask_C: `'.../Datasets/2MPZ_WIxSC/WISExSCOSmask_equatorial.fits.gz'`
    - star_map_G: `'.../Datasets/2MPZ_WIxSC/allwise_total_galactic.fits'`
    - star_map_C: `'.../Datasets/2MPZ_WIxSC/allwise_total_equatorial.fits'`
    - use_halo_model: `True`
    - hod_params:

        - lMmin_0: `11.697` / `11.859` / `12.103` / `12.144` / `12.084`
        - siglM_0: `0.345` / `0.345` / `0.345` / `0.345` / `0.345`
        - lM0_0: `11.697` / `11.859` / `12.103` / 12.144 / `13.013`
        - lM1_0: `13.035` / `13.146` / `13.414` / 13.434 / `12.084`
        - alpha_0: `1.0` / `1.0` / `1.0` / `1.0` /  `1.0`
        - fc_0: `1.0` / `1.0` / `1.0` / `1.0` / `1.0`
    """
    def __init__(self, config):
        self._get_defaults(config)
        self.z_edges = config.get('z_edges', [0, 0.5])
        self.ra_name, self.dec_name, self.in_rad = self._get_coords(config)

        self.cat_data = None
        self.npix = hp.nside2npix(self.nside)
        self.bn = self.config['bin_name']

        # Angular mask
        self.dndz = None
        self.stars = None
        self.delta_map = None
        self.nl_coupled = None
        self.nside_nl_threshold = config.get('nside_nl_threshold',
                                             4096)
        self.lmin_nl_from_data = config.get('lmin_nl_from_data',
                                            2000)

    def get_radec(self, cat):
        """
        Returns the RA and DEC fields \
        of the mapper's catalog.

        Args:
            cat (Array): catalog

        Returns:
            ra (Array), dec (Array)
        """
        if self.in_rad:
            return (np.degrees(cat[self.ra_name]),
                    np.degrees(cat[self.dec_name]))
        else:
            return cat[self.ra_name], cat[self.dec_name]

    def _get_coords(self, config):
        # Returns names of the coordinates \
        # in the chosen coordinate \
        # system ('G'-galactic or 'C'-celestial).

        # Args:
        #     config (Dict): configuration file

        # Returns:
        #     coord1_name (String)
        #     coord2_name (String)

        if self.coords == 'G':  # Galactic
            return 'L', 'B', False
        elif self.coords == 'C':  # Celestial/Equatorial
            return 'RA', 'DEC', True
        else:
            raise NotImplementedError(f"Unknown coordinates {self.coords}")

    def _get_catalog(self):
        # Loads chosen columns of the raw WIxSC catalog. \
        # Selects the chosen bin and masks it.
        file_data = self.config['data_catalog']
        if not os.path.isfile(file_data):
            raise ValueError(f"File {file_data} not found")
        # Read catalog
        cat = fitsio.read(file_data,
                          columns=[self.ra_name,
                                   self.dec_name,
                                   'W1MCORR', 'W2MCORR',
                                   'RCALCORR', 'BCALCORR',
                                   'ZPHOTO_CORR'])
        # Bin in redshift
        cat = self._bin_z(cat)
        # Sky mask
        cat = self._mask_catalog(cat)
        return cat

    def get_catalog(self):
        if self.cat_data is None:
            fn = 'WIxSC_rerun_bin' + self.bn + '.fits'
            self.cat_data = self._rerun_read_cycle(fn, 'FITSTable',
                                                   self._get_catalog)
        return self.cat_data

    def _mask_catalog(self, cat):
        # Applies binary mask to catalog.

        self.mask = self.get_mask()
        ra, dec = self.get_radec(cat)
        ipix = hp.ang2pix(self.nside, ra, dec, lonlat=True)
        # Mask is binary, so 0.1 or 0.00001 doesn't really matter
        return cat[self.mask[ipix] > 0.1]

    def _bin_z(self, cat):
        # Removes all sources in the catalog \
        # outside the chosen bin.

        return cat[(cat['ZPHOTO_CORR'] > self.z_edges[0]) &
                   (cat['ZPHOTO_CORR'] <= self.z_edges[1])]

    def _get_specsample(self, cat):
        # Loads the spectroscopic catalog of \
        # sources of the mapper and bins it.

        import pandas as pd
        # Read full spectroscopic sample
        ds = Table.from_pandas(pd.read_csv(self.config['spec_sample']))
        msk = ((ds['zCorr'] < self.z_edges[-1]) &
               (ds['zCorr'] >= self.z_edges[0]))
        return ds[msk]

    def _get_nz(self):
        # Uses the DIR algorithm to build \
        # the redshift distribution of sources \
        # from the mapper's photometric sample \
        # given the spectroscopic sample.

        c_p = self.get_catalog()
        c_s = self._get_specsample(c_p)

        # Sort spec sample by nested pixel index so jackknife
        # samples are spatially correlated.
        ip_s = hp.ring2nest(self.nside,
                            hp.ang2pix(self.nside, c_s['ra_WISE'],
                                       c_s['dec_WISE'], lonlat=True))
        idsort = np.argsort(ip_s)
        c_s = c_s[idsort]

        # Compute DIR N(z)
        z, nz, nz_jk = get_DIR_Nz(c_s, c_p,
                                  ['W1c', 'W2c', 'Bcc', 'Rcc'],
                                  zflag='Zspec',
                                  zrange=[0, 0.6],
                                  nz=150,
                                  bands_photo=['W1MCORR', 'W2MCORR',
                                               'BCALCORR', 'RCALCORR'],
                                  njk=self.config.get('n_jk_dir', 100))
        zm = 0.5*(z[1:] + z[:-1])
        return {'z_mid': zm, 'nz': nz, 'nz_jk': nz_jk}

    def get_nz(self, dz=0, return_jk_error=False):
        """
        Checks if mapper has already computed \
        redshift distribution of sources. \
        If so, loads it from the save file. \
        Otherwise, it calculates it. \
        Finally, it shifts it by a given amount \
        (dz=0 default).

        Kwargs:
            dz=0
            return_jk_error=False

        Returns:
            [z, nz] (Array)
        """
        if self.dndz is None:
            fn = 'nz_WIxSC_bin' + self.bn + '.npz'
            self.dndz = self._rerun_read_cycle(fn, 'NPZ', self._get_nz)
        return self._get_shifted_nz(dz, return_jk_error=return_jk_error)

    def get_signal_map(self):
        # If 'apply_galactic_correction' = True,
        # it applies the galactic correction.

        if self.delta_map is None:
            d = np.zeros(self.npix)
            self.cat_data = self.get_catalog()
            self.mask = self.get_mask()
            self.stars = self._get_stars()
            nmap_data = get_map_from_points(self.cat_data, self.nside,
                                            ra_name=self.ra_name,
                                            dec_name=self.dec_name,
                                            in_radians=self.in_rad)
            mean_n = self._get_mean_n(nmap_data)
            goodpix = self.mask > 0
            # Division by mask not really necessary, since it's binary.
            d[goodpix] = nmap_data[goodpix]/(mean_n*self.mask[goodpix])-1
            if self.config.get('apply_galactic_correction', True):
                gcorr = self._get_galactic_correction(d, self.stars,
                                                      self.mask)
                d -= gcorr['delta_map']
            self.delta_map = np.array([d])
        return self.delta_map

    def _get_mask(self):
        # We will assume the mask has been provided in the right
        # coordinates, so no further conversion is needed.
        mask = hp.ud_grade(hp.read_map(self.config['mask']),
                           nside_out=self.nside)
        return mask

    def _get_stars(self):
        # Returns the mapper's stars catalog.
        if self.stars is None:
            # Power = -2 makes sure the total number of stars is conserved
            # We assume the star map has been provided in the right coords
            self.stars = hp.ud_grade(hp.read_map(self.config['star_map']),
                                     nside_out=self.nside, power=-2)
            # Convert to stars per deg^2
            pix_srad = 4*np.pi/self.npix
            pix_deg2 = pix_srad*(180/np.pi)**2
            self.stars /= pix_deg2
        return self.stars

    def _get_mean_n(self, nmap):
        # Returns mean of a given counts map.
        self.stars = self._get_stars()
        self.mask = self.get_mask()
        goodpix = self.stars < 8515
        goodpix *= self.mask > 0
        n_mean = np.sum(nmap[goodpix])/np.sum(self.mask[goodpix])
        return n_mean

    def _get_galactic_correction(self, delta, stars, bmask, nbins=14, npoly=3):
        # Computes the galactic correction.

        # Args:
        #     delta (Array): signal map
        #     stars (Array): stars map
        #     bmask (Array): binary mask

        # Kwargs:
        #     nbins=14 (Int): number of bins
        #     npoly=3 (Int): order of the polynomial fit

        # Returns:
        #     {'stars' (Array): binned star density,
        #      'delta_mean' (Array): stars' bins mean,
        #      'delta_std' (Array): stars' bins std,
        #      'delta_func' (Function): polynomial fit,
        #      'delta_map' (Array): galactic correction} (Dict)

        # Create bins of star density
        stbins = np.linspace(np.log10(stars[bmask > 0].min()),
                             np.log10(stars[bmask > 0].max()), nbins+1)
        d_mean = []
        d_std = []
        # Loop over bins and compute <delta> and std(delta) in each star bin
        for ib, st in enumerate(stbins[:-1]):
            stp1 = stbins[ib+1]
            msk = bmask > 0
            msk *= np.log10(stars) <= stp1
            msk *= np.log10(stars) >= st
            dm = np.mean(delta[msk])
            dn = np.sum(msk).astype(float)
            ds = np.std(delta[msk])/np.sqrt(dn)
            d_mean.append(dm)
            d_std.append(ds)
        d_mean = np.array(d_mean)
        d_std = np.array(d_std)
        # Now create a 3-rd order polynomial fitting these data
        stmid = 0.5*(stbins[1:]+stbins[:-1])
        params = np.polyfit(stmid, d_mean, npoly, w=1./d_std)
        df = np.poly1d(params)
        # Create correction map
        d_corr = np.zeros_like(delta)
        d_corr[bmask > 0] = df(np.log10(stars[bmask > 0]))
        return {'stars': stmid,
                'delta_mean': d_mean,
                'delta_std': d_std,
                'delta_func': df,
                'delta_map': d_corr}

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            if ((self.nside < self.nside_nl_threshold) or
                    (self.config.get('nl_analytic', True))):
                cat_data = self.get_catalog()
                n = get_map_from_points(cat_data, self.nside,
                                        ra_name=self.ra_name,
                                        dec_name=self.dec_name,
                                        in_radians=self.in_rad)
                N_mean = self._get_mean_n(n)
                N_mean_srad = N_mean * self.npix / (4 * np.pi)
                mask = self.get_mask()
                N_ell = np.mean(mask) / N_mean_srad
            else:
                f = self.get_nmt_field()
                cl = nmt.compute_coupled_cell(f, f)[0]
                N_ell = np.mean(cl[self.lmin_nl_from_data:2*self.nside])
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
