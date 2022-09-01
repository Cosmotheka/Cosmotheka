from .mapper_base import MapperBase
from .utils import get_map_from_points, rotate_mask
from astropy.table import Table
import numpy as np
import healpy as hp


class MapperNVSS(MapperBase):
    """
    **Config**

        - data_catalog: `".../Datasets/NVSS/nvss.fits"`
        - mask_sources: `".../Datasets/NVSS/source_masks_nvss.txt"`
        - redshift_catalog: ".../Datasets/NVSS/100sqdeg_1uJy_s1400.fits"`
        - DEC_min_deg: `-40`
        - flux_min_mJy: `10`
        - fux_max_mJy: `1000`
        - GLAT_max_deg: `10`
        - mask_name: `"mask_NVSS"`
        - path_rerun: `".../Datasets/NVSS/xcell_runs"`
    """
    def __init__(self, config):
        self._get_defaults(config)
        self.rot = self._get_rotator('C')
        self.file_sourcemask = config.get('mask_sources', None)
        self.ra_name = 'RAJ2000'
        self.dec_name = 'DEJ2000'
        self.cat_data = None

        self.npix = hp.nside2npix(self.nside)
        # Angular mask
        self.delta_map = None
        self.nl_coupled = None
        self.dndz = None
        self.cat_redshift = None

    def get_catalog(self):
        """
        Returns the mapper's catalog of sources \
        (RA & DEC) after applying flux filters.

        Returns:
            cat_data (Array)
        """
        if self.cat_data is None:
            file_data = self.config['data_catalog']
            self.cat_data = Table.read(file_data)
            # Galactic coordinates
            r = hp.Rotator(coord=['C', 'G'])
            GLON, GLAT = r(self.cat_data['RAJ2000'], self.cat_data['DEJ2000'],
                           lonlat=True)
            self.cat_data['GLON'] = GLON
            self.cat_data['GLAT'] = GLAT
            # Angular and flux conditions
            self.cat_data = self.cat_data[
                (self.cat_data['DEJ2000'] >
                 self.config.get('DEC_min_deg', -40)) &
                (self.cat_data['S1_4'] >
                 self.config.get('flux_min_mJy', 10)) &
                (self.cat_data['S1_4'] <
                 self.config.get('flux_max_mJy', 1000)) &
                (np.fabs(self.cat_data['GLAT']) >
                 self.config.get('GLAT_max_deg', 5))]
        return self.cat_data

    def get_catalog_redshift(self):
        """
        Returns the mapper's catalog of redshifts \
        after applying flux filters.

        Returns:
            cat_redshift (Array)
        """
        if self.cat_redshift is None:
            file_data = self.config['redshift_catalog']
            self.cat_redshift = Table.read(file_data)
            # flux_mJy = 10.**(3+cat['itot_1400'])
            flux_mJy = 10.**(3+self.cat_redshift['itot_1400'])
            self.cat_redshift['flux_mJy'] = flux_mJy
            # Flux conditions
            self.cat_redshift = self.cat_redshift[
                (self.cat_redshift['flux_mJy'] > 10) &
                (self.cat_redshift['flux_mJy'] < 1000) &
                (self.cat_redshift['redshift'] <= 5)]
        return self.cat_redshift

    def get_signal_map(self, apply_galactic_correction=True):
        if self.delta_map is None:
            d = np.zeros(self.npix)
            self.cat_data = self.get_catalog()
            self.mask = self.get_mask()
            nmap_data = get_map_from_points(self.cat_data, self.nside,
                                            ra_name=self.ra_name,
                                            dec_name=self.dec_name,
                                            rot=self.rot)
            mean_n = np.average(nmap_data, weights=self.mask)
            goodpix = self.mask > 0
            # Division by mask not really necessary, since it's binary.
            d[goodpix] = nmap_data[goodpix]/(mean_n*self.mask[goodpix])-1
            self.delta_map = np.array([d])
        return self.delta_map

    def _get_mask(self):
        if self.config.get('mask_file', None) is not None:
            mask = hp.read_map(self.config['mask_file'])
            mask = hp.ud_grade(rotate_mask(mask, self.rot),
                               nside_out=self.nside)
            mask[mask > 0.5] = 1.
            mask[mask <= 0.5] = 0.
        else:
            mask = np.ones(self.npix)
            r = hp.Rotator(coord=['C', 'G'])
            RApix, DEpix = hp.pix2ang(self.nside, np.arange(self.npix),
                                      lonlat=True)
            lpix, bpix = r(RApix, DEpix, lonlat=True)
            # angular conditions
            mask[(DEpix < self.config.get('DEC_min_deg', -40)) |
                 (np.fabs(bpix) < self.config.get('GLAT_max_deg', 5))] = 0
            if self.file_sourcemask is not None:
                # holes catalog
                RAmask, DEmask, radmask = np.loadtxt(self.file_sourcemask,
                                                     unpack=True)
                vecmask = hp.ang2vec(RAmask, DEmask, lonlat=True)
                for vec, radius in zip(vecmask, radmask):
                    ipix_hole = hp.query_disc(self.nside, vec,
                                              np.radians(radius),
                                              inclusive=True)
                    mask[ipix_hole] = 0
            mask = rotate_mask(mask, self.rot, binarize=True)
        return mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            self.cat_data = self.get_catalog()
            self.mask = self.get_mask()
            nmap_data = get_map_from_points(self.cat_data, self.nside,
                                            ra_name=self.ra_name,
                                            dec_name=self.dec_name,
                                            rot=self.rot)
            N_mean = np.average(nmap_data, weights=self.mask)
            N_mean_srad = N_mean * self.npix / (4 * np.pi)
            N_ell = np.mean(self.mask) / N_mean_srad
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled

    def get_nz(self, dz=0):
        """
        Loads the redshift distribution catalog. \
        Then, it shifts the distribution by "dz" (default dz=0). \
        Finally, it returns the redshift distribtuion.

        Kwargs:
            dz=0

        Returns:
            [z, nz] (Array)
        """
        if self.dndz is None:
            self.cat_redshift = self.get_catalog_redshift()
            bins = np.arange(0, max(self.cat_redshift['redshift'])+0.1, 0.1)
            nz, bins = np.histogram(self.cat_redshift['redshift'], bins)
            zz = 0.5*(bins[1:]+bins[:-1])
            self.dndz = {'z_mid': zz, 'nz': nz}
        return self._get_shifted_nz(dz)

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
