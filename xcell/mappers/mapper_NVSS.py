from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.table import Table
import numpy as np
import healpy as hp


class MapperNVSS(MapperBase):
    def __init__(self, config):
        """
        config - dict
          {'data_catalog': 'Legacy_Survey_BASS-MZLS_galaxies-selection.fits',
           'mask': 'mask.fits',
           'z_edges': [0, 0.5],
           'path_rerun': '.',
           'n_jk_dir': 100,
           'mask_name': 'mask_2MPZ'}
        """
        self._get_defaults(config)
        self.file_sourcemask = config['mask_sources']
        self.ra_name = 'RAJ2000'
        self.dec_name = 'DEJ2000'
        self.cat_data = None

        self.npix = hp.nside2npix(self.nside)
        # Angular mask
        self.mask = None
        self.delta_map = None
        self.nl_coupled = None

    def get_catalog(self):
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
            self.cat_data = self.cat_data[(self.cat_data['DEJ2000'] > -40) &
                                          (self.cat_data['S1_4'] > self.config.get('flux_min_mJy', 10)) &
                                          (self.cat_data['S1_4'] < self.config.get('flux_max_mJy', 1000)) &
                                          (np.fabs(self.cat_data['GLAT']) > self.config.get('GLAT_max_deg', 5))]
        return self.cat_data

    # ill need this in the future
    def get_nz(self, dz=0, return_jk_error=False):
        raise NotImplementedError("N(z) not implemented yet")

    def get_signal_map(self, apply_galactic_correction=True):
        if self.delta_map is None:
            d = np.zeros(self.npix)
            self.cat_data = self.get_catalog()
            self.mask = self.get_mask()
            nmap_data = get_map_from_points(self.cat_data, self.nside,
                                            ra_name=self.ra_name,
                                            dec_name=self.dec_name)
            mean_n = np.average(nmap_data, weights=self.mask)
            goodpix = self.mask > 0
            # Division by mask not really necessary, since it's binary.
            d[goodpix] = nmap_data[goodpix]/(mean_n*self.mask[goodpix])-1
            self.delta_map = d
        return [self.delta_map]

    def get_mask(self):
        if self.mask is None:
            self.mask = np.ones(self.npix)
            r = hp.Rotator(coord=['C', 'G'])
            RApix, DEpix = hp.pix2ang(self.nside, np.arange(self.npix),
                                      lonlat=True)
            lpix, bpix = r(RApix, DEpix, lonlat=True)
            # angular conditions
            self.mask[(DEpix < -40) | (np.fabs(bpix) < self.config.get('GLAT_max_deg', 5))] = 0
            # holes catalog
            RAmask, DEmask, radiusmask = np.loadtxt(self.file_sourcemask,
                                                    unpack=True)
            vecmask = hp.ang2vec(RAmask, DEmask, lonlat=True)
            for vec, radius in zip(vecmask, radiusmask):
                ipix_hole = hp.query_disc(self.nside, vec, np.radians(radius),
                                          inclusive=True)
                self.mask[ipix_hole] = 0
        return self.mask

    # look at this function later
    def get_nl_coupled(self):
        if self.nl_coupled is None:
            self.cat_data = self.get_catalog()
            self.mask = self.get_mask()
            nmap_data = get_map_from_points(self.cat_data, self.nside,
                                            ra_name=self.ra_name,
                                            dec_name=self.dec_name)
            N_mean = np.average(nmap_data, weights=self.mask)
            N_mean_srad = N_mean * self.npix / (4 * np.pi)
            N_ell = np.mean(self.mask) / N_mean_srad
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled

    # for all mappers
    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
