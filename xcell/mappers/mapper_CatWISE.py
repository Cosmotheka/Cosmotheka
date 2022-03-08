from .mapper_base import MapperBase
from .utils import get_map_from_points
from astropy.table import Table
import numpy as np
import healpy as hp


class MapperCatWISE(MapperBase):
    def __init__(self, config):
        """
        config - dict
          {'data_catalog': 'xcell/tests/data/
          catwise_agns_masked_final_w1lt16p5_alpha.fits',
           'mask': 'xcell/tests/data/MASKS_exclude_master_final.fits',
           'mask_name': 'mask_CatWISE'}
        """
        self._get_defaults(config)
        self.file_sourcemask = config.get('mask_sources', None)
        self.cat_data = None

        self.npix = hp.nside2npix(self.nside)
        # Angular mask
        self.mask = None
        self.delta_map = None
        self.nl_coupled = None
        self.dndz = None
        # self.cat_redshift = None

    # CatWISE catalog
    def get_catalog(self):
        if self.cat_data is None:
            file_data = self.config['data_catalog']
            self.cat_data = Table.read(file_data)
            # Flux condition
            self.cat_data = self.cat_data[
                (self.cat_data['w1'] <
                 self.config.get('flux_max_W1', 16.4))]
        return self.cat_data

    # Density Map
    def get_signal_map(self, apply_ecliptic_correction=True):
        if self.delta_map is None:
            d = np.zeros(self.npix)
            self.cat_data = self.get_catalog()
            self.mask = self.get_mask()
            nmap_data = get_map_from_points(self.cat_data, self.nside,
                                            ra_name='ra',
                                            dec_name='dec')
            goodpix = self.mask > 0
            # ecliptic latitude correction -- SvH 5/3/22
            if apply_ecliptic_correction:
                # The fit that lead to the ecl. lat. correction was
                # done on the density map not the density contrast,
                # hence introduce the density here.
                dens = np.zeros(self.npix)
                pixelarea_deg2 = (hp.nside2resol(self.nside, arcmin=True)/60)**2
                dens[goodpix] = (nmap_data[goodpix] / 
                                 pixelarea_deg2) * self.mask[goodpix]
                # Transforms equatorial to ecliptic coordinates
                r = hp.Rotator(coord=['C', 'E'])
                # Get equatorial coordinates
                theta_EQ, phi_EQ = hp.pix2ang(self.nside,
                                              np.arange(hp.nside2npix(self.nside)))
                # Rotate to ecliptic
                theta_EC, phi_EC = r(theta_EQ, phi_EQ)
                # Make a map of ecliptic latitude
                ec_lat_map = 90-np.degrees(theta_EC)
                # this hard-coded number stems from the fit in 2009.14826
                dens += 0.0513 * np.abs(ec_lat_map)
                # modify the number counts per pixel before computing 
                # density contrast below
                nmap_data = dens * pixelarea_deg2
            #
            mean_n = np.average(nmap_data, weights=self.mask)
            # Division by mask not really necessary, since it's binary.
            d[goodpix] = nmap_data[goodpix]/(mean_n*self.mask[goodpix])-1
            self.delta_map = np.array([d])
        return self.delta_map

    def _cut_mask(self):
        mask = np.ones(self.npix)
        r = hp.Rotator(coord=['C', 'G'])
        RApix, DEpix = hp.pix2ang(self.nside, np.arange(self.npix),
                                  lonlat=True)
        lpix, bpix = r(RApix, DEpix, lonlat=True)
        # angular conditions
        mask[(np.fabs(bpix) < self.config.get('GLAT_max_deg',
                                              30))] = 0
        if self.file_sourcemask is not None:
            # holes catalog
            mask_holes = Table.read(self.file_sourcemask)
            vecmask = hp.ang2vec(mask_holes['ra'],
                                 mask_holes['dec'],
                                 lonlat=True)
            for vec, radius in zip(vecmask,
                                   mask_holes['radius']):
                ipix_hole = hp.query_disc(self.nside, vec,
                                          np.radians(radius),
                                          inclusive=True)
                mask[ipix_hole] = 0
        return mask

    def get_mask(self):
        if self.mask is None:
            if self.config.get('mask_file', None) is not None:
                self.mask = hp.ud_grade(hp.read_map(self.config['mask_file']),
                                        nside_out=self.nside)
            else:
                fn = f'CatWise_cutout_mask_ns{self.nside}.fits.gz'
                self.mask = self._rerun_read_cycle(fn, 'FITSMap',
                                                   self._cut_mask)
        return self.mask

    # Shot noise
    def get_nl_coupled(self):
        if self.nl_coupled is None:
            self.cat_data = self.get_catalog()
            self.mask = self.get_mask()
            nmap_data = get_map_from_points(self.cat_data, self.nside,
                                            ra_name='ra',
                                            dec_name='dec')
            N_mean = np.average(nmap_data, weights=self.mask)
            N_mean_srad = N_mean * self.npix / (4 * np.pi)
            N_ell = np.mean(self.mask) / N_mean_srad
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled

    def get_nz(self, dz=0):
        raise NotImplementedError("No dNdz for CatWISE yet")

    # Type
    def get_dtype(self):
        return 'galaxy_density'

    # Spin
    def get_spin(self):
        return 0
