from .mapper_base import MapperBase
from .utils import get_map_from_points, rotate_mask
from astropy.table import Table
import numpy as np
import healpy as hp


class MapperCatWISE(MapperBase):
    """
    **Config**

        - data_catalog: \
          `".../Datasets/CatWISE/catwise_agns_masked_final_w1lt16p5_alpha.fits"`
        - file_mask: \
          `".../Datasets/CatWISE/MASKS_exclude_master_final.fits"`
        - apply_ecliptic_correction: `True`
        - mask_name: `"mask_CatWISE"`
        - path_rerun: `".../Datasets/CatWISE/xcell_runs"`
    """
    def __init__(self, config):
        self._get_defaults(config)
        self.file_sourcemask = config.get('mask_sources', None)
        self.apply_ecliptic_correction = \
            config.get('apply_ecliptic_correction', True)
        self.cat_data = None

        self.npix = hp.nside2npix(self.nside)
        # Angular mask
        self.delta_map = None
        self.nl_coupled = None
        self.dndz = None
        self.rot = self._get_rotator('C')

    def get_catalog(self):
        """
        Returns the mapper's catalog after \
        applying flux thershold.

        Returns:
            catalog (Array)
        """
        if self.cat_data is None:
            file_data = self.config['data_catalog']
            self.cat_data = Table.read(file_data)
            # Flux condition
            self.cat_data = self.cat_data[
                (self.cat_data['w1'] <
                 self.config.get('flux_max_W1', 16.4))]
        return self.cat_data

    def _get_ecliptic_correction(self):
        # Correction to Density

        pixarea_deg2 = (hp.nside2resol(self.nside, arcmin=True)/60)**2
        # Transforms equatorial to ecliptic coordinates
        r = hp.Rotator(coord=[self.coords, 'E'])
        # Get coordinates in system of choice
        theta_EQ, phi_EQ = hp.pix2ang(self.nside, np.arange(self.npix))
        # Rotate to ecliptic
        theta_EC, phi_EC = r(theta_EQ, phi_EQ)
        # Make a map of ecliptic latitude
        ec_lat_map = 90-np.degrees(theta_EC)
        # this hard-coded number stems from the fit in 2009.14826
        correction = 0.0513 * np.abs(ec_lat_map) * pixarea_deg2
        return correction

    def get_signal_map(self):
        if self.delta_map is None:
            d = np.zeros(self.npix)
            self.cat_data = self.get_catalog()
            self.mask = self.get_mask()
            nmap_data = get_map_from_points(self.cat_data, self.nside,
                                            rot=self.rot, ra_name='ra',
                                            dec_name='dec')
            # ecliptic latitude correction -- SvH 5/3/22
            if self.apply_ecliptic_correction:
                correction = self._get_ecliptic_correction()
            else:
                correction = np.zeros_like(d)
            nmap_data = nmap_data + correction
            goodpix = self.mask > 0
            mean_n = np.average(nmap_data, weights=self.mask)
            # Division by mask not really necessary, since it's binary.
            d[goodpix] = nmap_data[goodpix]/(mean_n*self.mask[goodpix])-1
            self.delta_map = np.array([d])
        return self.delta_map

    def _cut_mask(self):
        # Generates the mask given the chosen resolution \
        # and the angular conditions in the configuration \
        # file. If "file_sourcemask" is not None it applies \
        # holes to the mask.

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

    def _get_mask(self):
        # Checks if the mapper has already computed \
        # the mask. If so, it loads it from a file. \
        # Otherwise, it calculates using "_cut_mask()". \
        # It also rotates the mask to the chose coordinates.

        if self.config.get('mask_file', None) is not None:
            mask = hp.ud_grade(hp.read_map(self.config['mask_file']),
                               nside_out=self.nside)
        else:
            mask = self._cut_mask()
        mask = rotate_mask(mask, self.rot, binarize=True)
        return mask

    def get_nl_coupled(self):
        # Shot noise

        if self.nl_coupled is None:
            self.cat_data = self.get_catalog()
            self.mask = self.get_mask()
            nmap_data = get_map_from_points(self.cat_data, self.nside,
                                            rot=self.rot, ra_name='ra',
                                            dec_name='dec')
            N_mean = np.average(nmap_data, weights=self.mask)
            N_mean_srad = N_mean * self.npix / (4 * np.pi)
            N_ell = np.mean(self.mask) / N_mean_srad
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled

    def get_nz(self, dz=0):
        """
        Not implemented yet.
        """
        raise NotImplementedError("No dNdz for CatWISE yet")

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
