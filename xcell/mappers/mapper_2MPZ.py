from .mapper_base import MapperBase
from .utils import get_map_from_points, get_DIR_Nz
import fitsio
import numpy as np
import healpy as hp
import os


class Mapper2MPZ(MapperBase):
    """
    **Config**

        - mask_name: `mask_2MPZWISC`
        - path_rerun: `'.../Datasets/2MPZ_WIxSC/xcell_runs'`
        - z_edges: `[0.0, 0.1]`
        - data_catalog: `'.../Datasets/2MPZ_WIxSC/2MPZ.fits'`
        - n_jk_dir: `100`
        - mask_G: `'.../Datasets/2MPZ_WIxSC/WISExSCOSmask_galactic.fits.gz'`
        - mask_C: `'.../Datasets/2MPZ_WIxSC/WISExSCOSmask_equatorial.fits.gz'`
        - use_halo_model: `True`
        - hod_params:

              - lMmin_0: `12.708493552845066`
              - siglM_0: `0.345`
              - lM0_0: `12.708493552845066`
              - lM1_0: `14.260818060157721`
              - alpha_0: `1.0`
              - fc_0: `1.0`
    """
    def __init__(self, config):
        self._get_defaults(config)
        self.z_edges = config.get('z_edges', [0, 0.5])
        self.ra_name, self.dec_name = self._get_coords(config)

        self.cat_data = None
        self.npix = hp.nside2npix(self.nside)

        # Angular mask
        self.dndz = None
        self.delta_map = None
        self.nl_coupled = None

    def _get_coords(self, config):
        # Returns mapper's ra and dec names \
        # in the mapper's catalog given the coordinates \
        # designated in the cofiguration file

        # Returns:
        #     ra_name (String), dec_name (String)

        if self.coords == 'G':  # Galactic
            return 'L', 'B'
        elif self.coords == 'C':  # Celestial/Equatorial
            return 'SUPRA', 'SUPDEC'
        else:
            raise NotImplementedError(f"Unknown coordinates {self.coords}")

    def get_catalog(self):
        """
        Returns the mapper catalog of sources.

        Returns:
            catalog (Array)
        """
        if self.cat_data is None:
            file_data = self.config['data_catalog']
            if not os.path.isfile(file_data):
                raise ValueError(f"File {file_data} not found")
            self.cat_data = fitsio.read(file_data)
            self.cat_data = self._bin_z(self.cat_data)
            self.cat_data = self._mask_catalog(self.cat_data)

        return self.cat_data

    def _mask_catalog(self, cat):
        # Applies binary mask to catalog

        self.mask = self.get_mask()
        ipix = hp.ang2pix(self.nside, cat[self.ra_name],
                          cat[self.dec_name], lonlat=True)
        # Mask is binary, so 0.1 or 0.00001 doesn't really matter.
        return cat[self.mask[ipix] > 0.1]

    def _bin_z(self, cat):
        # Removes all but the catalog sources \
        # inside the chosen redshift bin.

        return cat[(cat['ZPHOTO'] > self.z_edges[0]) &
                   (cat['ZPHOTO'] <= self.z_edges[1])]

    def _get_specsample(self, cat):
        # Selects the spectroscopic samples \
        # in the catalog

        ids = cat['ZSPEC'] > -1
        return cat[ids]

    def _get_nz(self):
        # Employs the DIR algorithm to build \
        # the sources redshift distribution of the \
        # catalog and returns it in the shape of a \
        # dictionary containing "z_mid"; the mid redshift \
        # of the redshift histogram, "nz"; the heights of \
        # the histogram, and "nz_jk".

        c_p = self.get_catalog()
        c_s = self._get_specsample(c_p)
        # Sort spec sample by nested pixel index so jackknife
        # samples are spatially correlated.
        ip_s = hp.ring2nest(self.nside,
                            hp.ang2pix(self.nside,
                                       c_s[self.ra_name],
                                       c_s[self.dec_name],
                                       lonlat=True))
        idsort = np.argsort(ip_s)
        c_s = c_s[idsort]
        # Compute DIR N(z)
        z, nz, nz_jk = get_DIR_Nz(c_s, c_p,
                                  ['JCORR', 'KCORR', 'HCORR',
                                   'W1MCORR', 'W2MCORR',
                                   'BCALCORR', 'RCALCORR', 'ICALCORR'],
                                  zflag='ZSPEC',
                                  zrange=[0, 0.4],
                                  nz=100,
                                  njk=self.config.get('n_jk_dir', 100))
        zm = 0.5*(z[1:] + z[:-1])
        return {'z_mid': zm, 'nz': nz, 'nz_jk': nz_jk}

    def get_nz(self, dz=0, return_jk_error=False):
        """
        Checks if mapper has precomputed the redshift \
        distribution. If not, it uses "_get_nz()" to obtain it. \
        Then, it shifts the distribution by "dz" (default dz=0).

        Kwargs:
            dz=0
            return_jk_error=False

        Returns:
            [z, nz] (Array)
        """
        if self.dndz is None:
            fn = 'nz_2MPZ.npz'
            self.dndz = self._rerun_read_cycle(fn, 'NPZ', self._get_nz)
        return self._get_shifted_nz(dz, return_jk_error=return_jk_error)

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
            self.delta_map = np.array([d])
        return self.delta_map

    def _get_mask(self):
        # Reads the mask of the mappper from a file \
        # and upgrades it to the chosen resolution.

        # We will assume the mask has been provided in the right
        # coordinates, so no further conversion is needed.
        mask = hp.ud_grade(hp.read_map(self.config['mask']),
                           nside_out=self.nside)
        return mask

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

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
