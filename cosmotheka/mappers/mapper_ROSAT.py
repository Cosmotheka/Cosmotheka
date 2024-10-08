from .utils import get_map_from_points, rotate_map, rotate_mask
from .mapper_base import MapperBase
import fitsio
import numpy as np
import healpy as hp


class MapperROSATXray(MapperBase):
    """ Implements X-ray count rate maps from ROSAT.

    A few details so we can document this properly in the future:

    Photon list was obtained from:
    http://dc.zah.uni-heidelberg.de/tableinfo/rosat.photons

    Exposure maps were obtained from e.g.:
    https://heasarc.gsfc.nasa.gov/FTP/rosat/data/\
    pspc/processed_data/rass/release/rs931844n00/

    The signal map in this case is the count rate density in
    units of counts/second/sr^-1. The model for this would
    need the effective instrument area. For ROSAT this is an
    energy-dependent quantity which includes a transfer function
    for E_true vs. E_channel. These can be found in
    https://heasarc.gsfc.nasa.gov/docs/rosat/pspc_matrices.html
    but we don't need them and won't provide them here.

    The mask is constructed by thresholding the exposure map
    and combining it with any other externa map (e.g. aiming
    to remove Galactic emission).

    **Config**

        - exposure_map: `'.../Datasets/ROSAT/exposure/exposure_mean.fits'`
        - photon_list: `'.../Datasets/ROSAT/rosat_photons_Egt0p4keV.fits'`
        - energy_range: `[0.5, 3.0]`
        - exposure_min: `100.0`
        - external_mask: \
        `'.../Datasets/2MPZ_WIxSC/WISExSCOSmask_equatorial.fits.gz'`
        - mask_name: `'mask_ROSAT'`
        - mapper_class: `'MapperROSATXray'`
    """
    map_name = 'ROSATXray'

    def __init__(self, config):
        self._get_defaults(config)
        self.rot = self._get_rotator('C')
        self.fname_expmap = config['exposure_map']
        self.fname_pholist = config['photon_list']
        self.fname_pscat = config['point_source_catalog']
        self.erange = config.get('energy_range', [0.5, 3.0])
        self.explimit = config.get('exposure_min', 100.0)
        self.mask_external = config.get('external_mask', None)
        self.mask_ps = config.get('mask_point_sources', False)

        # Nside for point source masking. This is fixed because the
        # corresponding pixel size is similar to the ROSAT PSF.
        self.nside_ps = 2048
        self.count_map = None
        self.expmap = None
        self.pholist = None
        # signal_map is a map of the countrate
        self.nl_coupled = None
        self.ps_mask = None

    def _get_ps_mask(self):
        """ Returns point-source mask at nside=2048
        """
        if self.ps_mask is None:
            f = fitsio.FITS(self.fname_pscat)
            cat = f[1].read()
            # Flux cut
            cat = cat[cat['RATE'] >
                      self.config['point_source_flux_cut_ctps']]
            nmap = get_map_from_points(cat, self.nside_ps,
                                       ra_name='RA_DEG',
                                       dec_name='DEC_DEG',
                                       rot=self.rot)
            self.ps_mask = (nmap == 0).astype(float)
        return self.ps_mask

    def _get_pholist(self):
        """
        Returns the mapper's catalog \
        after applying energy boundaries. \

        Returns:
            pholist (Arrays): catalog

        """
        if self.pholist is None:
            f = fitsio.FITS(self.fname_pholist)
            cat = f[1].read()
            msk = ((cat['energy_cor'] < self.erange[1]) &
                   (cat['energy_cor'] > self.erange[0]))
            self.pholist = cat[msk]
        return self.pholist

    def get_expmap(self):
        """
        Returns the mapper exposure map. \

        Returns:
            expmap (Array)

        """
        if self.expmap is None:
            mp = hp.read_map(self.fname_expmap)
            mp = rotate_map(mp, self.rot)
            self.expmap = hp.ud_grade(mp, nside_out=self.nside)
        return self.expmap

    def _get_count_map(self):
        if self.count_map is None:
            cat = self._get_pholist()
            if self.mask_ps:
                count_map = get_map_from_points(cat, self.nside_ps,
                                                ra_name='raj2000',
                                                dec_name='dej2000',
                                                rot=self.rot)
                # Mask point sources
                ps_mask = self._get_ps_mask()
                count_map = count_map * ps_mask
                # Down/up grade to desired resolution
                count_map = hp.ud_grade(count_map, nside_out=self.nside,
                                        power=-2)
                # Correct for loss of area
                ps_mask = hp.ud_grade(ps_mask, nside_out=self.nside)
                good = ps_mask > 0
                count_map[good] /= ps_mask[good]
            else:
                count_map = get_map_from_points(cat, self.nside,
                                                ra_name='raj2000',
                                                dec_name='dej2000',
                                                rot=self.rot)
            self.count_map = count_map
        return self.count_map

    def _get_signal_map(self):
        xpmap = self.get_expmap()
        mask = self.get_mask()
        count_map = self._get_count_map()
        signal_map = np.zeros(self.npix)
        goodpix = mask > 0.0
        signal_map[goodpix] = count_map[goodpix] / xpmap[goodpix]
        pixA = hp.nside2pixarea(self.nside)
        signal_map *= 1/pixA
        signal_map = np.array([signal_map])
        return signal_map

    def _get_mask(self):
        xpmap = self.get_expmap()
        mask = xpmap.copy()
        mask[xpmap <= self.explimit] = 0
        if self.mask_external is not None:
            msk = hp.read_map(self.mask_external)
            msk = rotate_mask(msk, self.rot, binarize=True)
            msk = hp.ud_grade(msk, nside_out=self.nside)
            mask *= msk
        if self.mask_ps:
            # Mask point sources
            ps_mask = hp.ud_grade(self._get_ps_mask(),
                                  nside_out=self.nside)
            mask *= ps_mask
        return mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            xpmap = self.get_expmap()
            mask = self.get_mask()
            count_map = self._get_count_map()
            goodpix = mask > 0.0
            # Mean count rate
            # CR_mean = \sum n_p / \sum exp_p
            cr_mean = np.sum(count_map[goodpix])/np.sum(xpmap[goodpix])
            # <mask^2/exposure>
            m2_ie = np.sum(mask[goodpix]**2/xpmap[goodpix]) / self.npix
            # Pixel area
            pixA = hp.nside2pixarea(self.nside)
            N_mean = cr_mean * m2_ie / pixA
            self.nl_coupled = N_mean * np.ones([1, 3*self.nside])
        return self.nl_coupled

    def get_dtype(self):
        return 'generic'

    def get_spin(self):
        return 0
