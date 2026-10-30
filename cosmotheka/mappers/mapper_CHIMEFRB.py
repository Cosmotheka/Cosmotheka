from .mapper_base import MapperBase
from .utils import get_map_from_points, rotate_mask
import fitsio
import numpy as np
import pymaster as nmt
import healpy as hp


class MapperCHIMEFRB(MapperBase):
    """
    Mapper for the dispersion measure (DM) of fast radio bursts (FRBs)
    in the second CHIME/FRB catalog (https://arxiv.org/abs/2601.09399,
    https://chime-frb.ca/catalogs/catalog2).

    The field represented is the mean DM of the FRBs in a given direction,
    with the sample mean subtracted. As in the cosmic shear mappers, the
    mask is the (weighted) number of FRBs in each pixel, and the signal
    map is the weighted mean of (DM - <DM>) in each pixel, so that
    mask * signal is the sum of DM fluctuations over all FRBs in each
    pixel. The noise power spectrum is the corresponding shot noise.

    The default sample contains one entry per FRB source, removing
    secondary sub-bursts (`sub_num > 0`), events detected during
    non-nominal telescope operation (`excluded_flag = 1`) and events
    detected in the beam sidelobes (`sidelobe_flag = 1`). Bursts without
    a fitted DM (`dm_fitb`) are also removed.

    **Config**

        - data_catalog: `'.../Datasets/CHIME/chimefrbcat2.fits'`
        - galactic_model: `'ne2001'` (default) / `'ymw16'` / `None`. \
        Model used to subtract the Milky Way contribution to the DM. \
        If `None`, the total observed DM (`dm_fitb`) is used.
        - repeaters: `'average'` (default; one entry per repeating \
        source, with its DM averaged over all its bursts and the \
        position of its first burst) / `'exclude'` / `'all'` (one \
        entry per burst)
        - remove_sidelobe: `True` (default)
        - remove_excluded: `True` (default)
        - dm_range: `[0, 5000]` (optional; DM range in pc/cm^3)
        - gb_min: `10.0` (optional; minimum absolute Galactic \
        latitude in degrees)
        - external_mask: `'...'` (optional; binary mask in \
        equatorial coordinates)
        - mask_name: `'mask_CHIMEFRB'`
        - path_rerun: `'.../Datasets/CHIME/xcell_reruns/'`
    """
    map_name = 'CHIMEFRB'
    dtype = 'generic'
    spin = 0

    def __init__(self, config):
        self._get_defaults(config)
        self.rot = self._get_rotator('C')
        self.galactic_model = config.get('galactic_model', 'ne2001')
        if self.galactic_model in [None, 'None', 'none']:
            self.galactic_model = None
            self.dm_column = 'dm_fitb'
        elif self.galactic_model in ['ne2001', 'ymw16']:
            # DM excess over the Milky Way contribution
            self.dm_column = f'dm_exc_{self.galactic_model}'
        else:
            raise ValueError("Unknown Galactic DM model "
                             f"{self.galactic_model}. Must be "
                             "'ne2001', 'ymw16' or None")
        self.repeaters = config.get('repeaters', 'average')
        if self.repeaters not in ['average', 'exclude', 'all']:
            raise ValueError(f"Unknown repeater option {self.repeaters}. "
                             "Must be 'average', 'exclude' or 'all'")
        self.remove_sidelobe = config.get('remove_sidelobe', True)
        self.remove_excluded = config.get('remove_excluded', True)
        self.dm_range = config.get('dm_range', None)
        self.gb_min = config.get('gb_min', None)
        self.mask_external = config.get('external_mask', None)
        self.map_name += f"_MW{self.galactic_model}_rep{self.repeaters}"

        self.cat_data = None
        self.dm_mean = None
        self.nl_coupled = None

    def get_catalog(self):
        """
        Returns the FRB catalog after applying all cuts, with \
        columns `RA`, `DEC` (in degrees) and `DM`.

        Returns:
            cat_data (Array)
        """
        if self.cat_data is None:
            d = fitsio.read(self.config['data_catalog'], ext=1)
            # One entry per burst (sub-bursts share position and DM)
            sel = d['sub_num'] == 0
            if self.remove_excluded:
                sel &= d['excluded_flag'] == 0
            if self.remove_sidelobe:
                sel &= d['sidelobe_flag'] == 0
            # Some events have no localisation
            sel &= np.isfinite(d['ra']) & np.isfinite(d['dec'])
            # Remove bursts where the fitburst pipeline failed. Their
            # DM excesses are based on the real-time detection DM, so
            # we drop them for all Galactic models for consistency.
            sel &= np.isfinite(d['dm_fitb'])
            if self.gb_min is not None:
                sel &= np.fabs(d['gb']) >= self.gb_min
            d = d[sel]

            rep = np.char.strip(d['repeater_name'].astype(str))
            is_rep = rep != ''
            ra = d['ra'][~is_rep]
            dec = d['dec'][~is_rep]
            dm = d[self.dm_column][~is_rep]
            if self.repeaters == 'all':
                ra = np.concatenate([ra, d['ra'][is_rep]])
                dec = np.concatenate([dec, d['dec'][is_rep]])
                dm = np.concatenate([dm, d[self.dm_column][is_rep]])
            elif self.repeaters == 'average':
                ra_r, dec_r, dm_r = [], [], []
                for name in np.unique(rep[is_rep]):
                    ids = np.where(rep == name)[0]
                    i0 = ids[np.argmin(d['mjd_400'][ids])]
                    ra_r.append(d['ra'][i0])
                    dec_r.append(d['dec'][i0])
                    dm_r.append(np.mean(d[self.dm_column][ids]))
                ra = np.concatenate([ra, ra_r])
                dec = np.concatenate([dec, dec_r])
                dm = np.concatenate([dm, dm_r])

            # DM cuts are applied after averaging over repeaters
            if self.dm_range is not None:
                good = (dm >= self.dm_range[0]) & (dm < self.dm_range[1])
                ra, dec, dm = ra[good], dec[good], dm[good]

            cat = np.zeros(len(dm), dtype=[('RA', 'f8'), ('DEC', 'f8'),
                                           ('DM', 'f8'), ('weight', 'f8')])
            cat['RA'] = ra
            cat['DEC'] = dec
            cat['DM'] = dm
            # TODO: should we consider some form of inverse-variance weighting?
            cat['weight'] = 1.0
            self.cat_data = cat
        return self.cat_data

    def get_mean_dm(self):
        """
        Returns the weighted mean DM of the sample, which \
        is subtracted from the signal map.

        Returns:
            dm_mean (float)
        """
        if self.dm_mean is None:
            cat = self.get_catalog()
            self.dm_mean = np.average(cat['DM'], weights=cat['weight'])
        return self.dm_mean

    def _get_ddm(self):
        # DM fluctuations of each FRB
        return self.get_catalog()['DM'] - self.get_mean_dm()

    def _get_external_mask(self):
        msk = hp.read_map(self.mask_external)
        msk = rotate_mask(msk, self.rot, binarize=True)
        return hp.ud_grade(msk, nside_out=self.nside)

    def _get_signal_map(self):
        cat = self.get_catalog()
        mask = self.get_mask()
        signal_map = get_map_from_points(cat, self.nside,
                                         w=cat['weight']*self._get_ddm(),
                                         rot=self.rot)
        goodpix = mask > 0
        signal_map[goodpix] /= mask[goodpix]
        signal_map[~goodpix] = 0
        return np.array([signal_map])

    def _get_mask(self):
        # Returns the weights map
        cat = self.get_catalog()
        mask = get_map_from_points(cat, self.nside, w=cat['weight'],
                                   rot=self.rot)
        if self.mask_external is not None:
            mask *= self._get_external_mask()
        return mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            cat = self.get_catalog()
            w2s2 = get_map_from_points(cat, self.nside,
                                       w=(cat['weight']*self._get_ddm())**2,
                                       rot=self.rot)
            w2s2[self.get_mask() == 0] = 0
            N_ell = hp.nside2pixarea(self.nside) * np.mean(w2s2)
            self.nl_coupled = N_ell * np.ones([1, 3*self.nside])
        return self.nl_coupled

    def _get_nmt_catalog_field(self, **kwargs):
        """
        Build a catalog-based NaMaster field for the FRB DMs.
        """
        cat = self.get_catalog()
        ddm = self._get_ddm()
        weights = cat['weight'].copy()
        lon, lat = cat['RA'], cat['DEC']
        if self.rot is not None:
            lon, lat = self.rot(lon, lat, lonlat=True)
            lon = np.mod(lon, 360.)
        if self.mask_external is not None:
            msk = self._get_external_mask()
            ipix = hp.ang2pix(self.nside, lon, lat, lonlat=True)
            weights *= msk[ipix]
        positions = np.array([lon, lat])

        # Retrieve the maximum multipole
        lmax = kwargs.get("lmax", 3*self.nside-1)

        return nmt.NmtFieldCatalog(positions, weights, ddm,
                                   spin=0, lmax=lmax, lonlat=True)
