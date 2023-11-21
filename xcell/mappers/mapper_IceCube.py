from .mapper_base import MapperBase
from .utils import get_map_from_points, rotate_mask
from astropy.table import Table
import numpy as np
import healpy as hp
from scipy.interpolate import interp1d


class MapperIceCube(MapperBase):
    def __init__(self, config):
        self._get_defaults(config)
        self.npix = hp.nside2npix(self.nside)
        self.seasons = config.get('seasons',
                                  ['40', '59', '79', '86_I', '86_II',
                                   '86_III', '86_IV', '86_V', '86_VI',
                                   '86_VII'])
        self.lE_range = config['lE_range']
        self.nseasons = len(self.seasons)
        self.rot = self._get_rotator('C')
        self.ra_name = 'RA[deg]'
        self.dec_name = 'Dec[deg]'
        self.map_energy = config.get('map_energy', False)
        self.E_name = 'log10(E/GeV)'
        self.cat_data = np.full(self.nseasons, None)
        self.flux_map = None
        self.nl_coupled = None

    def _get_events(self, season):
        if self.cat_data[season] is None:
            # loads in data
            event_dir = self.config['event_dir']
            event_name = f'{event_dir}/IC{self.seasons[season]}_exp.csv'
            self.cats_data = Table.read(event_name, format='ascii')
            # pick those in energy bin
            lE = self.cats_data[self.E_name]
            self.cat_data[season] = self.cats_data[(lE >= self.lE_range[0]) &
                                                   (lE < self.lE_range[1])]
        return self.cat_data[season]

    def _get_aeff(self, season):
        # loads in data
        Aeff_dir = self.config['Aeff_dir']
        if season >= 4:
            season_name = f'{Aeff_dir}/IC86_II_effectiveArea.csv'
        else:
            season_name = (f'{Aeff_dir}/IC{self.seasons[season]}' +
                           '_effectiveArea.csv')
        logE_min, logE_max, Dec_min, Dec_max, Aeff = np.loadtxt(season_name,
                                                                unpack=True,
                                                                skiprows=1)
        # convert Aeff to m^2
        Aeff *= 10**(-4)
        # get unique E values
        logE_min_u = np.unique(logE_min)
        logE_max_u = np.unique(logE_max)
        lenE = len(logE_min_u)
        assert lenE == len(logE_max_u)
        # get unique dec values
        decvals = (Dec_min + Dec_max)/2
        sindecvals_u = np.sin(np.radians(np.unique(decvals)))
        lenDec = len(sindecvals_u)
        # convert aeff to rectangular matrix
        assert lenE*lenDec == len(Aeff)
        Aeff = Aeff.reshape([lenDec, lenE]).T
        # get effective area interpolation function for each energy bin
        lE_l, lE_r = self.lE_range
        # only keeps relevant fine energy bins
        msk = (logE_min_u > lE_l) & (logE_max_u < lE_r)
        Aeff_h = list(Aeff[msk])
        lEmin_h = list(logE_min_u[msk])
        lEmax_h = list(logE_max_u[msk])
        Aeff_h.insert(0, Aeff[np.min(np.where(msk)) - 1])
        Aeff_h.append(Aeff[np.max(np.where(msk)) + 1])
        Aeff_h = np.array(Aeff_h)
        lEmax_h.insert(0, lEmin_h[0])
        lEmin_h.insert(0, lE_l)
        lEmin_h.append(lEmax_h[-1])
        lEmax_h.append(lE_r)
        lEmin_h = np.array(lEmin_h)
        lEmax_h = np.array(lEmax_h)
        # calculates weighted effective area
        alpha = self.config.get('alpha', 3.7)
        wAeff = np.sum(Aeff_h[:, :]*(10**(lEmax_h*(1-alpha)) -
                                     10**(lEmin_h*(1-alpha)))[:, None],
                       axis=0)/(10**(lE_r*(1-alpha)) -
                                10**(lE_l*(1-alpha)))
        # creates interpolation function
        Aeff_inter = interp1d(sindecvals_u, wAeff,
                              bounds_error=False, fill_value=0.0)
        return Aeff_inter

    def _get_flux_and_mask(self):
        # creates base maps and inverse Aeff sums
        n_sum_map = np.zeros(self.npix)
        aeff_sum_map = np.zeros(self.npix)
        mask_bin = np.ones(self.npix, dtype=bool)
        # creates number count maps for each energy bin
        for i in range(self.nseasons):
            cats = self._get_events(i)
            Aeff_i = self._get_aeff(i)
            if self.map_energy:
                w = 10**cats[self.E_name]
            else:
                w = None
            ncount = get_map_from_points(cats, self.nside,
                                         ra_name=self.ra_name,
                                         dec_name=self.dec_name,
                                         rot=self.rot, w=w)
            mskbin, Aeff_map = self._get_aeff_mask(Aeff_i)
            n_sum_map[mskbin] += ncount[mskbin]
            aeff_sum_map[mskbin] += Aeff_map[mskbin]
            mask_bin *= mskbin
        # creates flux map
        goodpix = mask_bin * (aeff_sum_map > 0)
        flux = np.zeros(self.npix)
        flux[goodpix] = n_sum_map[goodpix]/aeff_sum_map[goodpix]
        time = 1.0
        Ompix = 4*np.pi/self.npix
        flux /= time*Ompix
        # mask propto area
        mask = mask_bin*aeff_sum_map/np.amax(aeff_sum_map)
        # Remove monopole
        flux_mean = np.sum(flux*mask)/np.sum(mask)
        flux[goodpix] = flux[goodpix]-flux_mean
        return flux, mask, n_sum_map, aeff_sum_map

    def _get_aeff_mask(self, Aeff):
        # finds dec of all map pixels
        _, dec = hp.pix2ang(self.nside, np.arange(self.npix), lonlat=True)
        # builds Aeff maps and masks
        Aeff_map = Aeff(np.sin(np.radians(dec)))
        # removes pixels below declination threshold
        Dec_min = self.config.get('Dec_min', -5)
        Aeff_map[dec < Dec_min] = 0
        # Rotate to desired coordinates
        Aeff_map = rotate_mask(Aeff_map, self.rot)
        # Threshold it
        A_thr = self.config.get('Aeff_threshold', 0.1)
        Aeff_mask = Aeff_map > np.amax(Aeff_map)*A_thr
        return Aeff_mask, Aeff_map

    def _get_mask(self):
        return self._get_flux_and_mask()[1]

    def get_signal_map(self):
        if self.flux_map is None:
            self.flux_map = self._get_flux_and_mask()[0]
        return [self.flux_map]

    def get_nl_coupled(self):
        return np.zeros((1, 3*self.nside))
        #if self.nl_coupled is None:
        #    nmap, mask = self._get_nmap_and_mask()
        #    nmean = np.sum(nmap*mask)/np.sum(mask)
        #    nmean_srad = nmean*self.npix/(4*np.pi)
        #    N_ell = np.mean(mask)/nmean_srad
        #    self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        #return self.nl_coupled

    def get_spin(self):
        return 0

    def get_dtype(self):
        return 'generic'
