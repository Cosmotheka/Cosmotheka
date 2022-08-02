from .mapper_base import MapperBase
# from .utils import get_map_from_points, rotate_mask
from astropy.table import Table
import numpy as np
import healpy as hp
from scipy.interpolate import interp1d


class MapperIceCube(MapperBase):
    def __init__(self, config, logE_ranges=[np.log10(300), np.log10(300)+1, np.log10(300)+2, np.log10(300)+3]):

        self._get_defaults(config)
        self.npix = hp.nside2npix(self.nside)
        self.seasons = ['40', '59', '79', '86_I', '86_II',
                        '86_III', '86_IV', '86_V', '86_VI', '86_VII']
        self.lE_ranges = logE_ranges
        self.nEbins = len(self.lE_ranges) - 1
        self.nseasons = len(self.seasons)
        self.rot = self._get_rotator('G')
        self.r_c2g = hp.Rotator(coord=['C', 'G'])
        self.r_g2c = hp.Rotator(coord=['G', 'C'])
        self.ra_name = 'RA[deg]'
        self.dec_name = 'Dec[deg]'
        self.E_name = 'log10(E/GeV)'
        self.cat_data = np.full((self.nseasons, self.nEbins), None)
        self.LastMaskSeasons = None
        self.mask = None
        self.LastMapSeasons = None
        self.delta_map = np.full(self.nEbins, None)

    def _get_events(self, season):
        if None in self.cat_data[season]:
            # loads in data
            EventDir = self.config['EventDir']
            event_name = f'{EventDir}/IC{self.seasons[season]}_exp.csv'
            self.cats_data = Table.read(event_name, format='ascii')
            # split into energy bins
            for i in range(self.nEbins):
                self.cat_data[season][i] = self.cats_data[
                    (self.cats_data[self.E_name] >= self.lE_ranges[i]) &
                    (self.cats_data[self.E_name] < self.lE_ranges[i+1])]
        return self.cat_data[season]

    def _get_aeff(self, season):
        # loads in data
        AeffDir = self.config['AeffDir']
        if season >= 4:
            season_name = f'{AeffDir}/IC86_II_effectiveArea.csv'
        else:
            season_name = (f'{AeffDir}/IC{self.seasons[season]}' +
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
        Aeff_inters = []
        for i in range(self.nEbins):
            lE_l = self.lE_ranges[i]
            lE_r = self.lE_ranges[i+1]
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
            WeightedAeffs = np.sum(Aeff_h[:, :]*(10**(lEmax_h*(1-alpha)) -
                                   10**(lEmin_h*(1-alpha)))[:, None],
                                   axis=0)/(10**(lE_r*(1-alpha)) -
                                            10**(lE_l*(1-alpha)))
            # creates interpolation function
            Aeff_inter = interp1d(sindecvals_u, WeightedAeffs,
                                  bounds_error=False, fill_value=0.0)
            Aeff_inters.append(Aeff_inter)
        return Aeff_inters

    def _get_aeff_mask(self, Aeff):
        # finds dec of all map pixels
        lon, lat = hp.pix2ang(self.nside, np.arange(self.npix), lonlat=True)
        _, dec = self.r_g2c(lon, lat, lonlat=True)
        # builds Aeff maps and masks
        Aeff_map = Aeff(np.sin(np.radians(dec)))
        Aeff_mask = Aeff_map > np.amax(Aeff_map)*self.config.get(
            'Aeff_Threshold', 0.1)
        # removes pixels below declination threshold
        Aeff_mask[dec < self.config.get('ICDecMin', -5)] = 0
        Aeff_map[dec < self.config.get('ICDecMin', -5)] = 0
        return Aeff_mask, Aeff_map

    def get_mask(self, seasons='all'):
        if seasons == 'all':
            seasons = range(self.nseasons)
        if self.mask is None or seasons != self.LastMaskSeasons:
            # creates base mask
            self.mask = np.ones(self.npix, dtype=bool)
            for i in seasons:
                Aeff = self._get_aeff(i)
                # finds and combines aeff masks for each season and energy bin
                for j in range(self.nEbins):
                    AeffMask, _ = self._get_aeff_mask(Aeff[j])
                    self.mask *= AeffMask
            self.LastMaskSeasons = seasons
        return self.mask

    def get_signal_map(self, Ebin, seasons='all'):
        if seasons == 'all':
            seasons = range(self.nseasons)
        if self.delta_map[Ebin] is None or seasons != self.LastMapSeasons:
            # creates base maps and inverse Aeff sums
            nmap_t = np.zeros(self.npix)
            inv_aeff_t = np.zeros(self.npix)
            mask = self.get_mask(seasons)
            # creates number count maps for each energy bin
            for i in seasons:
                cats = self._get_events(i)
                Aeff_i = self._get_aeff(i)
                lon, lat = self.r_c2g(cats[Ebin][self.ra_name],
                                      cats[Ebin][self.dec_name],
                                      lonlat=True)
                ipix = hp.ang2pix(self.nside, lon, lat, lonlat=True)
                ncount = np.bincount(ipix, minlength=self.npix)
                _, AeffMap = self._get_aeff_mask(Aeff_i[Ebin])
                nmap_t[mask] += ncount[mask]/AeffMap[mask]
                inv_aeff_t[mask] += 1/AeffMap[mask]
            # creates delta maps and normalises with inv_aeff_t
            self.delta_map[Ebin] = np.zeros(self.npix)
            nmap = np.zeros(self.npix)
            nmap[mask] = nmap_t[mask]/inv_aeff_t[mask]
            nmean = np.sum(nmap*mask)/np.sum(mask)
            self.delta_map[Ebin] = (nmap/nmean-1)*mask
            self.LastMapSeasons = seasons
        return self.delta_map[Ebin]
