from .utils import get_map_from_points, rotate_mask
from .mapper_base import MapperBase
from astropy.table import Table, vstack
from scipy.integrate import simps
import numpy as np
import healpy as hp
import os


class MapperDELS(MapperBase):
    map_name = 'DELS'

    def __init__(self, config):
        """
        config - dict
          {'data_catalogs':['Legacy_Survey_BASS-MZLS_galaxies-selection.fits'],
           'zbin': 0,
           'z_name': 'PHOTOZ_3DINFER',
           'num_z_bins': 500,
           'binary_mask': 'Legacy_footprint_final_mask.fits',
           'completeness_map': 'Legacy_footprint_completeness_mask_128.fits',
           'star_map': 'allwise_total_rot_1024.fits',
           'nside': 1024,
           'mask_name': 'mask_DELS'}
        """
        self._get_defaults(config)
        self.rot = self._get_rotator('C')
        self.pz = config.get('z_name', 'PHOTOZ_3DINFER')
        self.num_z_bins = config.get('num_z_bins', 500)

        self.cat_data = None
        self.npix = hp.nside2npix(self.nside)

        bin_edges = [[0.00, 0.30],
                     [0.30, 0.45],
                     [0.45, 0.60],
                     [0.60, 0.80]]

        self.zbin = config['zbin']
        self.z_edges = bin_edges[self.zbin]
        self.map_name += f"_bin{self.zbin}"

        # Angular mask flag
        self.mskflag = None
        self.delta_map = None
        self.nl_coupled = None
        self.comp_map = None
        self.stars = None
        self.bmask = None

    def _get_catalog(self):
        cat = []
        for file_data in self.config['data_catalogs']:
            if not os.path.isfile(file_data):
                raise ValueError(f"File {file_data} not found")
            c = Table.read(file_data, format='fits')
            c.keep_columns(['RA', 'DEC', self.pz])
            cat.append(c)
        cat = vstack(cat)
        cat = self._bin_z(cat)
        return cat.as_array()

    def get_catalog(self):
        if self.cat_data is None:
            fn = f'{self.map_name}_cat.fits'
            self.cat_data = self._rerun_read_cycle(fn, 'FITSTable',
                                                   self._get_catalog)
        return self.cat_data

    def _get_angmask(self):
        if self.mskflag is None:
            cat = self.get_catalog()
            bmask = hp.read_map(self.config['binary_mask'])
            nside = hp.npix2nside(len(bmask))
            ipix = hp.ang2pix(nside, cat['RA'], cat['DEC'],
                              lonlat=True)
            self.mskflag = bmask[ipix] > 0.
        return self.mskflag

    def _bin_z(self, cat):
        return cat[(cat[self.pz] >= self.z_edges[0]) &
                   (cat[self.pz] < self.z_edges[1])]

    def _get_lorentzian(self, zz):
        # a m s
        params = np.array([[1.257, -0.0010, 0.0122],
                           [1.104, 0.0076, 0.0151],
                           [1.476, -0.0024, 0.0155],
                           [2.019, -0.0042, 0.0265]])
        [a, m, s] = params[self.zbin]
        return 1./(1+((zz[:, None]-zz[None, :]-m)/s)**2/(2*a))**a

    def _get_nz(self):
        cat_data = self.get_catalog()
        mskflag = self._get_angmask()
        h, b = np.histogram(cat_data[self.pz][mskflag],
                            range=[-0.3, 1], bins=self.num_z_bins)
        z_arr = 0.5 * (b[:-1] + b[1:])
        kernel = self._get_lorentzian(z_arr)
        nz_photo = h.astype(float)
        nz_spec = simps(kernel*nz_photo[None, :], x=z_arr, axis=1)
        nz_spec /= simps(nz_spec, x=z_arr)
        return {'z_mid': z_arr, 'nz': nz_spec}

    def get_nz(self, dz=0):
        if self.dndz is None:
            fn = f'{self.map_name}_dndz.npz'
            self.dndz = self._rerun_read_cycle(fn, 'NPZ', self._get_nz)
        return self._get_shifted_nz(dz)

    def _get_signal_map(self, apply_galactic_correction=True):
        d = np.zeros(self.npix)
        cat_data = self.get_catalog()
        comp_map = self._get_comp_map()
        bmask = self._get_binary_mask()
        stars = self._get_stars()
        nmap_data = get_map_from_points(cat_data, self.nside, rot=self.rot)
        mean_n = self._get_mean_n(nmap_data)
        goodpix = bmask > 0
        d[goodpix] = nmap_data[goodpix]/(mean_n*comp_map[goodpix])-1
        if apply_galactic_correction:
            gcorr = self._get_galactic_correction(d, stars, bmask)
            d -= gcorr['delta_map']
        delta_map = np.array([d])
        return delta_map

    def _get_galactic_correction(self, delta, stars, bmask, nbins=14, npoly=5):
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
        # Now create a 5-th order polynomial fitting these data
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

    def _get_mean_n(self, nmap):
        self.comp_map = self._get_comp_map()
        self.stars = self._get_stars()
        self.bmask = self._get_binary_mask()
        goodpix = self.comp_map > 0.95
        goodpix *= self.stars < 8515
        goodpix *= self.bmask > 0
        n_mean = np.sum(nmap[goodpix])/np.sum(self.comp_map[goodpix])
        return n_mean

    def _get_stars(self):
        if self.stars is None:
            # Power = -2 makes sure the total number of stars is conserved
            stars = hp.read_map(self.config['star_map'])
            stars = rotate_mask(stars, self.rot)
            self.stars = hp.ud_grade(stars, nside_out=self.nside,
                                     power=-2)
            # Convert to stars per deg^2
            pix_srad = 4*np.pi/self.npix
            pix_deg2 = pix_srad*(180/np.pi)**2
            self.stars /= pix_deg2
        return self.stars

    def _get_comp_map(self):
        if self.comp_map is None:
            comp_map = hp.read_map(self.config['completeness_map'])
            comp_map = rotate_mask(comp_map, self.rot)
            self.comp_map = hp.ud_grade(comp_map,
                                        nside_out=self.nside)
            self.comp_map[self.comp_map < 0.1] = 0.
        return self.comp_map

    def _get_binary_mask(self):
        if self.bmask is None:
            bmsk = hp.read_map(self.config['binary_mask'])
            bmsk = rotate_mask(bmsk, self.rot)
            self.bmask = hp.ud_grade(bmsk, nside_out=self.nside)
            self.bmask[self.bmask < 0.5] = 0
            self.bmask[self.bmask >= 0.5] = 1
        return self.bmask

    def _get_mask(self):
        self.bmask = self._get_binary_mask()
        self.comp_map = self._get_comp_map()
        return self.bmask * self.comp_map

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            cat_data = self.get_catalog()
            n = get_map_from_points(cat_data, self.nside,
                                    rot=self.rot)
            N_mean = self._get_mean_n(n)
            N_mean_srad = N_mean * self.npix / (4 * np.pi)
            mask = self.get_mask()
            N_ell = np.mean(mask) / N_mean_srad
            self.nl_coupled = N_ell * np.ones((1, 3*self.nside))
        return self.nl_coupled

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
