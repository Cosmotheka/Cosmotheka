from .utils import get_map_from_points
from .mapper_base import MapperBase
from astropy.table import Table
import numpy as np
import healpy as hp


class MapperQuaia(MapperBase):
    """
    """
    map_name = 'Quaia'

    def __init__(self, config):
        self._get_defaults(config)
        self.rot = self._get_rotator('C')
        self.num_z_bins = config.get('num_z_bins', 500)
        self.z_edges = config.get('z_edges', [0, 4.5])
        self.zbin_name = 'z%.3lf_%.3lf' % (self.z_edges[0], self.z_edges[1])
        self.z_name = config.get("z_name", "redshift_quaia")

        self.cat_data = None
        self.npix = hp.nside2npix(self.nside)

        self.nl_coupled = None
        self.mskflag = None
        self.ipix = None
        self.dndz = None

    def _get_catalog(self):
        # Returns the mapper's catalog \
        # after binning it in redshift.

        cat = Table.read(self.config['data_catalog'], format='fits')
        cat.keep_columns(['ra', 'dec', self.z_name, self.z_name+'_err'])
        z_d = cat[self.z_name]
        mask_bin = (z_d < self.z_edges[1]) & (z_d >= self.z_edges[0])
        cat = cat[mask_bin]
        return cat.as_array()

    def get_catalog(self):
        """
        Checks if the mapper has already \
        loaded the chosen bin of the catalog. \
        If so, it loads it from the save file. \
        Otherwise, it cuts the original file

        Returns:
            catalog (Array)
        """
        if self.cat_data is None:
            fn = f'{self.map_name}_{self.zbin_name}_cat.fits'
            self.cat_data = self._rerun_read_cycle(fn, 'FITSTable',
                                                   self._get_catalog)
        return self.cat_data

    def _get_mask(self):
        fname_sel = f'selection_{self.coords}'
        msk = hp.ud_grade(hp.read_map(self.config[fname_sel]),
                          nside_out=self.nside)
        msk_thr = self.config.get('mask_threshold', 0.5)
        msk = msk / np.amax(msk)
        msk[msk < msk_thr] = 0
        fname_extra = self.config.get('mask_extra_{self.coords}')
        if fname_extra:
            m = hp.ud_grade(hp.read_map(fname_extra),
                            nside_out=self.nside)
            m = (m > 0).astype(float)
            msk *= m
        return msk

    def _get_ipix(self):
        if self.ipix is None:
            cat = self.get_catalog()
            self.ipix = hp.ang2pix(self.nside, cat['ra'],
                                   cat['dec'], lonlat=True)
        return self.ipix

    def _get_angmask(self):
        if self.mskflag is None:
            self.get_catalog()
            mask = self.get_mask()
            ipix = self._get_ipix()
            self.mskflag = mask[ipix] > 0
        return self.mskflag

    def _get_nz(self):
        # Builds the redshift distributions of \
        # the sources of the mapper's catalog.

        cat = self.get_catalog()
        mskflag = self._get_angmask()
        c = cat[mskflag]
        zm = c[self.z_name]

        if self.config.get('nz_spec', False):
            nz, b = np.histogram(zm, range=[0., 4.5],
                                 bins=self.num_z_bins)
            zs = 0.5 * (b[:-1] + b[1:])
        else:
            zs = np.linspace(0., 4.5, self.num_z_bins)
            sz = c[self.z_name+'_err']
            nz = np.array([np.sum(np.exp(-0.5*((z-zm)/sz)**2)/sz)
                           for z in zs])

        return {'z_mid': zs, 'nz': nz}

    def get_nz(self, dz=0):
        """
        Checks if mapper has precomputed the redshift \
        distribution. If not, it uses "_get_nz()" to obtain it. \
        Then, it shifts the distribution by "dz" (default dz=0).

        Kwargs:
            dz=0

        Returns:
            [z, nz] (Array)
        """
        if self.dndz is None:
            fn = f'{self.map_name}_{self.zbin_name}_dndz.npz'
            self.dndz = self._rerun_read_cycle(fn, 'NPZ', self._get_nz)
        return self._get_shifted_nz(dz)

    def _get_signal_map(self, apply_galactic_correction=True):
        cat = self.get_catalog()
        mask = self.get_mask()
        bmask = mask > 0
        nmap = get_map_from_points(cat, self.nside, rot=self.rot,
                                   ra_name='ra', dec_name='dec')
        nmean = np.sum(nmap*bmask)/np.sum(mask)
        delta = np.zeros(self.npix)
        delta[bmask] = nmap[bmask]/(nmean*mask[bmask])-1
        return delta

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            cat = self.get_catalog()
            mask = self.get_mask()
            bmask = mask > 0
            nmap = get_map_from_points(cat, self.nside, rot=self.rot,
                                       ra_name='ra', dec_name='dec')
            nmean = np.sum(nmap*bmask)/np.sum(mask)
            nmean_srad = nmean * self.npix / (4 * np.pi)
            nl = np.mean(mask) / nmean_srad
            self.nl_coupled = nl * np.ones((1, 3*self.nside))
        return self.nl_coupled

    def get_dtype(self):
        return 'galaxy_density'

    def get_spin(self):
        return 0
