from .mapper_base import MapperBase
import numpy as np
import healpy as hp
import pyccl as ccl


class MapperDummy(MapperBase):
    def __init__(self, config):
        """
        config - dict
          {'dtype': 'galaxy_density',
           'seed': None,
           'nside': ***,
           'fsky': 0.2,
           'cosmo': {
                'Omega_c': 0.2640,
                'Omega_b': 0.0493,
                'h': 0.6736,
                'n_s': 0.9649,
                'sigma8': 0.8111,
                'w0': -1,
                'wa': 0,
                'transfer_function': 'boltzmann_camb',
                'baryons_power_spectrum': 'nobaryons'
                }
            'zbin': 0,
            'width': 0.5,
            }
        """
        self._get_defaults(config)
        self.seed = self.config.get('seed', None)
        self.fsky = self.config.get('fsky', 0.2)
        cosmo = {
            # Planck 2018: Table 2 of 1807.06209
            # Omega_m: 0.3133
            'Omega_c': 0.2640,
            'Omega_b': 0.0493,
            'h': 0.6736,
            'n_s': 0.9649,
            'sigma8': 0.8111,
            'w0': -1,
            'wa': 0,
            'transfer_function': 'boltzmann_camb',
            'baryons_power_spectrum': 'nobaryons',
        }
        self.cosmo_pars = self.config.get('cosmo', cosmo)
        self.noise_level = self.config.get('noise_level', 0)
        self.cosmo = ccl.Cosmology(**self.cosmo_pars)
        ccl.sigma8(self.cosmo)
        self.dtype = self.config.get('dtype', 'galaxy_density')
        self._check_dtype()
        self.spin = self._get_spin_from_dtype(self.dtype)
        self.nmaps = 1
        if self.spin:
            self.nmaps = 2

        self.signal_map = None
        self.nl_coupled = None
        self.mask = None
        self.dndz = None
        self.cl = None

    def _check_dtype(self):
        dtypes = ['galaxy_density', 'galaxy_shear', 'cmb_convergence']
        if self.dtype not in dtypes:
            raise ValueError(f'Tracer type {self.dtype} not implemented.')

    def _get_spin_from_dtype(self, dtype):
        if dtype in ['galaxy_density', 'cmb_convergence']:
            spin = 0
        elif dtype == 'galaxy_shear':
            spin = 2
        return spin

    def get_nz(self, dz=0):
        if self.dndz is None:
            if self.dtype == 'galaxy_density':
                z, nz = np.loadtxt('xcell/tests/data/DESY1gc_dndz_bin0.txt',
                                   usecols=(1, 3), unpack=True)
            elif self.dtype == 'galaxy_shear':
                z, nz = np.loadtxt('xcell/tests/data/Nz_DIR_z0.1t0.3.asc',
                                   usecols=(0, 1), unpack=True)
            elif self.dtype == 'cmb_convergence':
                return None

            self.dndz = np.array([z, nz])

        z, nz = self.dndz
        z_dz = z + dz
        sel = z_dz >= 0

        return np.array([z_dz[sel], nz[sel]])

    def get_cl(self):
        if self.cl is None:
            ls = np.arange(3 * self.nside)
            dtype = self.get_dtype()
            if dtype == 'galaxy_density':
                z, nz = self.get_nz()
                b = np.ones_like(z)
                tracer = ccl.NumberCountsTracer(self.cosmo, has_rsd=False,
                                                dndz=(z, nz), bias=(z, b))
            elif dtype == 'galaxy_shear':
                z, nz = self.get_nz()
                tracer = ccl.WeakLensingTracer(self.cosmo, dndz=(z, nz))
            elif dtype == 'cmb_convergence':
                tracer = ccl.CMBLensingTracer(self.cosmo, z_source=1100)

            self.cl = ccl.angular_cl(self.cosmo, tracer, tracer, ls)

        return self.cl

    def get_signal_map(self):
        if self.signal_map is None:
            np.random.seed(self.seed)
            cl = self.get_cl()
            if self.spin == 0:
                self.signal_map = [hp.synfast(cl, self.nside, verbose=False)]
            elif self.spin == 2:
                _, mq, mu = hp.synfast([0*cl, cl, 0*cl, 0*cl],
                                       self.nside, new=True,
                                       verbose=False)
                self.signal_map = [mq, mu]
        return self.signal_map

    def get_mask(self, aps=1., dec0=0., ra0=0.):
        if self.mask is None:
            if self.fsky >= 1:
                self.mask = np.ones(hp.nside2npix(self.nside))
            else:
                # This generates a correctly-apodized mask
                v0 = np.array([np.sin(np.radians(90-dec0)) *
                               np.cos(np.radians(ra0)),
                               np.sin(np.radians(90-dec0)) *
                               np.sin(np.radians(ra0)),
                               np.cos(np.radians(90-dec0))])
                vv = np.array(hp.pix2vec(self.nside,
                                         np.arange(hp.nside2npix(self.nside))))
                cth = np.sum(v0[:, None]*vv, axis=0)
                th = np.arccos(cth)
                th0 = np.arccos(1-2*self.fsky)
                th_apo = np.radians(aps)
                id0 = np.where(th >= th0)[0]
                id1 = np.where(th <= th0-th_apo)[0]
                idb = np.where((th > th0-th_apo) & (th < th0))[0]
                x = np.sqrt((1 - np.cos(th[idb] - th0)) / (1 - np.cos(th_apo)))
                mask_apo = np.zeros(hp.nside2npix(self.nside))
                mask_apo[id0] = 0.
                mask_apo[id1] = 1.
                mask_apo[idb] = x-np.sin(2 * np.pi * x) / (2 * np.pi)
                self.mask = mask_apo
        return self.mask

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            # Coupled analytical noise bias
            self.nl_coupled = np.zeros((self.nmaps*self.nmaps,
                                        3*self.nside)) + self.noise_level
        return self.nl_coupled

    def get_dtype(self):
        return self.dtype

    def get_spin(self):
        return self.spin
