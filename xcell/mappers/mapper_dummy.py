from .mapper_base import MapperBase
import numpy as np
import healpy as hp


class MapperDummy(MapperBase):
    def __init__(self, config):
        """
        config - dict
          {'data_catalogs': ***,
           'random_catalogs': ***,
           'z_edges': ***,
           'nside': ***,
           'nside_mask': ***,
           'mask_name': ***}
        """
        self._get_defaults(config)
        self.spin = self.config.get('spin', 0)
        self.l0 = self.config.get('l0', 20.)
        self.alpha = self.config.get('alpha', 1.)
        self.npix = hp.nside2npix(self.nside)

        self.signal_map = None
        self.nl_coupled = None
        self.mask = None

    def get_cl(self, ls):
        return 1./(ls+self.l0)**self.alpha

    def get_signal_map(self, seed=1000):
        if self.signal_map is None:
            np.random.seed(seed)
            ls = np.arange(3*self.nside)
            cl = self.get_cl(ls)
            if self.spin == 0:
                self.signal_map = [hp.synfast(cl, self.nside)]
            elif self.spin == 2:
                _, mq, mu = hp.synfast([0*cl, cl, 0*cl, 0*cl],
                                       self.nside)
                self.signal_map = [mq, mu]
        return self.signal_map

    def get_mask(self, aps=1., fsk=0.2, dec0=0., ra0=0.):
        if self.mask is None:
            if fsk >= 1:
                self.mask = np.ones(self.npix)
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
                th0 = np.arccos(1-2*fsk)
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
            self.nl_coupled = np.zeros((1, 3*self.nside))
        return self.nl_coupled

    def get_dtype(self):
        return 'generic'

    def get_spin(self):
        return self.spin
