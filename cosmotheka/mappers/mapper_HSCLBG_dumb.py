"""Implements a Dummy mapper that can be used to test the MapperBase."""
from .mapper_base import MapperBase
import numpy as np
import healpy as hp


class MapperHSCLBGdumb(MapperBase):
    """Simple mapper for the galaxy overdensity of HSC LBGs from
    number-count map effective pixel fraction map."""
    map_name = 'HSCLBG_dumb'
    dtype = "galaxy_density"
    spin = 0
    masked_on_input = True

    def __init__(self, config):
        """
        - config: dictionary with the following keys:
            - map_n: path to the number-count map (fits)
            - map_w: path to the effective pixel fraction map (fits)
            - mask_threshold: threshold for the effective pixel
                              fraction to define the mask (float)
            - dndz: path to the dndz file (txt)
            - filter: filter name (u, g, r, i, z, y)
            - mask_name: mask name
        """
        self._get_defaults(config)
        self.fname_map_n = self.config.get('map_n', None)
        self.fname_map_w = self.config.get('map_w', None)
        self.fname_dndz = self.config.get('dndz', None)
        self.mask_threshold = self.config.get('mask_threshold', 0.5)
        self.filter = self.config.get('filter', 'g')
        self.map_n = None
        self.nl_coupled = None
        self.rot = self._get_rotator("C")
        self.map_name += f"_{self.filter}drop"

    def get_nz(self, dz=0):
        if self.dndz is None:
            d = np.load(self.fname_dndz)
            self.dndz = {'z_mid': d['z_centers'], 'nz': d[f'nz_{self.filter}']}
        return self._get_shifted_nz(dz)

    def _get_maps(self):
        if self.map_n is None or self.map_w is None:
            self.map_n = hp.ud_grade(hp.read_map(self.fname_map_n),
                                     nside_out=self.nside)
            self.map_w = hp.ud_grade(hp.read_map(self.fname_map_w),
                                     nside_out=self.nside)
            self.map_w[self.map_w < self.mask_threshold] = 0
            self.map_n[self.map_w == 0] = 0
        return self.map_n, self.map_w

    def _get_signal_map(self):
        nmap, wmap = self._get_maps()
        nmean = np.sum(nmap) / np.sum(wmap)
        delta = nmap / nmean - wmap
        return np.array([delta])

    def _get_mask(self):
        _, wmap = self._get_maps()
        return wmap

    def _get_nl_coupled(self):
        nmap, wmap = self._get_maps()
        nmean = np.sum(nmap) / np.sum(wmap)
        ndens = nmean * self.npix / (4 * np.pi)
        nl = np.mean(wmap) / ndens
        nl_coupled = nl * np.ones((1, 3*self.nside))
        return {"nls": nl_coupled}

    def get_nl_coupled(self):
        if self.nl_coupled is None:
            fn = "_".join(
                [
                    f"{self.map_name}_Nell",
                    f"coord{self.coords}",
                    f"ns{self.nside}.npz",
                ]
            )
            d = self._rerun_read_cycle(fn, "NPZ", self._get_nl_coupled)
            self.nl_coupled = d["nls"]
        return self.nl_coupled

    def get_dtype(self):
        return self.dtype

    def get_spin(self):
        return self.spin
