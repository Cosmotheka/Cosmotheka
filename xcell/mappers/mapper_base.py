import pymaster as nmt
import numpy as np


class MapperBase(object):
    def __init__(self, config):
        self._get_defaults(config)

    def _get_defaults(self, config):
        self.config = config
        self.mask_name = config.get('mask_name', None)
        self.nside = config['nside']
        self.nmt_field = None
        self.custom_auto = False
        # Option introduced to modify the Mode Coupling Matrix
        # In case the map has an implicit mask applied
        # See ACTk case for an example
        self.mask_power = config.get('mask_power', 1)

    def get_signal_map(self):
        raise NotImplementedError("Do not use base class")

    def get_contaminants(self):
        return None

    def get_beam(self):
        return None

    def get_mask(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_coupled(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_covariance(self):
        raise NotImplementedError("Do not use base class")

    def get_ell(self):
        return np.arange(3 * self.nside)

    def _get_nmt_field(self, signal=None, **kwargs):
        if signal is None:
            signal = self.get_signal_map(**kwargs)
        mask = self.get_mask(**kwargs)
        cont = self.get_contaminants(**kwargs)
        beam = self.get_beam(**kwargs)
        n_iter = kwargs.get('n_iter', 0)
        return nmt.NmtField(mask, signal, beam=beam,
                            templates=cont, n_iter=n_iter)

    def get_nmt_field(self, **kwargs):
        if self.nmt_field is None:
            self.nmt_field = self._get_nmt_field(signal=None, **kwargs)
        return self.nmt_field
