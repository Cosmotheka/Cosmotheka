import pymaster as nmt
import numpy as np
from .utils import get_beam, get_rerun_data, save_rerun_data


class MapperBase(object):
    def __init__(self, config):
        self._get_defaults(config)

    def _get_defaults(self, config):
        self.config = config
        self.mask_name = config.get('mask_name', None)
        self.beam_info = config.get('beam_info', None)
        self.mask_power = config.get('mask_power', 1)
        self.nside = config['nside']
        self.nmt_field = None
        self.beam = None
        self.custom_auto = False

    def get_signal_map(self):
        raise NotImplementedError("Do not use base class")

    def get_contaminants(self):
        return None

    def get_mask(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_coupled(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_covariance(self):
        raise NotImplementedError("Do not use base class")

    def _rerun_read_cycle(self, fname, ftype, func,
                          section=None, saved_by_func=False):
        d = get_rerun_data(self, fname, ftype,
                           section=section)
        if d is None:
            d = func()
            if not saved_by_func:
                save_rerun_data(self, fname, ftype, d)
        return d

    def get_ell(self):
        return np.arange(3 * self.nside)

    def get_beam(self):
        if self.beam is None:
            self.beam = get_beam(self.nside, self.beam_info)
        return self.beam

    def _get_nmt_field(self, signal=None, **kwargs):
        if signal is None:
            signal = self.get_signal_map(**kwargs)
        mask = self.get_mask(**kwargs)
        cont = self.get_contaminants(**kwargs)
        beam = self.get_beam()
        n_iter = kwargs.get('n_iter', 0)
        return nmt.NmtField(mask, signal, beam=beam,
                            templates=cont, n_iter=n_iter)

    def get_nmt_field(self, **kwargs):
        if self.nmt_field is None:
            self.nmt_field = self._get_nmt_field(signal=None, **kwargs)
        return self.nmt_field
