import pymaster as nmt


class MapperBase(object):
    def __init__(self, config):
        self._get_defaults(config)

    def _get_defaults(self, config):
        self.config = config
        self.mask_name = config.get('mask_name', None)
        self.nside = config['nside']
        self.nmt_field = None
        self.custom_auto = False

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
