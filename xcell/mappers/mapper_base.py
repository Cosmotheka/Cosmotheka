class MapperBase(object):
    def __init__(self, config):
        pass

    def get_signal_map(self):
        raise NotImplementedError("Do not use base class")

    def get_mask(self):
        raise NotImplementedError("Do not use base class")

    def get_nmt_field(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_coupled(self):
        raise NotImplementedError("Do not use base class")

    def get_nl_covariance(self):
        raise NotImplementedError("Do not use base class")
