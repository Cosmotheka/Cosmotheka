from .mapper_ACT_compsept import MapperACTCompSept


class MapperACTDR6tSZ(MapperACTCompSept):
    """
    ACT DR6 Compton-y mapper class.
    Maps are convolved with a 1.6arcmin Gaussian beam.
    """
    def __init__(self, config):
        self._get_ACT_defaults(config)
        self.deproj_type = config.get("deproj_type", None)
        if self.deproj_type is not None:
            self.file_map = self.file_map.replace(
              ".fits", f"_deproj_{self.deproj_type}.fits"
            )
            self.map_name = f"{self.map_name}_deproj_{self.deproj_type}"

    def get_dtype(self):
        return 'cmb_tSZ'

    def get_spin(self):
        return 0
