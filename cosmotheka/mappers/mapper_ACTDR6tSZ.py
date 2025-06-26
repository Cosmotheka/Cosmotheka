from .mapper_ACT_compsept import MapperACTCompSept
from pixell import reproject, enmap
from .utils import rotate_mask
import numpy as np
import pymaster as nmt


class MapperACTDR6tSZ(MapperACTCompSept):
    """
    ACT DR6 Compton-y mapper class.
    Maps are convolved with a 1.6arcmin Gaussian beam.
    """
    def __init__(self, config):
        super().__init__(config)
        self.deproj_type = config.get("deproj_type", None)
        self.ps_mask_file = config.get("ps_mask_file", None)
        self.inpaint_mask_file = config.get("inpaint_mask_file", None)

        self.footprint_aposize = config.get("footprint_aposize", 60.)
        self.ps_aposize = config.get("ps_aposize", 5.)

        if self.deproj_type is not None:
            self.file_map = self.file_map.replace(
              ".fits", f"_deproj_{self.deproj_type}.fits"
            )
            self.map_name = f"{self.map_name}_deproj_{self.deproj_type}"

        ps_in = False
        if self.ps_mask_file is not None:
            self.mask_name = f"{self.mask_name}_ps"
            ps_in = True
        if self.inpaint_mask_file is not None:
            self.mask_name = f"{self.mask_name}_inpaint"
            ps_in = True

        self.mask_name = (
            f"{self.mask_name}_apo{self.footprint_aposize:.1f}arcmin"
        )
        if ps_in:
            self.mask_name = (
                f"{self.mask_name}_psapod{self.ps_aposize:.1f}arcmin"
            )

    def _get_src_binary_mask(self, file):
        msk = enmap.read_map(file)
        msk = reproject.map2healpix(
            msk,
            nside=self.nside,
            lmax=self.lmax,
            method="spline",
            order=1
        )
        msk = rotate_mask(msk, self.rot)
        msk = 1 - msk
        msk = np.where(msk < 0.99, 0, 1)
        return msk

    def _get_mask(self):
        self.pixell_mask = self._get_pixell_mask()
        msk = reproject.map2healpix(
            self.pixell_mask,
            nside=self.nside,
            lmax=self.lmax,
            method="spline",
            order=1
        )
        msk = rotate_mask(msk, self.rot)
        # Binarize mask
        msk = np.where(msk < 0.99, 0, 1)

        src_mask = np.ones_like(msk)
        if self.ps_mask_file is not None:
            ps_mask = self._get_src_binary_mask(self.ps_mask_file)
            src_mask *= ps_mask

        if self.inpaint_mask_file is not None:
            inpaint_mask = self._get_src_binary_mask(self.inpaint_mask_file)
            src_mask *= inpaint_mask

        if np.any(src_mask < 1):
            src_mask = nmt.mask_apodization(
                src_mask,
                aposize=self.ps_aposize/60,  # Convert to deg
                apotype="C1"
            )
        msk = nmt.mask_apodization(
            msk,
            aposize=self.footprint_aposize/60,  # Convert to deg
            apotype="C1"
        )
        msk *= src_mask
        return msk

    def get_dtype(self):
        return 'cmb_tSZ'

    def get_spin(self):
        return 0
