import numpy as np
import fitsio
from .mapper_DESI_LRG import MapperDESILRG
from astropy.table import Table, hstack


class MapperDESIBGS(MapperDESILRG):
    """
    Mapper class for the DESI BGS data set
    """
    map_name = "DESI_BGS"

    def _get_zbin_label(self, config):
        self.zmin = config["pzmin"]
        self.zmax = config["pzmax"]
        zbin_label = f'zmin{self.zmin}_zmax{self.zmax}'
        return zbin_label

    def _get_default_cuts(self):
        cuts = {
            "target_maskbits": [1, 12, 13],
            "min_nobs": 2,
            "max_sigmaz": 0.05,
            "remove_island": True,
        }
        return cuts

    def get_catalog(self):
        """
        Returns the mapper's data or random catalog.

        Returns:
            catalog (Table): Astropy Table with the catalog data.
        """

        if self.cat is None:
            print("Loading BGS catalog...", flush=True)
            cat_path = self.config["data_catalog"]
            cat = Table(fitsio.read(cat_path))
            print("Loading pz catalog...", flush=True)
            cat_path = self.config["data_catalog_pz"]
            cat = hstack([
                cat,
                Table(fitsio.read(cat_path))
            ])
            print(f"Loaded catalog with {len(cat)} objects")

            mask = self._get_quality_cuts(cat)
            cat = cat[mask]
            print(f"Number of BGS objects after quality cuts {len(cat)}")

            # Bin in photo-z
            zmin = self.zmin
            zmax = self.zmax
            mask_zbin = np.ones(len(cat), dtype=bool)
            if zmin is not None:
                mask_zbin &= cat["Z_PHOT_MEAN"] >= zmin
            if zmax is not None:
                mask_zbin &= cat["Z_PHOT_MEAN"] <= zmax
            cat = cat[mask_zbin]
            print(
                f"Number of BGS objects in z-bin [{zmin}, {zmax}] {len(cat)}"
            )

            self.cat = cat

        return self.cat

    def _get_quality_cuts(self, cat, randoms=False):
        """
        Return the quality cuts mask to apply to the catalog.
        randoms_clean = randoms[mask]
        """
        mask = np.ones(len(cat), dtype=bool)

        target_maskbits = self.cuts["target_maskbits"]
        for bit in target_maskbits:
            mask &= (cat["MASKBITS"] & 2**bit) == 0
        print(f"MASKBITS. Keeping {mask.sum()} objects")

        # 2+ exposures
        mask &= cat["NOBS_G"][:] >= self.cuts["min_nobs"]
        mask &= cat["NOBS_R"][:] >= self.cuts["min_nobs"]
        mask &= cat["NOBS_Z"][:] >= self.cuts["min_nobs"]
        print(f"Pixel exposures. Keeping {mask.sum()} objects")

        # Apply cuts from external maps
        for syst in self.config.get("external_maps", []):
            if syst.get('apply', True):
                mask &= self._get_map_threshold_mask(
                    syst['path'], syst['threshold'], cat,
                    field=syst.get('field', 0))
                print(f"{syst['name']}. Keeping {mask.sum()} objects")

        # Remove "islands" in the NGC
        # Extra cut in quality_cuts.py (used in MWhite+2021)
        if self.cuts["remove_island"]:
            mask &= ~(
                (cat["DEC"][:] < -10.5)
                & (cat["RA"][:] > 120)
                & (cat["RA"][:] < 260)
            )
            print(f"Island. Keeping {mask.sum()} objects")

        if not randoms:
            # Photo-z cut
            std_cut = self.cuts["max_sigmaz"] * (1 + cat["Z_PHOT_MEDIAN"])
            mask &= cat["Z_PHOT_STD"] < std_cut
            print(f"Z_STD cut. Keeping {mask.sum()} objects")

        return mask

    def _load_spec_catalog(self):
        """
        """
        cat = Table(fitsio.read(self.config["spec_catalog_xmatch"]))
        # Restriction from 2510.14135 (don't know how relevant it is)
        mask = cat["COADD_FIBERSTATUS"] == 0
        mask &= np.isin(cat["ZWARN"], [0, 4])
        mask &= np.isin(cat["SPECTYPE"], ["GALAXY", "QSO"])
        mask &= ~np.isnan(cat["Z"])
        mask &= cat["NPIXELS"] != 0

        return cat[mask]

    def _get_nz(self):
        spec_cat = self._load_spec_catalog()
        if self.cat is None:
            self.get_catalog()

        boolmask = np.isin(spec_cat["TARGETID"], self.cat["TARGETID"])
        zspec = spec_cat["Z"][boolmask]

        bins = np.linspace(0, 1, 101)
        nz, edges = np.histogram(zspec, bins=bins)

        zmin, zmax = edges[:-1], edges[1:]
        z_mid = zmin + (zmax - zmin) / 2

        return {"z_mid": z_mid, "nz": nz}

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
            fn = f'{self.map_name}_dndz.npz'
            self.dndz = self._rerun_read_cycle(fn, 'NPZ', self._get_nz)
        return self._get_shifted_nz(dz)

    def _get_weight_col_name(self):
        return "weight_pzbin"

    def compute_weights(self, randoms):
        """
        Compute the weights for the randoms.
        :param randoms: astropy Table with the randoms
        :return: weights per z-bin
        """
        return {"weight_pzbin": np.ones(len(randoms))}
