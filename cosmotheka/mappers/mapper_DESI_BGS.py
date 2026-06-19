import numpy as np
import healpy as hp
import fitsio
from .mapper_base import MapperBase
from astropy.table import Table, hstack
import os
from .utils import get_map_from_points


class MapperDESIBGS(MapperBase):
    """
    Mapper class for the DESI BGS data set
    """
    map_name = "DESI_BGS"
    dtype = "galaxy_density"
    spin = 0
    masked_on_input = True

    def __init__(self, config):
        self._get_defaults(config)

        self.cat = None
        self.data_maps = {"n": None, "w": None, "w2": None}
        self.alpha = None
        self.nl_coupled = None
        self.rot = self._get_rotator("C")

        # Randoms
        self._list_randoms = None
        self._download_missing_randoms = config.get(
            "download_missing_randoms", False
        )
        self._remove_downloaded_randoms_after_clean = config.get(
            "remove_downloaded_randoms_after_clean", True
        )
        # We use maps since the randoms are too large to fit in memory
        self.randoms_maps = {"n": None, "w": None, "w2": None}
        self._randoms_path = config.get("randoms_path", None)
        self._randoms_selection = config.get("randoms_selection", None)
        # To avoid loading the same randoms multiple times
        self._loaded_randoms = {}

        # Suffix to change the map name and rerun files
        suffix_parts = []

        # Quality cuts
        self._stardens_good_hp_idx = None
        self._stardens_nside = None

        cuts = self._get_default_cuts()

        self.cuts = {}
        keys_cuts = sorted(cuts.keys())
        for k in keys_cuts:
            v = cuts[k]
            self.cuts[k] = config.get(k, v)

            if self.cuts[k] != v:
                vnew = self.cuts[k]
                k = k.replace("_", "")
                suffix_parts.append(f"{k}{vnew}")
        self.suffix_weights = "_".join(suffix_parts)

        # Parts affecting other parts of the mapper
        # Mask threshold
        self.mask_threshold = config.get("mask_threshold", 0.2)
        if self.mask_threshold != 0.2:
            suffix_parts.append(f"maskthreshold{self.mask_threshold}")

        # zbin
        self.zmin = config.get("pzmin")
        self.zmax = config.get("pzmax")
        if self.zmin is not None:
            suffix_parts.append(f"zmin{self.zmin}")
        if self.zmax is not None:
            suffix_parts.append(f"zmax{self.zmax}")

        # Join the suffix parts
        suffix = "_".join(suffix_parts)

        # Modify the map name
        self.map_name += f"_{suffix}" if suffix else ""

        # Mask name
        # If not given, we use the same name as the map name since the mask is
        # basically given by the randoms
        self.mask_name = config.get("mask_name", self.map_name)

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

    def _get_default_cuts(self):
        cuts = {
            "target_maskbits": [1, 12, 13],
            "min_nobs": 2,
            # "max_ebv": 0.15,
            "max_sigmaz": 0.05,
            "max_stardens": 2500,
            "remove_island": True,
        }
        return cuts

    def _get_quality_cuts(self, cat, randoms=False):
        """
        """
        mask = np.ones(len(cat), dtype=bool)

        # # Veto mask
        # mask *= cat["lrg_mask"][:] == 0
        # print("Veto mask. Keeping ", mask.sum())

        if randoms:
            # MASKBITS cut. The veto mask for randoms seem to miss some
            # MASKBITS cuts. This is why I put it after and only for randoms.
            target_maskbits = self.cuts["target_maskbits"]
            for bit in target_maskbits:
                mask &= (cat["MASKBITS"] & 2**bit) == 0

            print(f"MASKBITS. Keeping {mask.sum()} objects")

        # 2+ exposures
        mask &= cat["NOBS_G"][:] >= self.cuts["min_nobs"]
        mask &= cat["NOBS_R"][:] >= self.cuts["min_nobs"]
        mask &= cat["NOBS_Z"][:] >= self.cuts["min_nobs"]
        print(f"Pixel exposures. Keeping {mask.sum()} objects")

        # # E(B-V) < 0.15
        # if self.cuts["max_ebv"] is not None:
        #     mask &= cat["EBV"][:] < self.cuts["max_ebv"]
        #     print(f"EBV. Keeping {mask.sum()} objects")

        # Apply cut on stellar density
        mask &= self._get_stardens_mask(cat)
        print(f"Stellar density. Keeping {mask.sum()} objects")

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

    def _get_stardens_mask(self, cat):
        """
        Returns a mask for the LRGs to keep based on the stellar density map.
        """
        if self._stardens_good_hp_idx is None:
            fname = self.config["stardens_path"]
            stardens = fitsio.read(fname)  # Stellar density map
            self._stardens_nside = hp.npix2nside(stardens.size)
            self._stardens_good_hp_idx = stardens["HPXPIXEL"][
                stardens["STARDENS"] < self.cuts["max_stardens"]
            ]

        lrg_hp_idx = hp.ang2pix(
            self._stardens_nside, cat["RA"], cat["DEC"], lonlat=True
        )
        mask = np.isin(lrg_hp_idx, self._stardens_good_hp_idx)

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

    def get_nz(self, dz=0):
        """
        Computes the redshift distribution of sources.  Then, it shifts the
        distribution by "dz" (default dz=0).

        Kwargs:
            dz=0

        Returns:
            [z, nz] (Array)
        """
        if self.dndz is None:

            spec_cat = self._load_spec_catalog()
            if self.cat is None:
                self.get_catalog()

            boolmask = np.isin(spec_cat["TARGETID"], self.cat["TARGETID"])
            zspec = spec_cat["Z"][boolmask]

            bins = np.linspace(0, 1, 101)
            nz, edges = np.histogram(zspec, bins=bins)

            zmin, zmax = edges[:-1], edges[1:]
            z_mid = zmin + (zmax - zmin) / 2

            self.dndz = {"z_mid": z_mid, "nz": nz}

        return self._get_shifted_nz(dz)

    def get_dtype(self):
        return self.dtype

    def get_spin(self):
        return self.spin

    def _get_list_randoms(self):
        """
        Returns a list of randoms to include
        """
        if self._list_randoms is not None:
            return self._list_randoms

        path = self._randoms_path

        list_randoms = []
        if self._randoms_selection is not None:
            if type(self._randoms_selection) is list:
                list_randoms = self._randoms_selection
            elif os.path.isfile(self._randoms_selection):
                # If the path is a file, it should contain a list of randoms
                with open(self._randoms_selection, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.endswith(".fits"):
                            line = line.split(".")[0]
                        list_randoms.append(line)
            else:
                raise ValueError(
                    f"Invalid randoms selection: {self._randoms_selection}. "
                    "It should be a file or a list."
                )
        else:
            if os.path.isdir(path):
                # If the path is a directory, we assume it contains the randoms
                files = os.listdir(path)
                for f in files:
                    if (
                        not (f.endswith(".fits") or f.endswith(".fits.gz"))
                        or not f.startswith("randoms-")
                        or f.startswith(".")
                    ):
                        continue
                    fname = f.split(".")[0]
                    fname = fname.replace("_clean_weights", "")
                    list_randoms.append(fname)
            elif os.path.isfile(path):
                # Split the path to get the base name and dir
                self._randoms_path = os.path.dirname(path)
                basename = os.path.basename(path)
                list_randoms.append(basename.split(".")[0])
            else:
                raise ValueError(
                    f"Invalid path for randoms: {path}. It should be a "
                    "directory or a file."
                )

        if not list_randoms:
            raise ValueError(
                "No randoms found. Please check the path or selection."
            )

        self._list_randoms = list_randoms

        return self._list_randoms

    def _download_randoms_file(self, base_name):
        """
        Downloads the randoms from the DESI data portal.
        """
        if not self._download_missing_randoms:
            raise ValueError(
                "If you want to download randoms, set "
                '"download_missing_random" to True.'
            )

        print(
            f"Downloading randoms {base_name} from {self._randoms_path}...",
            flush=True,
        )
        url = (
            "https://data.desi.lbl.gov/public/ets/target/catalogs/dr9/0.49.0/"
            f"randoms/resolve/{base_name}.fits"
        )
        rand_file = os.path.join(self._randoms_path, f"{base_name}.fits")
        # Download the randoms file
        try:
            os.system(f"wget {url} -O {rand_file}")
            print(f"[{base_name}] Downloaded {rand_file}.", flush=True)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            raise

        return rand_file

    def _load_full_randoms(self, base_name):
        random_path = self._randoms_path
        rand_file = os.path.join(random_path, f"{base_name}.fits")
        rand_mask_name = f"{base_name}-lrgmask_v1.1.fits.gz"
        lrgmask_file = os.path.join(
            self.config["randoms_lrgmask_path"], rand_mask_name
        )
        downloaded = False

        # Check if the randoms file exists
        if (
            not os.path.exists(rand_file)
            and not self._download_missing_randoms
        ):
            raise FileNotFoundError(
                f"Randoms file {rand_file} does not exist."
            )
        elif not os.path.exists(rand_file) and self._download_missing_randoms:
            print(
                f"[{base_name}] Randoms file does not exist, downloading...",
                flush=True,
            )
            self._download_randoms_file(base_name)
            downloaded = True

        # Load the randoms
        print(
            f"[{base_name}] Loading randoms from {rand_file}...",
            flush=True,
        )
        columns = [
            "RA",
            "DEC",
            "NOBS_G",
            "NOBS_R",
            "NOBS_Z",
            "MASKBITS",
            "EBV",
            # The following ones are used to compute the weights
            "GALDEPTH_G",
            "GALDEPTH_R",
            "GALDEPTH_Z",
            "PSFSIZE_G",
            "PSFSIZE_R",
            "PSFSIZE_Z",
            "PHOTSYS",
        ]

        randoms = Table(fitsio.read(rand_file, columns=columns))

        print(
            f"[{base_name}] Loaded randoms with {len(randoms)} entries.",
            flush=True,
        )

        print(
            f"[{base_name}] Loading lrgmask from {lrgmask_file}...",
            flush=True,
        )

        lrgmask = Table(fitsio.read(lrgmask_file))
        randoms = hstack([randoms, lrgmask])

        return randoms, downloaded

    def __get_clean_randoms_with_weights(self, base_name):
        print("Loading randoms for", base_name, flush=True)
        randoms, downloaded = self._load_full_randoms(base_name)

        # Apply cuts
        print(f"[{base_name}] Applying quality cuts...", flush=True)
        mask_good = self._get_quality_cuts(randoms, randoms=True)
        randoms = randoms[mask_good]
        print(f"[{base_name}] Final: {len(randoms)}", flush=True)

        # Compute weights
        print(f"[{base_name}] Computing weights...", flush=True)
        weights = self.compute_weights(randoms)

        # Clean the randoms file
        print(
            "Removing unnecessary columns...",
            flush=True,
        )
        cols_to_keep = ["RA", "DEC"]
        print(
            f"Keeping columns {cols_to_keep} and weights.",
            flush=True,
        )
        randoms = randoms[cols_to_keep]
        print("Catalog reduced, Adding weights...", flush=True)
        randoms = hstack([randoms, Table(weights)])
        # randoms = hstack([randoms[cols_to_keep], Table(weights)])
        print("Catalog merged")
        # Remove the downloaded randoms file if requested
        if downloaded and self._remove_downloaded_randoms_after_clean:
            fn = os.path.join(self._randoms_path, f"{base_name}.fits")
            if os.path.exists(fn):
                print(
                    f"[{base_name} Removing the downloaded randoms {fn}",
                    flush=True,
                )
                os.remove(fn)
        print("returned randoms", flush=True)
        return randoms

    def get_clean_randoms_with_weights(self, base_name):
        if base_name in self._loaded_randoms:
            return self._loaded_randoms[base_name]

        fn = "".join(
            [f"{base_name}_clean_weights", f"{self.suffix_weights}.fits"]
        )
        print(f"{fn}", flush=True)
        randoms = Table(
            self._rerun_read_cycle(
                fn,
                "FITSTable",
                self.__get_clean_randoms_with_weights,
                base_name=base_name,
            )
        )

        # Only keep one randoms file in memory at a time
        self._loaded_randoms = {base_name: randoms}

        return randoms

    def get_randoms_maps(self):
        if self.randoms_maps["n"] is not None:
            return self.randoms_maps

        list_randoms = self._get_list_randoms()
        npix = hp.nside2npix(self.nside)

        randoms_maps = np.zeros((3, npix))

        # Hack to remove the density definition from the randoms map name
        map_name = self.map_name.replace("_densdefZhou2023", "")

        # TODO: consider if I want to save the sum of all maps. Problem, it
        # makes the code a bit more complex and it's difficult to know which
        # randoms when into the map.
        for base_name in list_randoms:
            # weight_col = f"weight_pzbin{self.zbin + 1}"
            weight_col = "weight_pzbin"

            def f():
                randoms = self.get_clean_randoms_with_weights(base_name)
                w = np.array(randoms[weight_col])
                map_ngal = np.zeros((3, npix))
                for power in [0, 1, 2]:
                    print(f"Computing map for {base_name} with weights to the power of {power}...", flush=True)
                    map_ngal[power] = get_map_from_points(
                        randoms,
                        self.nside,
                        rot=self.rot,
                        w=w**power if power > 0 else None,
                    )
                    print(f"Finished map for {base_name} with weights to the power of {power}.", flush=True)
                return map_ngal

            fname = "_".join(
                [
                    f"map_{map_name}_{base_name}",
                    "n-w-w2",
                    f"coord{self.coords}",
                    f"ns{self.nside}.fits.gz",
                ]
            )
            map_nrand = self._rerun_read_cycle(fname, "FITSMap", f)

            randoms_maps += map_nrand

        for i, key in enumerate(["n", "w", "w2"]):
            self.randoms_maps[key] = randoms_maps[i]
        return self.randoms_maps

    def compute_weights(self, randoms):
        """
        Compute the weights for the randoms.
        :param randoms: astropy Table with the randoms
        :return: weights per z-bin
        """
        # Placeholder, need to check where data products are
        return {"weight_pzbin": np.ones(len(randoms))}

    def _get_alpha(self):
        """
        Computes alpha parameter that makes <w_data> = alpha * <w_random>.
        """
        if self.alpha is None:
            w_data = self.get_data_maps()["w"]
            w_random = self.get_randoms_maps()["w"]
            self.alpha = np.sum(w_data) / np.sum(w_random)
        return self.alpha

    def get_data_maps(self):
        if self.data_maps["n"] is None:
            # They don't apply weights to the data "to avoid shot noise"
            cat_data = self.get_catalog()
            nmap_data = get_map_from_points(cat_data, self.nside, rot=self.rot)
            self.data_maps["n"] = nmap_data
            self.data_maps["w"] = nmap_data  # \sum w = n with w=1
            self.data_maps["w2"] = nmap_data  # \sum w2 = n with w=1

        return self.data_maps

    def _get_signal_map(self):
        # Instead of providing the overdensity map, we provide the
        # difference map; i.e. delta * mask, for better NmtField stability.
        nmap_data = self.get_data_maps()["n"]
        mask = self.get_mask()  # Recall mask = alpha * w_random
        signal_map = nmap_data - mask

        return signal_map

    def _get_mask(self):
        # Calculates the mask based on the randoms (m = alpha * w_random).
        alpha = self._get_alpha()
        w_map = self.get_randoms_maps()["w"]

        mask = alpha * w_map

        # Apply a threshold
        goodpix = mask > 0
        avg = np.mean(mask[goodpix])
        goodpix = mask > self.mask_threshold * avg
        print(
            "Masking pixels with less than",
            f"{self.mask_threshold:.2f} average weight.",
        )

        mask[~goodpix] = 0.0
        return mask

    def _get_nl_coupled(self):
        """
        Computes the noise power spectrum for the mapper.
        """
        print("Calculing N_l from weights")
        alpha = self._get_alpha()
        pixel_A = hp.nside2pixarea(self.nside)

        mask = self.get_mask()
        w2_data = self.get_data_maps()["w2"]
        w2_random = self.get_randoms_maps()["w2"]

        goodpix = mask > 0
        N_ell = w2_data[goodpix].sum() + alpha**2 * w2_random[goodpix].sum()
        N_ell *= pixel_A**2 / (4 * np.pi)
        nl_coupled = N_ell * np.ones((1, 3 * self.nside))

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
