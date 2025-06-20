import os
import numpy as np
import healpy as hp
import pymaster as nmt
import fitsio
from .utils import get_map_from_points, get_rerun_data, save_rerun_data
from .mapper_base import MapperBase
from astropy.table import Table, hstack
import yaml
import warnings


class MapperDESILRG(MapperBase):
    """
    Mapper class for the DESI LRGs data set from (Zhou et al. 2023, 2309.06443).

    Config:
        - zbin: `0` / `1` / `2` / `3`  [Note that the official files use the range [1, 4]]
        - data_catalog: \
          `'./DESI/DESI_LRGs_Zhou2023/lrg_xcorr_2023_v1/catalogs/dr9_lrg_pzbins.fits'`
        - file_dndz: \
          `'./DESI/DESI_LRGs_Zhou2023/lrg_xcorr_2023_v1/catalogs/dndz_lrg_pzbins.fits'`
        - randoms_path: \
        - randoms_selection: 
        - randoms_lrgmask_path:
        - sample: `'main'` / `'extended'` (default `'main'`)
        - nside: `4096`
        - imaging_weights_coeffs: `catalogs/imaging_weights/main_lrg_linear_coeffs_pz.yaml`
        - overdensity_definition: 'cosmotheka' / 'Zhou2023' (default 'cosmotheka')

    """

    map_name = "DESI_LRG"
    dtype = "galaxy_density"
    spin = 0

    def __init__(self, config):
        self._get_defaults(config)
        self.cat = None
        self.data_maps = {"n": None, "w": None, "w2": None}
        self.alpha = None
        self.nl_coupled = None
        self.rot = self._get_rotator("C")
        self.zbin = config["zbin"]
        # Randoms
        self._list_randoms = None
        self._download_missing_random = config.get(
            "download_missing_random", True
        )
        self._remove_random_after_clean = config.get(
            "remove_random_after_clean", True
        )
        # Since we don't have enough RAM to load all randoms, we do it at the
        # level of maps
        self.random_maps = {"n": None, "w": None, "w2": None}
        self._randoms_path = config.get("randoms_path", None)
        self._randoms_selection = config.get("randoms_selection", None)
        # To avoid loading the same randoms multiple times
        self._loaded_randoms = {}

        # Overdensity definition
        self.overdensity_definition = config.get(
            "overdensity_definition", "cosmotheka"
        )
        if self.overdensity_definition == "Zhou2023":
            self.masked_on_input = False
            self.map_name += "_defZhou2023"
        elif self.overdensity_definition == "cosmotheka":
            self.masked_on_input = True
        else:
            raise ValueError(
                f"Invalid overdensity definition: {self.overdensity_definition}. "
                "It should be 'cosmotheka' or 'Zhou2023'."
            )

        # Cuts
        self._stardens_good_hp_idx = None
        self._stardens_nside = None
        cuts = self._get_default_cuts()

        suffix_parts = []
        self.cuts = {}
        for k, v in cuts.items():
            self.cuts[k] = config.get(k, v)

            if self.cuts[k] != v:
                k = k.replace("_", "")
                suffix_parts.append(f"{k}{v}")
        self.suffix = "_".join(suffix_parts)
        self.map_name += f"_{self.suffix}" if self.suffix else ""

    def _get_default_cuts(self):
        cuts = {
            "target_min_nobs": 1,
            "target_maskbits": [1, 12, 13],
            "min_nobs": 2,
            "max_ebv": 0.15,
            "max_stardens": 2500,
            "stardens_nside": 64,
        }
        return cuts

    def _get_stardens_mask(self, cat):
        """
        Returns a mask for the LRGs to keep based on the stellar density map.
        """
        if self._stardens_good_hp_idx is None:
            fname = self.config["stardens_path"]
            stardens = fitsio.read(fname)  # Stellar density map
            self._stardens_nside = hp.npix2nside(stardens.size)
            self._stardens_good_hp_idx = stardens["HPXPIXEL"][
                stardens["STARDENS"] < 2500
            ]

        lrg_hp_idx = hp.ang2pix(
            self._stardens_nside, cat["RA"], cat["DEC"], lonlat=True
        )
        mask = np.in1d(lrg_hp_idx, self._stardens_good_hp_idx)

        return mask

    def _get_quality_cuts(self, cat, randoms=False):
        """
        Return the quality cuts mask to apply to the catalog.
        randoms_clean = randoms[mask]
        """
        # As described in Section 3.3 of Zhou+2023
        # More explicit in quality_cuts.py
        mask = np.ones(len(cat), dtype=bool)

        # Veto mask
        mask *= cat["lrg_mask"][:] == 0
        print("Veto mask. Keeping ", mask.sum())

        if randoms:
            # MASKBITS cut. The veto mask for randoms seem to miss some MASKBITS
            # that's why I put it after, to show it.
            target_maskbits = [1, 12, 13]  # TODO: read it from config
            for bit in target_maskbits:
                mask &= (cat["MASKBITS"] & 2**bit) == 0

            print("MASBITS. Keeping ", mask.sum())

        # 2+ exposures
        key = "PIXEL_NOBS" if not randoms else "NOBS"
        mask *= cat[f"{key}_G"][:] >= 2
        mask *= cat[f"{key}_R"][:] >= 2
        mask *= cat[f"{key}_Z"][:] >= 2
        print("Pixel exposures. Keeping ", mask.sum())

        # E(B-V) < 0.15
        mask *= cat["EBV"][:] < 0.15
        print("EBV. Keeping ", mask.sum())

        # Apply cut on stellar density
        mask *= self._get_stardens_mask(cat)
        print("Stellar density. Keeping ", mask.sum())

        # Remove "islands" in the NGC
        # Extra cut in quality_cuts.py (used in MWhite+2021)
        mask *= ~(
            (cat["DEC"][:] < -10.5)
            & (cat["RA"][:] > 120)
            & (cat["RA"][:] < 260)
        )
        print("Island. Keeping ", mask.sum())

        return mask

    def get_catalog(self):
        """
        Returns the mapper's data or random catalog.

        Returns:
            catalog (Table): Astropy Table with the catalog data.
        """

        if self.cat is None:
            cat_path = self.config["data_catalog"]
            cat = Table(fitsio.read(cat_path))
            weights_path = self.config["weights_catalog"]
            weights = Table(fitsio.read(weights_path))

            cat = hstack([cat, weights])
            print("Loaded catalog with", len(cat), "LRGs")

            mask = self._get_quality_cuts(cat)
            cat = cat[mask]

            print(
                "Number of LRGs after cuts ", len(cat)
            )  # vs 9996023 that outputs quality_cuts.py
            self.cat = cat

        return self.cat

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
            fname = self.config["file_dndz"]
            dndz = Table.read(
                fname, format="ascii", header_start=0, data_start=1
            )
            z_mid = dndz["zmin"] + (dndz["zmax"] - dndz["zmin"]) / 2
            nz = dndz[f"bin_{self.zbin + 1}_combined"]
            self.dndz = {"z_mid": z_mid, "nz": nz}
        return self._get_shifted_nz(dz)

    def _get_alpha(self):
        """
        Computes alpha parameter that makes <w_data> = alpha * <w_random>.
        """
        if self.alpha is None:
            w_data = self.get_data_maps()["w"]
            w_random = self.get_randoms_maps()["w"]
            if self.overdensity_definition == "Zhou2023":
                mask = self._get_mask_Zhou2023()
                msk = mask.astype(bool)
                self.alpha = np.mean(w_data[msk] / w_random[msk])
            else:
                self.alpha = np.sum(w_data) / np.sum(w_random)
        return self.alpha

    def get_data_maps(self):
        if self.data_maps["n"] is None:
            # They don't apply weights to the data "to avoid shot noise"
            cat_data = self.get_catalog()
            nmap_data = get_map_from_points(cat_data, self.nside, rot=self.rot)
            self.data_maps["n"] = nmap_data
            self.data_maps["w"] = nmap_data.copy()  # \sum w = n with w=1
            self.data_maps["w2"] = nmap_data.copy()  # \sum w2 = n with w=1

        return self.data_maps

    def _get_mask_Zhou2023(self):
        # Copied from https://github.com/NoahSailer/MaPar/blob/main/maps/make_lrg_maps.py
        rmap = self.get_randoms_maps()["w"].copy()

        mask = np.zeros_like(rmap)
        msk = np.nonzero(rmap > 0)[0]
        avg = np.mean(rmap[msk])
        msk = np.nonzero(rmap > 0.20 * avg)[0]
        mask[msk] = 1.0

        return mask

    def _get_signal_map_Zhou2023(self):
        # Based on https://github.com/NoahSailer/MaPar/blob/main/maps/make_lrg_maps.py

        dmap = self.get_data_maps()["n"].copy()
        rmap = self.get_randoms_maps()["w"].copy()

        mask = self._get_mask_Zhou2023()
        msk = mask.astype(bool)

        alpha = self._get_alpha()
        omap = np.zeros_like(rmap)  # Use rmap for the right dtype
        omap[msk] = dmap[msk] / (alpha * rmap[msk]) - 1

        return omap

    def _get_signal_map(self):
        if self.overdensity_definition == "Zhou2023":
            signal_map = self._get_signal_map_Zhou2023()
        else:
            # Instead of providing the overdensity map, we provide the
            # difference map; i.e. delta * mask, for better NmtField stability.
            nmap_data = self.get_data_maps()["n"]
            mask = self.get_mask()  # Recall mask = alpha * w_random
            signal_map = nmap_data - mask

        return signal_map

    def _get_mask(self):
        if self.overdensity_definition == "Zhou2023":
            mask = self._get_mask_Zhou2023()
        else:
            # Calculates the mask based on the randoms (m = alpha * w_random).
            alpha = self._get_alpha()
            w_map = self.get_randoms_maps()["w"]

            mask = alpha * w_map
        return mask

    def _get_nl_coupled(self):
        """
        Computes the noise power spectrum for the mapper.
        """
        if self.overdensity_definition == "Zhou2023":
            # Copied from https://github.com/NoahSailer/MaPar/blob/main/maps/make_lrg_maps.py
            dmap = self.get_data_maps()["n"].copy()
            rmap = self.get_randoms_maps()["w"].copy()
            wmap = rmap / (self.get_randoms_maps()["n"] + 1e-30)

            shot_Noah = np.sum(dmap[msk] / wmap[msk]) ** 2 / np.sum(
                dmap[msk] / wmap[msk] ** 2
            )
            shot_Noah = (
                np.sum(mask_Noah) * hp.nside2pixarea(nside, False) / shot_Noah
            )
        else:
            print("Calculing N_l from weights")
            alpha = self._get_alpha()
            pixel_A = hp.nside2pixarea(self.nside)

            mask = self.get_mask()
            w2_data = self.get_data_maps()["w2"]
            w2_random = self.get_randoms_maps()["w2"]

            goodpix = mask > 0
            N_ell = (
                w2_data[goodpix].sum() + alpha**2 * w2_random[goodpix].sum()
            )
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

    def get_dtype(self):
        return self.dtype

    def get_spin(self):
        return self.spin

    def __get_clean_randoms_with_weights(self, base_name):
        print("Loading randoms for", base_name, flush=True)
        randoms = self._load_full_randoms(base_name)

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
            f"Removing unnecessary columns...",
            flush=True,
        )
        cols_to_keep = ["RA", "DEC"]
        randoms = hstack([randoms[cols_to_keep], Table(weights)])
        return randoms

    def get_clean_randoms_with_weights(self, base_name):
        if base_name in self._loaded_randoms:
            return self._loaded_randoms[base_name]

        fn = "_".join([f"{base_name}-clean-weights", f"{self.suffix}.fits.gz"])
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
        if self.random_maps["n"] is not None:
            return self.random_maps

        list_randoms = self._get_list_randoms()

        random_maps = {
            "n": np.zeros(hp.nside2npix(self.nside), dtype=int),
            "w": np.zeros(hp.nside2npix(self.nside), dtype=float),
            "w2": np.zeros(hp.nside2npix(self.nside), dtype=float),
        }

        # TODO: consider if I want to save the sum of all maps. Problem, it
        # makes the code a bit more complex and it's difficult to know which
        # randoms when into the map.
        for base_name in list_randoms:
            weight_col = f"weight_pzbin{self.zbin + 1}"
            for power in [0, 1, 2]:

                def f():
                    randoms = self.get_clean_randoms_with_weights(base_name)
                    w = np.array(randoms[weight_col])
                    w = w**power if power > 0 else None
                    map_ngal = get_map_from_points(
                        randoms,
                        self.nside,
                        rot=self.rot,
                        w=w,
                    )
                    return map_ngal

                fname = "_".join(
                    [
                        f"map_{base_name}",
                        f"nrand" if power == 0 else weight_col,
                        f"coord{self.coords}",
                        f"ns{self.nside}",
                        f"{self.suffix}",
                    ]
                )
                fname += "w2.fits.gz" if power == 2 else ".fits.gz"
                map_nrand = self._rerun_read_cycle(fname, "FITSMap", f)

                if power == 0:
                    k = "n"
                elif power == 1:
                    k = "w"
                elif power == 2:
                    k = "w2"
                random_maps[k] += map_nrand

        self.random_maps = random_maps
        return self.random_maps

    def _get_list_randoms(self):
        """
        Returns a list of randoms to include
        """
        if self._list_randoms is not None:
            return self._list_randoms

        path = self._randoms_path

        list_randoms = []
        if self._randoms_selection is not None:
            if os.path.isfile(self._randoms_selection):
                # If the path is a file, we assume it contains a list of randoms
                with open(path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.endswith(".fits"):
                            line = line.split(".")[0]
                        list_randoms.append(line)
            elif type(self._randoms_selection) is list:
                list_randoms = self._randoms_selection
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
                    list_randoms.append(f.split(".")[0])
            elif os.path.isfile(path):
                list_randoms.append(path.split(".")[0])
            else:
                raise ValueError(
                    f"Invalid path for randoms: {path}. It should be a directory or a file."
                )

        if not list_randoms:
            raise ValueError(
                "No randoms found. Please check the path or selection."
            )

        self._list_randoms = list_randoms

        return self._list_randoms

    def compute_weights(self, randoms):
        """
        Compute the weights for the randoms.
        :param randoms: astropy Table with the randoms
        :return: weights per z-bin
        """

        # Convert depths to units of magnitude
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            randoms["galdepth_gmag_ebv"] = (
                -2.5 * (np.log10((5 / np.sqrt(randoms["GALDEPTH_G"]))) - 9)
                - 3.214 * randoms["EBV"]
            )
            randoms["galdepth_rmag_ebv"] = (
                -2.5 * (np.log10((5 / np.sqrt(randoms["GALDEPTH_R"]))) - 9)
                - 2.165 * randoms["EBV"]
            )
            randoms["galdepth_zmag_ebv"] = (
                -2.5 * (np.log10((5 / np.sqrt(randoms["GALDEPTH_Z"]))) - 9)
                - 1.211 * randoms["EBV"]
            )

        weights_path = self.config["imaging_weights_coeffs"]

        # Load weights
        with open(weights_path, "r") as f:
            linear_coeffs = yaml.safe_load(f)

        key_weight = "weight"
        # TODO: This might not be the best way to handle this.
        if "no_ebv" in weights_path:
            key_weight = "weight_noebv"

        weights = {}

        # Compute the weights for all z-bins at once
        for bin_index in range(1, 5):  # 4 bins
            print(f"Computing weights for bin {bin_index}...", flush=True)
            key = f"{key_weight}_pzbin{bin_index}"

            weights[key] = self._compute_weights_Noah(
                randoms, linear_coeffs, bin_index
            )

        return weights

    def _load_full_randoms(self, base_name):
        random_path = self._randoms_path
        rand_file = os.path.join(random_path, f"{base_name}.fits")
        rand_mask_name = f"{base_name}-lrgmask_v1.1.fits.gz"
        lrgmask_file = os.path.join(
            self.config["randoms_lrgmask_path"], rand_mask_name
        )
        base_url = "https://data.desi.lbl.gov/public/ets/target/catalogs/dr9/0.49.0/randoms/resolve/"

        # Check if the randoms file exists
        if not os.path.exists(rand_file) and not self._download_missing_random:
            raise FileNotFoundError(
                f"Randoms file {rand_file} does not exist."
            )
        elif not os.path.exists(rand_file) and self._download_missing_random:
            print(
                f"[{base_name}] Randoms file does not exist, downloading...",
                flush=True,
            )
            # Download the randoms file
            url = base_url + f"{base_name}.fits"
            try:
                os.system(
                    f"wget {url} -O {rand_file}",
                )
                print(f"[{base_name}] Downloaded {rand_file}.", flush=True)
            except Exception as e:
                print(f"Error downloading {url}: {e}")
                raise

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
        ]
        columns += [
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

        return randoms

    def _compute_weights_Noah(self, randoms, linear_coeffs, bin_index):
        # Copied from https://github.com/NoahSailer/MaPar/blob/main/maps/assign_randoms_weights.py

        # Assign zero weights to randoms with invalid imaging properties
        mask_bad = np.full(len(randoms), False)
        xnames_fit = list(linear_coeffs["south_bin_1"].keys())
        xnames_fit.remove("intercept")
        for col in xnames_fit:
            mask_bad |= ~np.isfinite(randoms[col])
        if np.sum(mask_bad) != 0:
            print("{} invalid randoms".format(np.sum(mask_bad)))
        #
        weights = np.zeros(len(randoms))
        #
        for field in ["north", "south"]:
            if field == "south":
                photsys = "S"
            elif field == "north":
                photsys = "N"
            #
            mask = randoms["PHOTSYS"] == photsys
            mask &= ~mask_bad
            data = np.column_stack(
                [randoms[mask][xname] for xname in xnames_fit]
            )
            #
            bin_str = "{}_bin_{}".format(field, bin_index)
            #
            # Create array of coefficients, with the first coefficient being the intercept
            coeffs = np.array(
                [linear_coeffs[bin_str]["intercept"]]
                + [linear_coeffs[bin_str][xname] for xname in xnames_fit]
            )
            # create 2-D array of imaging properties, with the first columns being unity
            data1 = np.insert(data, 0, 1.0, axis=1)
            weights[mask] = np.dot(coeffs, data1.T)
        return weights
