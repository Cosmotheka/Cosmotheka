#!/usr/bin/python
from .data import Data
from .theory import Theory
import numpy as np
import pymaster as nmt
import os
import warnings


class ClBase():
    def __init__(self, data, tr1, tr2, ignore_existing_yml=False):
        self.data = Data(data=data, ignore_existing_yml=ignore_existing_yml)
        self.tr1 = tr1
        self.tr2 = tr2
        self.nside = self.data.data['sphere']['nside']
        self._read_symmetric = self.data.read_symmetric(tr1, tr2)
        if self._read_symmetric:
            warnings.warn('Reading/computing the symmetric element.')
        self._mapper1 = None
        self._mapper2 = None
        #
        self.cl_file = None
        self.ell = None
        self.cl = None
        self.cl_cp = None

    def get_outdir(self, subdir=''):
        root = self.data.data['output']
        if self._read_symmetric:
            trreq = self.data.get_tracers_bare_name_pair(self.tr2, self.tr1,
                                                         '_')
        else:
            trreq = self.data.get_tracers_bare_name_pair(self.tr1, self.tr2,
                                                         '_')
        outdir = os.path.join(root, subdir, trreq)
        return outdir

    def get_mappers(self):
        if self._mapper1 is None:
            self._mapper1 = self.data.get_mapper(self.tr1)
            self._mapper2 = self._mapper1 if self.tr1 == self.tr2 else \
                self.data.get_mapper(self.tr2)
        return self._mapper1, self._mapper2

    def get_cl_file(self):
        raise ValueError('Cl_Base class is not to be used directly!')

    def get_ell_cl(self):
        if self.ell is None:
            self.get_cl_file()
        return self.ell, self.cl

    def get_ell_cl_cp(self):
        """
        Return the noisless coupled Cell
        """
        if self.ell is None:
            self.get_cl_file()
        return np.arange(3 * self.nside), self.cl_cp

    def get_n_cls(self):
        s1, s2 = self.get_spins()
        nmap1 = 1 + (s1 > 0)
        nmap2 = 1 + (s2 > 0)
        return nmap1 * nmap2

    def get_spins(self):
        mapper1, mapper2 = self.get_mappers()
        s1 = mapper1.get_spin()
        s2 = mapper2.get_spin()
        return s1, s2

    def get_dtypes(self):
        mapper1, mapper2 = self.get_mappers()
        d1 = mapper1.get_dtype()
        d2 = mapper2.get_dtype()
        return d1, d2


class Cl(ClBase):
    def __init__(self, data, tr1, tr2, ignore_existing_yml=False):
        super().__init__(data, tr1, tr2, ignore_existing_yml)
        self.outdir = self.get_outdir()
        os.makedirs(self.outdir, exist_ok=True)
        self.b = self.get_NmtBin()
        self.recompute_cls = self.data.data['recompute']['cls']
        self.recompute_mcm = self.data.data['recompute']['mcm']
        # Not needed to load cl if already computed
        self._w = None
        self._wcov = None
        ##################
        self.nl = None
        self.nl_cp = None
        self.cl_cp = None
        self.wins = None
        self.cls_cov = None

    def get_NmtBin(self):
        if self._read_symmetric:
            trs = self.data.get_tracers_bare_name_pair(self.tr2, self.tr1)
        else:
            trs = self.data.get_tracers_bare_name_pair(self.tr1, self.tr2)
        if ((trs in self.data.data['cls']) and
                ('bpw_edges' in self.data.data['cls'][trs].keys())):
            bpw_edges = np.array(self.data.data['cls'][trs]['bpw_edges'])
        else:
            bpw_edges = np.array(self.data.data['bpw_edges'])
        nside = self.nside
        # 3*nside == ells[-1] + 1
        bpw_edges = bpw_edges[bpw_edges <= 3 * nside]
        # Exhaust lmax --> gives same result as previous method,
        # but adds 1 bpw (not for 4096)
        if 3*nside not in bpw_edges:
            bpw_edges = np.append(bpw_edges, 3*nside)
        b = nmt.NmtBin.from_edges(bpw_edges[:-1], bpw_edges[1:])
        return b

    def get_nmt_fields(self):
        mapper1, mapper2 = self.get_mappers()
        f1 = mapper1.get_nmt_field()
        f2 = mapper2.get_nmt_field()
        return f1, f2

    def get_workspace(self, read_unbinned_MCM=True):
        if self._w is None:
            self._w = \
                self._compute_workspace(read_unbinned_MCM=read_unbinned_MCM)
        return self._w

    def get_workspace_cov(self):
        if self._wcov is None:
            spin0 = self.data.data['cov'].get('spin0', False)
            if spin0 and (self.get_spins() != (0, 0)):
                self._wcov = self._compute_workspace(spin0=spin0,
                                                     read_unbinned_MCM=False)
            else:
                self._wcov = self.get_workspace(read_unbinned_MCM=False)

        return self._wcov

    def _is_cross(self, mapper1, mapper2):
        mask1 = mapper1.mask
        mask2 = mapper2.mask
        sky = np.array(mask1*maks2)
        if np.all(sky==0):
            cross = False 
        else:
            cross = True
        return cross

    def _compute_workspace(self, spin0=False, read_unbinned_MCM=True):
        # Check if the fields are already of spin0 to avoid computing the
        # workspace twice
        spin0 = spin0 and (self.get_spins() != (0, 0))
        mask1, mask2 = self.get_masks_names()
        if self._read_symmetric:
            mask1, mask2 = mask2, mask1
        if spin0:
            fname = os.path.join(self.outdir, f'w0__{mask1}__{mask2}.fits')
        else:
            fname = os.path.join(self.outdir, f'w__{mask1}__{mask2}.fits')
        w = nmt.NmtWorkspace()
        if self.recompute_mcm or (not os.path.isfile(fname)):
            n_iter = self.data.data['sphere']['n_iter_mcm']
            l_toeplitz, l_exact, dl_band = self.data.check_toeplitz('cls')
            if spin0:
                m1, m2 = self.get_mappers()
                msk1 = m1.get_mask()
                msk2 = m2.get_mask()
                f1 = nmt.NmtField(msk1, None, spin=0)
                f2 = nmt.NmtField(msk2, None, spin=0)
            else:
                f1, f2 = self.get_nmt_fields()
            w.compute_coupling_matrix(f1, f2, self.b, n_iter=n_iter,
                                      l_toeplitz=l_toeplitz, l_exact=l_exact,
                                      dl_band=dl_band)
            # Recheck again in case other process has started writing it
            if (not os.path.isfile(fname)):
                w.write_to(fname)
            self.recompute_mcmc = False
        else:
            w.read_from(fname, read_unbinned_MCM)
        return w
    
    def cross_match_gals(self, cat1, cat2, cat1_columns,
                     cat2_columns, return_ix_xmat=False):
    """
    Match the galaxies in both cat1_sample and cat2_sample.

    Arguments
    ---------
        cat1 (fits): frist cataloge.
        cat2 (fits): second catalog.
        cat1_columns (list): List with the column names of the right ascension
        and declination columns for the first cataloge. They are asumed to be in
        degrees.
        cat2_columns (list): Same as photo_columns but for the second cataloge.
        return_ix_xmat (bool): If True return the indices that slice the
        photo_sample and spec_sample after cross-matching. Default False.
    Returns
    -------
        fits: photo_xmat: Subsample of the photometric sample that
        cross-matches with the spectroscopic
        fits: spec_xmat: As above, but for the spectroscopic sample
        array: pix_xmat: Array with the indices of the galaxies in the
        photometric sample with spectroscopic counterpart.
    """
    # Cut photo_sample around COSMOS area to speed up matching
    ra1, dec1 = columns_from_fits(cat1, cat1_columns)
    ra2, dec2 = columns_from_fits(cat2, cat2_columns)
    arcmin = 10/60
    sel = (ra1 >= ra2.min() - arcmin) * (ra1 <= ra2.max() + arcmin) * \
          (dec1 >= dec2.min() - arcmin) * (dec1 <= dec2.max() + arcmin)

    ra1 = ra1[sel]
    ra2 = ra2[sel]
    dec1 = dec1[sel]
    dec2 = dec2[sel]
    
    # Based on
    # https://github.com/LSSTDESC/DEHSC_LSS/blob/master/hsc_lss/cosmos_weight.py
    # Match coordinates
    cat1_skycoord = SkyCoord(ra=ra1, dec=dec1, unit='deg')
    cat2_skycoord = SkyCoord(ra=ra2, dec=dec2, unit='deg')


    # Nearest neighbors
    # Cross-match from spec to photo, not photo to spec
    cat1_index, dist_2d, _ = \
        cat2_skycoord.match_to_catalog_sky(cat1_skycoord)

    # Cut everything further than 1 arcsec
    mask = dist_2d.degree * 60 * 60 < 1
    pix_xmat = cat1_index[mask]
    cat1_xmat = cat1_sample[pix_xmat]
    cat2_xmat = cat2_sample[mask]

    # Check if there are multiple cross-matchings
    rdev = pix_xmat.size / np.unique(pix_xmat).size - 1
    print(f'Multiple cross-matching: {100 * rdev:.2f}%', flush=True)

    if np.abs(rdev) > 0:
        print('Removing multiple cross-matching', flush=True)
        pix_xmat, dist_2d_xmat, sel = remove_further_duplicates(pix_xmat,
                                                                dist_2d[mask])
        # Update mask
        ix_to_remove = np.where(~sel)[0]
        ix_true_in_mask = np.where(mask)[0]
        mask[ix_true_in_mask[ix_to_remove]] = False

        rdev = pix_xmat.size / np.unique(pix_xmat).size - 1
        print(f'Multiple cross-matching after cleaning: {100 * rdev:.2f}%',
              flush=True)

        cat1_xmat, cat2_xmat = cat1_xmat[sel], cat2_xmat[sel]

    if return_ix_xmat:
        return cat1_xmat, cat2_xmat, pix_xmat, mask
    else:
        return cat1_xmat, cat2_xmat

    
    def get_shared_shot_noise(self, mapper1, mapper2):
        cat1 = mapper1.get_catalog()
        cat2 = mapper2.get_catalog()
        cols1 = mapper1._get_cat_cols()
        cols2 = mapper2._get_cat_cols()
        shared_cat = self.cross_match_gals(cat1, cat2, cols1, cols2)
        shared_count = len(shared_cat)
        if shared_count == 0:
            shot_noise = 0 
        else: 
            cat1_count = len(cat1)
            cat2_count = len(cat2)
            shot_noise = (shared_count/(cat1_count+cat2_count))
        return shot_noise
    

    def get_cl_file(self):
        if self._read_symmetric:
            fname = os.path.join(self.outdir, f'cl_{self.tr2}_{self.tr1}.npz')
        else:
            fname = os.path.join(self.outdir, f'cl_{self.tr1}_{self.tr2}.npz')
        ell = self.b.get_effective_ells()
        recompute = self.recompute_cls or self.recompute_mcm
        if recompute or (not os.path.isfile(fname)):
            mapper1, mapper2 = self.get_mappers()
            f1, f2 = self.get_nmt_fields()
            w = self.get_workspace()
            wins = w.get_bandpower_windows()

            # Compute power spectrum
            # If auto-correlation, compute noise and,
            # if needed, the custom signal power spectrum.
            auto = self.tr1 == self.tr2
            cross = self._is_cross(mapper1, mapper2)
            # Noise
            if auto:
                nl_cp = mapper1.get_nl_coupled()
                nl = w.decouple_cell(nl_cp)
            else:
                n_cls = self.get_n_cls()
                nl_cp = np.zeros((n_cls, 3 * self.nside))
                nl = np.zeros([n_cls, self.b.get_n_bands()])
            if cross:
                nl_cross = self.get_shared_shot_noise(mapper1, mapper2)
                nl += nl_cross*np.ones_like(nl)
                
            # Signal
            if auto and mapper1.custom_auto:
                cl_cp = mapper1.get_cl_coupled()
                cl = w.decouple_cell(cl_cp)
                cls_cov = mapper1.get_cls_covar_coupled()
                # This function should return a dictionary with the
                # following contents:
                #  - 'cross': the C_ell that should be used for the
                #       auto-correlation of this mapper when
                #       calculating any cross-covariance involving
                #       it. E.g. if we call this field "a", this will
                #       play the role of C_ell^aa when computing the
                #       cross-covariance Cov(C_ell^ab, C_ell^aa)
                #       or Cov(C_ell^ab,C_ell^ac).
                #  - 'auto_11/12/22': when computing the
                #       auto-covariance of this mapper's auto-correlation,
                #       it will be computed as the auto-covariance of
                #       a general power spectrum C_ell^12, which involves
                #       3 power spectra: C_ell^11, C_ell^12, C_ell^22.
                #       These are given by the corresponding entries in
                #       the dictionary.
                cl_cov_cp = cls_cov['cross']
                cl_cov_11_cp = cls_cov['auto_11']
                cl_cov_12_cp = cls_cov['auto_12']
                cl_cov_22_cp = cls_cov['auto_22']
                # mapper.get_cl_coupled is assumed to return
                # the noise-less power spectrum. No need
                # to subtract it here.
            else:
                cl_cov_cp = nmt.compute_coupled_cell(f1, f2)
                # A standard auto-correlation auto-covariance
                # is just ~propto 2*C_ell^2 rather than
                # (C_ell^11 C_ell^22+C_ell^12^2), which can be
                # achieved by equating all 3 C_ells.
                cl_cov_11_cp = cl_cov_cp
                cl_cov_12_cp = cl_cov_cp
                cl_cov_22_cp = cl_cov_cp
                cl = w.decouple_cell(cl_cov_cp)
                cl_cp = cl_cov_cp - nl_cp
                cl -= nl
            # Note that while we have subtracted the noise
            # bias from `cl_cp`, `cl_cov_cp` still includes it.
            correction = 1
            if (mapper1.mask_power > 1) or (mapper2.mask_power > 1):
                # Applies correction factor if masks have been
                # implicitly applied to the maps
                # See ACTk for reference
                n_a = mapper1.mask_power
                n_b = mapper2.mask_power
                w_a = mapper1.get_mask()
                w_b = mapper2.get_mask()
                correction = np.mean(w_a*w_b)/np.mean(w_a**n_a*w_b**n_b)
                # Apply correction to all Cl's
                cl *= correction
                cl_cp *= correction
                cl_cov_cp *= correction
                cl_cov_11_cp *= correction
                cl_cov_12_cp *= correction
                cl_cov_22_cp *= correction

            np.savez(fname, ell=ell, cl=cl, cl_cp=cl_cp, nl=nl,
                     nl_cp=nl_cp, cl_cov_cp=cl_cov_cp,
                     cl_cov_11_cp=cl_cov_11_cp,
                     cl_cov_12_cp=cl_cov_12_cp,
                     cl_cov_22_cp=cl_cov_22_cp,
                     wins=wins, correction=correction)
            self.recompute_cls = False

        cl_file = np.load(fname)
        cl = cl_file['cl']
        if np.any(ell != cl_file['ell']):
            raise ValueError('The file {} does not have the expected bpw. \
                             Aborting!'.format(fname))

        self.cl_file = cl_file
        self.ell = cl_file['ell']
        self.cl = cl_file['cl']
        self.cl_cp = cl_file['cl_cp']
        self.nl = cl_file['nl']
        self.nl_cp = cl_file['nl_cp']
        self.wins = cl_file['wins']
        self.cls_cov = {'cross': cl_file['cl_cov_cp'],
                        'auto_11': cl_file['cl_cov_11_cp'],
                        'auto_12': cl_file['cl_cov_12_cp'],
                        'auto_22': cl_file['cl_cov_22_cp']}

        return cl_file

    def get_ell_nl(self):
        if self.ell is None:
            self.get_cl_file()
        return self.ell, self.nl

    def get_ell_nl_cp(self):
        if self.nl_cp is None:
            self.get_cl_file()
        return np.arange(3 * self.nside), self.nl_cp

    def get_ell_cl_cp_cov(self):
        if self.ell is None:
            self.get_cl_file()
        return np.arange(3 * self.nside), self.cls_cov['cross']

    def get_ell_cls_cp_cov_auto(self):
        if self.ell is None:
            self.get_cl_file()
        cl11 = self.cls_cov['auto_11']
        cl12 = self.cls_cov['auto_12']
        cl22 = self.cls_cov['auto_22']
        ell = np.arange(3 * self.nside)
        return ell, cl11, cl12, cl22

    def get_masks(self):
        mapper1, mapper2 = self.get_mappers()
        m1 = mapper1.get_mask()
        m2 = mapper2.get_mask()
        return m1, m2

    def get_masks_names(self):
        mapper1, mapper2 = self.get_mappers()
        m1 = mapper1.mask_name
        m2 = mapper2.mask_name
        return m1, m2

    def get_bandpower_windows(self):
        if self.ell is None:
            self.get_cl_file()
        return self.wins


class ClFid(ClBase):
    def __init__(self, data, tr1, tr2, ignore_existing_yml=False):
        super().__init__(data, tr1, tr2, ignore_existing_yml)
        self.supported_dtypes = ['galaxy_density',
                                 'galaxy_shear',
                                 'cmb_tSZ',
                                 'cmb_convergence']
        m1, m2 = self.get_mappers()
        dt1 = m1.get_dtype()
        dt2 = m2.get_dtype()
        if ((dt1 not in self.supported_dtypes) or
                (dt2 not in self.supported_dtypes)):
            raise NotImplementedError("Fiducial C_ells cannot be "
                                      f"computed for types {dt1}, {dt2}")

        self.outdir = self.get_outdir('fiducial')
        os.makedirs(self.outdir, exist_ok=True)
        self.th = Theory(self.data.data)
        self._ccl_tr1 = None
        self._ccl_tr2 = None
        self.cl_data = Cl(data, tr1, tr2, ignore_existing_yml=True)
        self.cl_binned = None  # This is the one you compare with data
        self.ell_binned = None

    def get_tracers_ccl(self):
        if self._ccl_tr1 is None:
            mapper1, mapper2 = self.get_mappers()
            trlist = self.data.data['tracers']
            self._ccl_tr1 = self.th.compute_tracer_ccl(self.tr1,
                                                       trlist[self.tr1],
                                                       mapper1)
            self._ccl_tr2 = self.th.compute_tracer_ccl(self.tr2,
                                                       trlist[self.tr2],
                                                       mapper2)
        return self._ccl_tr1, self._ccl_tr2

    def get_cl_file(self):
        nside = self.data.data['sphere']['nside']
        if self._read_symmetric:
            fname = os.path.join(self.outdir, f'cl_{self.tr2}_{self.tr1}.npz')
        else:
            fname = os.path.join(self.outdir, f'cl_{self.tr1}_{self.tr2}.npz')
        ell = np.arange(3 * nside)
        if not os.path.isfile(fname):
            ccl_tr1, ccl_tr2 = self.get_tracers_ccl()
            cl = self.th.get_ccl_cl(ccl_tr1, ccl_tr2, ell)
            b1 = self.data.get_bias(self.tr1)
            b2 = self.data.get_bias(self.tr2)
            cl *= b1*b2
            tracers = self.data.data['tracers']
            fiducial = self.data.data['cov']['fiducial']
            d1, d2 = self.get_dtypes()
            for dtype, tr in zip([self.tr1, self.tr2], [d1, d2]):
                if (dtype == 'galaxy_shear') and fiducial['wl_m']:
                    cl = (1 + tracers[tr]['m']) * cl

            # This is only valid for LCDM and spins 0 and 2.
            s1, s2 = self.get_spins()
            size = s1 + s2
            if size == 0:
                size = 1
            cl_vector = np.zeros((size, cl.size))
            cl_vector[0] = cl

            w = self.cl_data.get_workspace()
            cl_cp = w.couple_cell(cl_vector)
            cl_binned = w.decouple_cell(cl_cp)
            ell_binned = self.cl_data.get_ell_cl()[0]
            np.savez_compressed(fname, cl=cl_vector, ell=ell, cl_cp=cl_cp,
                                ell_binned=ell_binned, cl_binned=cl_binned)

        cl_file = np.load(fname)
        if np.any(cl_file['ell'] != ell):
            raise ValueError(f'The ell in {fname} does not match the ell \
                               from nside={nside}')

        self.ell = cl_file['ell']
        self.cl = cl_file['cl']
        self.cl_cp = cl_file['cl_cp']
        self.cl_binned = cl_file['cl_binned']
        self.ell_binned = cl_file['ell_binned']
        return cl_file

    def get_ell_cl_binned(self):
        """
        Return the binned Cell (i.e. decouple(couple(cl))). This one is the one
        you compare with data.
        """
        if self.ell is None:
            self.get_cl_file()
        return self.ell_binned, self.cl_binned


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from \
                                     data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('tr1', type=str, help='Tracer 1 name')
    parser.add_argument('tr2', type=str, help='Tracer 2 name')
    parser.add_argument('--fiducial', default=False, action='store_true',
                        help='Compute the fiducial model Cl')
    args = parser.parse_args()

    data = Data(data_path=args.INPUT).data
    if args.fiducial:
        cl = ClFid(data, args.tr1, args.tr2)
    else:
        cl = Cl(data, args.tr1, args.tr2)
    cl.get_cl_file()
