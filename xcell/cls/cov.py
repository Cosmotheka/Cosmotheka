#!/usr/bin/python
from .cl import Cl, ClFid
from .common import Data
import os
import numpy as np
import pymaster as nmt


class Cov():
    def __init__(self, data, trA1, trA2, trB1, trB2):
        self.data = Data(data=data)
        self.outdir = self.get_outdir()
        os.makedirs(self.outdir, exist_ok=True)
        self.trA1 = trA1
        self.trA2 = trA2
        self.trB1 = trB1
        self.trB2 = trB2
        cl_dic, clfid_dic = self._load_Cls()
        self.clA1A2 = cl_dic[(trA1, trA2)]
        self.clB1B2 = cl_dic[(trB1, trB2)]
        self.clA1B1 = cl_dic[(trA1, trB1)]
        self.clA1B2 = cl_dic[(trA1, trB2)]
        self.clA2B1 = cl_dic[(trA2, trB1)]
        self.clA2B2 = cl_dic[(trA2, trB2)]
        self.clfid_A1B1 = clfid_dic[(trA1, trB1)]
        self.clfid_A1B2 = clfid_dic[(trA1, trB2)]
        self.clfid_A2B1 = clfid_dic[(trA2, trB1)]
        self.clfid_A2B2 = clfid_dic[(trA2, trB2)]
        self.recompute_cov = self.data.data['recompute']['cov']
        self.recompute_cmcm = self.data.data['recompute']['cmcm']
        self.cov = None
        # Noise marginalization?
        self.nl_marg = False
        if (trA1 == trA2 == trB1 == trB2):
            trconf = self.data.data['tracers'][trA1]
            self.nl_marg = trconf.get('nl_marginalize', False)
            self.nl_prior = trconf.get('nl_prior', 1E30)

    def _load_Cls(self):
        data = self.data.data
        trs_comb = [(self.trA1, self.trA2),
                    (self.trB1, self.trB2),
                    (self.trA1, self.trB1),
                    (self.trA1, self.trB2),
                    (self.trA2, self.trB1),
                    (self.trA2, self.trB2)]

        # Load data Cls
        cl_dic = {}
        for trs in trs_comb:
            if trs not in cl_dic.keys():
                cl_dic[trs] = Cl(data, *trs)

        # Load fiducial Cls
        clfid_dic = {}
        for trs in trs_comb[2:]:
            if trs not in clfid_dic.keys():
                clfid_dic[trs] = ClFid(data, *trs)

        return cl_dic, clfid_dic

    def get_outdir(self):
        root = self.data.data['output']
        outdir = os.path.join(root, 'cov')
        return outdir

    def get_covariance_workspace(self):
        mask1, mask2 = self.clA1A2.get_masks_names()
        mask3, mask4 = self.clB1B2.get_masks_names()
        fname = os.path.join(self.outdir,
                             f'cw__{mask1}__{mask2}__{mask3}__{mask4}.fits')
        cw = nmt.NmtCovarianceWorkspace()
        recompute = self.data.data['recompute']['cmcm']
        if recompute or (not os.path.isfile(fname)):
            n_iter = self.data.data['healpy']['n_iter_cmcm']
            l_toeplitz, l_exact, dl_band = self.data.check_toeplitz('cov')
            fA1, fB1 = self.clA1B1.get_nmt_fields()
            fA2, fB2 = self.clA2B2.get_nmt_fields()
            cw.compute_coupling_coefficients(fA1, fA2, fB1, fB2,
                                             n_iter=n_iter,
                                             l_toeplitz=l_toeplitz,
                                             l_exact=l_exact,
                                             dl_band=dl_band)
            cw.write_to(fname)
            self.recompute_cmcm = False
        else:
            cw.read_from(fname)

        return cw

    def _get_cl_for_cov(self, clab, clab_fid, ma, mb):
        mean_mamb = np.mean(ma * mb)
        nl_cp = clab.get_ell_nl_cp()[1]
        if not mean_mamb:
            cl_cp = np.zeros_like(nl_cp)
        else:
            w = clab.get_workspace()
            cl_cp = w.couple_cell(clab_fid.get_ell_cl()[1])
            cl_cp = (cl_cp + nl_cp) / mean_mamb

        return cl_cp

    def get_covariance(self):
        fname = os.path.join(self.outdir,
                             'cov_{}_{}_{}_{}.npz'.format(self.trA1,
                                                          self.trA2,
                                                          self.trB1,
                                                          self.trB2))
        recompute = self.recompute_cov or self.recompute_cmcm
        if (not recompute) and os.path.isfile(fname):
            self.cov = np.load(fname)['cov']
            return self.cov

        # Load all masks once
        m_a1, m_a2 = self.clA1A2.get_masks()
        m_b1, m_b2 = self.clB1B2.get_masks()

        # Compute weighted Cls
        cla1b1 = self._get_cl_for_cov(self.clA1B1, self.clfid_A1B1, m_a1, m_b1)
        cla1b2 = self._get_cl_for_cov(self.clA1B2, self.clfid_A1B2, m_a1, m_b2)
        cla2b1 = self._get_cl_for_cov(self.clA2B1, self.clfid_A2B1, m_a2, m_b1)
        cla2b2 = self._get_cl_for_cov(self.clA2B2, self.clfid_A2B2, m_a2, m_b2)

        #####
        if np.any(cla1b1) or np.any(cla1b2) or np.any(cla2b1) or \
                np.any(cla2b2):
            wa = self.clA1A2.get_workspace()
            wb = self.clB1B2.get_workspace()
            cw = self.get_covariance_workspace()

            s_a1, s_a2 = self.clA1A2.get_spins()
            s_b1, s_b2 = self.clB1B2.get_spins()

            cov = nmt.gaussian_covariance(cw, s_a1, s_a2, s_b1, s_b2,
                                          cla1b1, cla1b2, cla2b1, cla2b2,
                                          wa, wb)
        else:
            size1 = self.clA1A2.get_ell_cl()[1].size
            size2 = self.clB1B2.get_ell_cl()[1].size
            cov = np.zeros((size1, size2))

        if self.nl_marg:
            _, nl = self.clA1A2.get_ell_nl()
            nl = nl.flatten()
            cov += self.nl_prior**2 * (nl[:, None] * nl[None, :])

        self.cov = cov
        np.savez_compressed(fname, cov=cov)
        self.recompute_cov = False
        return cov


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute Cls and cov from \
                                     data.yml file")
    parser.add_argument('INPUT', type=str, help='Input YAML data file')
    parser.add_argument('trA1', type=str, help='Tracer A1 name')
    parser.add_argument('trA2', type=str, help='Tracer A2 name')
    parser.add_argument('trB1', type=str, help='Tracer B1 name')
    parser.add_argument('trB2', type=str, help='Tracer B2 name')
    args = parser.parse_args()

    data = Data(data_path=args.INPUT).data
    cov = Cov(data, args.trA1, args.trA2, args.trB1, args.trB2)
    cov.get_covariance()
