#!/usr/bin/python
from .cl import Cl, ClFid
from .data import Data
from .theory import Theory
import os
import numpy as np
import pymaster as nmt
import time


class Cov():
    def __init__(self, data, trA1, trA2, trB1, trB2,
                 ignore_existing_yml=False):
        self.data = Data(data=data, ignore_existing_yml=ignore_existing_yml)
        self.tmat = self.data.get_tracer_matrix()
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
        self.clfid_A1A2 = clfid_dic[(trA1, trA2)]
        self.clfid_B1B2 = clfid_dic[(trB1, trB2)]
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
        # Spin-0 approximation
        self.spin0 = self.data.data['cov'].get('spin0', False)
        # Multiplicative bias marginalization
        self.m_marg = self.data.data['cov'].get('m_marg', False)
        self.do_NG = self.data.data['cov'].get('non_Gaussian', False)

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
                cl_dic[trs] = Cl(data, *trs, ignore_existing_yml=True)

        # Load fiducial Cls
        clfid_dic = {}
        for trs in trs_comb:
            if trs not in clfid_dic.keys():
                # Cl computed from data if needed
                if self.tmat[trs]['clcov_from_data']:
                    cl = cl_dic[trs]
                else:
                    # Try to compute the fiducial Cl
                    try:
                        cl = ClFid(data, *trs, ignore_existing_yml=True)
                    except NotImplementedError as e:
                        if self.data.data['cov'].get('data_fallback', False):
                            # If that fails (e.g. unknown data type)
                            # this will be computed from the data.
                            cl = cl_dic[trs]
                            self.tmat[trs]['clcov_from_data'] = True
                        else:
                            raise e
                clfid_dic[trs] = cl

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
            n_iter = self.data.data['sphere']['n_iter_cmcm']
            l_toeplitz, l_exact, dl_band = self.data.check_toeplitz('cov')
            fA1, fB1 = self.clA1B1.get_nmt_fields()
            fA2, fB2 = self.clA2B2.get_nmt_fields()
            cw.compute_coupling_coefficients(fA1, fA2, fB1, fB2,
                                             n_iter=n_iter,
                                             l_toeplitz=l_toeplitz,
                                             l_exact=l_exact,
                                             dl_band=dl_band)
            # Recheck again in case other process has started writing it
            if (not os.path.isfile(fname)):
                cw.write_to(fname)
            self.recompute_cmcm = False
        else:
            cw.read_from(fname)

        return cw

    def _get_cl_for_cov(self, clab, clab_fid, ma, mb):
        mean_mamb = np.mean(ma * mb)
        if not mean_mamb:
            cl_cp = np.zeros((clab.get_n_cls(), 3*clab.nside))
        else:
            if isinstance(clab_fid, ClFid):  # Compute from theory
                cl_cp = clab_fid.get_ell_cl_cp()[1] + clab.get_ell_nl_cp()[1]
            else:  # Compute from data
                # In this case we've requested to compute this
                # C_ell from the data, so `clab_fid` is actually
                # a `Cl` object (not a `ClFid` object).
                cl_cp = clab_fid.get_ell_cl_cp_cov()[1]
            cl_cp = cl_cp / mean_mamb

        return cl_cp

    def _get_covariance_spin0_approx(self, cw,  s_a1, s_a2, s_b1, s_b2, cla1b1,
                                     cla1b2, cla2b1, cla2b2, wa, wb):
        nbpw_a = wa.wsp.bin.n_bands
        nbpw_b = wb.wsp.bin.n_bands
        nclsa = np.max([1, s_a1 + s_a2])
        nclsb = np.max([1, s_b1 + s_b2])
        cov = np.zeros([nbpw_a, nclsa, nbpw_b, nclsb])
        # 00, 02
        if (s_a1 + s_a2 == 0) and (s_b1 + s_b2 == 2):
            c_tt_te = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[0]],
                                              [cla2b1[0]], [cla2b2[0]],
                                              wa, wb)
            c_tt_tb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[-1]],
                                              [cla2b1[0]], [cla2b2[-1]],
                                              wa, wb)
            cov[:, 0, :, 0] = c_tt_te
            cov[:, 0, :, 1] = c_tt_tb
        # 02, 00
        elif (s_a1 + s_a2 == 2) and (s_b1 + s_b2 == 0):
            c_te_tt = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[0]],
                                              [cla2b1[0]], [cla2b2[0]],
                                              wa, wb)
            c_tb_tt = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[0]],
                                              [cla2b1[-1]], [cla2b2[-1]],
                                              wa, wb)
            cov[:, 0, :, 0] = c_te_tt
            cov[:, 1, :, 0] = c_tb_tt
        # 00, 22
        elif (s_a1 + s_a2 == 0) and (s_b1 + s_b2 == 4):
            c_tt_ee = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[0]],
                                              [cla2b1[0]], [cla2b2[0]],
                                              wa, wb)
            c_tt_eb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[-1]],
                                              [cla2b1[0]], [cla2b2[-1]],
                                              wa, wb)
            c_tt_be = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[-1]], [cla1b2[0]],
                                              [cla2b1[-1]], [cla2b2[0]],
                                              wa, wb)
            c_tt_bb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[-1]], [cla1b2[-1]],
                                              [cla2b1[-1]], [cla2b2[-1]],
                                              wa, wb)
            cov[:, 0, :, 0] = c_tt_ee
            cov[:, 0, :, 1] = c_tt_eb
            cov[:, 0, :, 2] = c_tt_be
            cov[:, 0, :, 3] = c_tt_bb
        # 22, 00
        elif (s_a1 + s_a2 == 4) and (s_b1 + s_b2 == 0):
            c_ee_tt = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[0]],
                                              [cla2b1[0]], [cla2b2[0]],
                                              wa, wb)
            c_eb_tt = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[0]],
                                              [cla2b1[-1]], [cla2b2[-1]],
                                              wa, wb)
            c_be_tt = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[-1]], [cla1b2[-1]],
                                              [cla2b1[0]], [cla2b2[0]],
                                              wa, wb)
            c_bb_tt = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[-1]], [cla1b2[-1]],
                                              [cla2b1[-1]], [cla2b2[-1]],
                                              wa, wb)
            cov[:, 0, :, 0] = c_ee_tt
            cov[:, 1, :, 0] = c_eb_tt
            cov[:, 2, :, 0] = c_be_tt
            cov[:, 3, :, 0] = c_bb_tt
        # 02, 02
        elif (s_a1 + s_a2 == 2) and (s_b1 + s_b2 == 2):
            c_te_te = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[0]],
                                              [cla2b1[0]], [cla2b2[0]],
                                              wa, wb)
            c_te_tb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[-1]],
                                              [cla2b1[0]], [cla2b2[1]],
                                              wa, wb)
            c_tb_te = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[0]],
                                              [cla2b1[-1]], [cla2b2[2]],
                                              wa, wb)
            c_tb_tb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[-1]],
                                              [cla2b1[-1]], [cla2b2[3]],
                                              wa, wb)
            cov[:, 0, :, 0] = c_te_te
            cov[:, 0, :, 1] = c_te_tb
            cov[:, 1, :, 0] = c_tb_te
            cov[:, 1, :, 1] = c_tb_tb
        # 02, 22
        elif (s_a1 + s_a2 == 2) and (s_b1 + s_b2 == 4):
            c_te_ee = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[0]],
                                              [cla2b1[0]], [cla2b2[0]],
                                              wa, wb)
            c_te_eb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[-1]],
                                              [cla2b1[0]], [cla2b2[1]],
                                              wa, wb)
            c_te_be = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[-1]], [cla1b2[0]],
                                              [cla2b1[1]], [cla2b2[0]],
                                              wa, wb)
            c_te_bb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[-1]], [cla1b2[-1]],
                                              [cla2b1[1]], [cla2b2[1]],
                                              wa, wb)
            c_tb_ee = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[0]],
                                              [cla2b1[2]], [cla2b2[2]],
                                              wa, wb)
            c_tb_eb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[-1]],
                                              [cla2b1[2]], [cla2b2[3]],
                                              wa, wb)
            c_tb_be = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[-1]], [cla1b2[0]],
                                              [cla2b1[3]], [cla2b2[2]],
                                              wa, wb)
            c_tb_bb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[-1]], [cla1b2[-1]],
                                              [cla2b1[3]], [cla2b2[3]],
                                              wa, wb)
            cov[:, 0, :, 0] = c_te_ee
            cov[:, 0, :, 1] = c_te_eb
            cov[:, 0, :, 2] = c_te_be
            cov[:, 0, :, 3] = c_te_bb
            cov[:, 1, :, 0] = c_tb_ee
            cov[:, 1, :, 1] = c_tb_eb
            cov[:, 1, :, 2] = c_tb_be
            cov[:, 1, :, 3] = c_tb_bb
        # 22, 02
        elif (s_a1 + s_a2 == 4) and (s_b1 + s_b2 == 2):
            c_ee_te = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[0]],
                                              [cla2b1[0]], [cla2b2[0]],
                                              wa, wb)
            c_ee_tb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[1]],
                                              [cla2b1[0]], [cla2b2[1]],
                                              wa, wb)
            c_eb_te = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[0]],
                                              [cla2b1[-1]], [cla2b2[2]],
                                              wa, wb)
            c_eb_tb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[1]],
                                              [cla2b1[-1]], [cla2b2[3]],
                                              wa, wb)
            c_be_te = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[-1]], [cla1b2[2]],
                                              [cla2b1[0]], [cla2b2[0]],
                                              wa, wb)
            c_be_tb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[-1]], [cla1b2[3]],
                                              [cla2b1[0]], [cla2b2[1]],
                                              wa, wb)
            c_bb_te = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[-1]], [cla1b2[2]],
                                              [cla2b1[-1]], [cla2b2[2]],
                                              wa, wb)
            c_bb_tb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[-1]], [cla1b2[3]],
                                              [cla2b1[-1]], [cla2b2[3]],
                                              wa, wb)
            cov[:, 0, :, 0] = c_ee_te
            cov[:, 1, :, 0] = c_eb_te
            cov[:, 2, :, 0] = c_be_te
            cov[:, 3, :, 0] = c_bb_te
            cov[:, 0, :, 1] = c_ee_tb
            cov[:, 1, :, 1] = c_eb_tb
            cov[:, 2, :, 1] = c_be_tb
            cov[:, 3, :, 1] = c_bb_tb
        # 22, 22
        elif (s_a1 + s_a2 == 4) and (s_b1 + s_b2 == 4):
            c_ee_ee = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[0]],
                                              [cla2b1[0]], [cla2b2[0]],
                                              wa, wb)
            c_ee_eb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[1]],
                                              [cla2b1[0]], [cla2b2[1]],
                                              wa, wb)
            c_ee_be = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[1]], [cla1b2[0]],
                                              [cla2b1[1]], [cla2b2[0]],
                                              wa, wb)
            c_ee_bb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[1]], [cla1b2[1]],
                                              [cla2b1[1]], [cla2b2[1]],
                                              wa, wb)
            c_eb_ee = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[0]],
                                              [cla2b1[2]], [cla2b2[2]],
                                              wa, wb)
            c_eb_eb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[0]], [cla1b2[1]],
                                              [cla2b1[2]], [cla2b2[3]],
                                              wa, wb)
            c_eb_be = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[1]], [cla1b2[0]],
                                              [cla2b1[3]], [cla2b2[2]],
                                              wa, wb)
            c_eb_bb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[1]], [cla1b2[1]],
                                              [cla2b1[3]], [cla2b2[3]],
                                              wa, wb)
            c_be_ee = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[2]], [cla1b2[2]],
                                              [cla2b1[0]], [cla2b2[0]],
                                              wa, wb)
            c_be_eb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[2]], [cla1b2[3]],
                                              [cla2b1[0]], [cla2b2[1]],
                                              wa, wb)
            c_be_be = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[3]], [cla1b2[2]],
                                              [cla2b1[1]], [cla2b2[0]],
                                              wa, wb)
            c_be_bb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[3]], [cla1b2[3]],
                                              [cla2b1[1]], [cla2b2[1]],
                                              wa, wb)
            c_bb_ee = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[2]], [cla1b2[2]],
                                              [cla2b1[2]], [cla2b2[2]],
                                              wa, wb)
            c_bb_eb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[2]], [cla1b2[3]],
                                              [cla2b1[2]], [cla2b2[3]],
                                              wa, wb)
            c_bb_be = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[3]], [cla1b2[2]],
                                              [cla2b1[3]], [cla2b2[2]],
                                              wa, wb)
            c_bb_bb = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cla1b1[3]], [cla1b2[3]],
                                              [cla2b1[3]], [cla2b2[3]],
                                              wa, wb)
            cov[:, 0, :, 0] = c_ee_ee
            cov[:, 0, :, 1] = c_ee_eb
            cov[:, 0, :, 2] = c_ee_be
            cov[:, 0, :, 3] = c_ee_bb
            cov[:, 1, :, 0] = c_eb_ee
            cov[:, 1, :, 1] = c_eb_eb
            cov[:, 1, :, 2] = c_eb_be
            cov[:, 1, :, 3] = c_eb_bb
            cov[:, 2, :, 0] = c_be_ee
            cov[:, 2, :, 1] = c_be_eb
            cov[:, 2, :, 2] = c_be_be
            cov[:, 2, :, 3] = c_be_bb
            cov[:, 3, :, 0] = c_bb_ee
            cov[:, 3, :, 1] = c_bb_eb
            cov[:, 3, :, 2] = c_bb_be
            cov[:, 3, :, 3] = c_bb_bb

        return cov.reshape([nclsa*nbpw_a, nclsb*nbpw_b])

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
        itime = time.time()
        m_a1, m_a2 = self.clA1A2.get_masks()
        m_b1, m_b2 = self.clB1B2.get_masks()
        ftime = time.time()
        print(f'Masks read. It took {(ftime - itime) / 60} min', flush=True)

        # Compute weighted Cls
        # Check if it's the auto-covariance of an auto-correlation
        auto_auto = self.trA1 == self.trA2 == self.trB1 == self.trB2
        # Check if it must be computed from data
        aa_data = auto_auto and self.tmat[(self.trA1,
                                           self.trA2)]['clcov_from_data']
        # If so, get these C_ells
        itime = time.time()
        if aa_data:
            mean_mamb = np.mean(m_a1**2)
            _, cla1b1, cla1b2, cla2b2 = self.clA1B1.get_ell_cls_cp_cov_auto()
            cla2b2 = cla2b2 / mean_mamb
            cla2b1 = cla1b2 / mean_mamb
            cla1b2 = cla1b2 / mean_mamb
            cla1b1 = cla1b1 / mean_mamb
        else:
            cla1b1 = self._get_cl_for_cov(self.clA1B1, self.clfid_A1B1,
                                          m_a1, m_b1)
            cla1b2 = self._get_cl_for_cov(self.clA1B2, self.clfid_A1B2,
                                          m_a1, m_b2)
            cla2b1 = self._get_cl_for_cov(self.clA2B1, self.clfid_A2B1,
                                          m_a2, m_b1)
            cla2b2 = self._get_cl_for_cov(self.clA2B2, self.clfid_A2B2,
                                          m_a2, m_b2)
        ftime = time.time()
        print(f'Computed C_ells. It took {(ftime - itime) / 60} min',
              flush=True)

        notnull = (np.any(cla1b1) or np.any(cla1b2) or
                   np.any(cla2b1) or np.any(cla2b2))
        s_a1, s_a2 = self.clA1A2.get_spins()
        s_b1, s_b2 = self.clB1B2.get_spins()

        size1 = self.clA1A2.get_ell_cl()[1].size
        size2 = self.clB1B2.get_ell_cl()[1].size
        cov_G = np.zeros((size1, size2))
        cov_NG = np.zeros((size1, size2))
        cov_nlm = np.zeros((size1, size2))
        cov_mm = np.zeros((size1, size2))
        if notnull:
            itime = time.time()
            wa = self.clA1A2.get_workspace_cov()
            wb = self.clB1B2.get_workspace_cov()
            ftime = time.time()
            print(f'Read workspaces. It took {(ftime - itime) / 60} min',
                  flush=True)

            itime = time.time()
            cw = self.get_covariance_workspace()
            ftime = time.time()
            print('Get covariance workspace. It took ' +
                  f'{(ftime - itime) / 60} min', flush=True)

            itime = time.time()
            if self.spin0 and (s_a1 + s_a2 + s_b1 + s_b2 != 0):
                cov_G = self._get_covariance_spin0_approx(cw, s_a1, s_a2, s_b1,
                                                          s_b2, cla1b1, cla1b2,
                                                          cla2b1, cla2b2, wa,
                                                          wb)

            else:
                cov_G = nmt.gaussian_covariance(cw, s_a1, s_a2, s_b1, s_b2,
                                                cla1b1, cla1b2, cla2b1, cla2b2,
                                                wa, wb)
            ftime = time.time()
            print('Computed Gaussian covariance. It took ' +
                  f'{(ftime - itime) / 60} min', flush=True)

        if self.nl_marg and notnull:
            itime = time.time()
            cov_nlm = self.get_covariance_nl_marg()
            ftime = time.time()
            print(f'Computed nl_marg. It took {(ftime - itime) / 60} min',
                  flush=True)

        if self.m_marg and notnull:
            itime = time.time()
            cov_mm = self.get_covariance_m_marg()
            ftime = time.time()
            print(f'Computed m_marg. It took {(ftime - itime) / 60} min',
                  flush=True)

        if self.do_NG and notnull:
            fsky = self.data.data['cov'].get('fsky_NG', None)
            if fsky is None:  # Calculate from masks
                fsky = np.mean(((m_a1 > 0) & (m_a2 > 0) &
                                (m_b1 > 0) & (m_b2 > 0)))
            kinds = self.data.data['cov'].get('NG_terms',
                                              ['1h'])
            for kind in kinds:
                cov_NG += self.get_covariance_ng_halomodel(s_a1, s_a2,
                                                           s_b1, s_b2,
                                                           fsky, kind)

        itime = time.time()
        self.cov = cov_G + cov_nlm + cov_mm + cov_NG
        ftime = time.time()
        print('Added all covariances terms. It took ' +
              f'{(ftime - itime) / 60} min', flush=True)

        itime = time.time()
        np.savez_compressed(fname, cov=self.cov, cov_G=cov_G, cov_NG=cov_NG,
                            cov_nl_marg=cov_nlm, cov_m_marg=cov_mm)
        ftime = time.time()
        print(f'Saved cov npz file. It took {(ftime - itime) / 60} min',
              flush=True)
        self.recompute_cov = False
        return self.cov

    def get_covariance_ng_halomodel(self, s_a1, s_a2, s_b1, s_b2,
                                    fsky, kind='1h'):
        ellA = self.clA1A2.b.get_effective_ells()
        ellB = self.clB1B2.b.get_effective_ells()
        nclsa = np.max([1, s_a1 + s_a2])
        nclsb = np.max([1, s_b1 + s_b2])
        cov = np.zeros([ellA.size, nclsa, ellB.size, nclsb])
        th = Theory(self.data.data)
        mpA1, mpA2 = self.clA1A2.get_mappers()
        mpB1, mpB2 = self.clB1B2.get_mappers()
        trlist = self.data.data['tracers']
        ccl_trA1 = th.compute_tracer_ccl(self.trA1,
                                         trlist[self.trA1],
                                         mpA1)
        bA1 = self.data.get_bias(self.trA1)
        ccl_trA2 = th.compute_tracer_ccl(self.trA2,
                                         trlist[self.trA2],
                                         mpA2)
        bA2 = self.data.get_bias(self.trA2)
        ccl_trB1 = th.compute_tracer_ccl(self.trB1,
                                         trlist[self.trB1],
                                         mpB1)
        bB1 = self.data.get_bias(self.trB1)
        ccl_trB2 = th.compute_tracer_ccl(self.trB2,
                                         trlist[self.trB2],
                                         mpB2)
        bB2 = self.data.get_bias(self.trB2)
        covNG = th.get_ccl_cl_covNG(ccl_trA1, ccl_trA2, ellA,
                                    ccl_trB1, ccl_trB2, ellB,
                                    fsky, kind=kind)
        # NG covariances can only be calculated for E-modes
        cov[:, 0, :, 0] = covNG*bA1*bA2*bB1*bB2
        return cov.reshape([ellA.size*nclsa, ellB.size*nclsb])

    def get_covariance_nl_marg(self):
        _, nl = self.clA1A2.get_ell_nl()
        nl = nl.flatten()
        if (self.trA1 == self.trA2 == self.trB1 == self.trB2):
            cov = self.nl_prior**2 * (nl[:, None] * nl[None, :])
        else:
            cov = np.zeros((nl.size, nl.size))

        return cov

    def get_covariance_m_marg(self):
        _, cla1a2 = self.clfid_A1A2.get_ell_cl()
        _, clb1b2 = self.clfid_B1B2.get_ell_cl()
        # Window convolution only needed if computed from theory
        if isinstance(self.clfid_A1A2, ClFid):
            wins_a1a2 = self.clA1A2.get_bandpower_windows()
            ncls_a1a2, nell_cp = cla1a2.shape
            wins_a1a2 = wins_a1a2.reshape((-1, ncls_a1a2 * nell_cp))
            cla1a2 = wins_a1a2.dot(cla1a2.flatten()).reshape((ncls_a1a2, -1))
        if isinstance(self.clfid_B1B2, ClFid):
            wins_b1b2 = self.clB1B2.get_bandpower_windows()
            ncls_b1b2, nell_cp = clb1b2.shape
            wins_b1b2 = wins_b1b2.reshape((-1, ncls_b1b2 * nell_cp))
            clb1b2 = wins_b1b2.dot(clb1b2.flatten()).reshape((ncls_b1b2, -1))

        t_a1, t_a2 = self.clA1A2.get_dtypes()
        t_b1, t_b2 = self.clB1B2.get_dtypes()
        #
        sigma_a1 = sigma_a2 = sigma_b1 = sigma_b2 = 0
        if t_a1 == 'galaxy_shear':
            sigma_a1 = self.data.data['tracers'][self.trA1].get('sigma_m', 0)
        if t_a2 == 'galaxy_shear':
            sigma_a2 = self.data.data['tracers'][self.trA2].get('sigma_m', 0)
        if t_b1 == 'galaxy_shear':
            sigma_b1 = self.data.data['tracers'][self.trB1].get('sigma_m', 0)
        if t_b2 == 'galaxy_shear':
            sigma_b2 = self.data.data['tracers'][self.trB2].get('sigma_m', 0)
        #
        cov = cla1a2.flatten()[:, None] * clb1b2.flatten()[None, :]
        cov *= (sigma_a1 * sigma_b1 + sigma_a1 * sigma_b2 +
                sigma_a2 * sigma_b1 + sigma_a2 * sigma_b2)
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
    parser.add_argument('--m_marg', default=False, action='store_true',
                        help='Compute multiplicative bias marginalization cov')
    parser.add_argument('--nl_marg', default=False, action='store_true',
                        help='Compute noise bias marginalization cov')
    args = parser.parse_args()

    data = Data(data_path=args.INPUT).data
    cov = Cov(data, args.trA1, args.trA2, args.trB1, args.trB2)

    if args.m_marg:
        cov.get_covariance_m_marg()
    elif args.nl_marg:
        cov.get_covariance_nl_marg()
    else:
        cov.get_covariance()
