#!/usr/bin/python
from glob import glob
from ..mappers import mapper_from_name
from warnings import warn
import yaml
import os
import shutil


class Data():
    def __init__(self, data_path='', data={}, override=False):
        if (data_path) and (data):
            raise ValueError('Only one of data_path or data must be given. \
                             Both set.')
        elif data_path:
            self.data_path = data_path
            self.data = self.read_data(data_path)
        elif data:
            self.data_path = None
            self.data = data
        else:
            raise ValueError('One of data_path or data must be set. \
                             None set.')

        os.makedirs(self.data['output'], exist_ok=True)
        self._check_yml_in_outdir(override)

    def _check_yml_in_outdir(self, override=False):
        outdir = self.data['output']
        fname = os.path.join(outdir, '*.yml')
        files = glob(fname)
        if (len(files) == 1) and (not override):
            warn(f'A YML file was found in outdir: {outdir}. Using it \
                 instead of input config.')
            self.data_path = files[0]
            self.data = self.read_data(files[0])
        elif len(files) > 1:
            raise ValueError(f'More than 1 YML file in outdir: {outdir}.')
        elif ((len(files) == 0) or override) and self.data_path:
            shutil.copy(self.data_path, outdir)
        else:
            fname = os.path.join(outdir, 'data.yml')
            with open(fname, 'w') as f:
                yaml.dump(self.data, f)

    def read_data(self, data_path):
        with open(data_path) as f:
            data = yaml.safe_load(f)
        return data

    def get_tracers_used(self, wsp=False):
        tracers = []
        for trk, trv in self.data['cls'].items():
            tr1, tr2 = trk.split('-')
            if trv['compute'] != 'None':
                tracers.append(tr1)
                tracers.append(tr2)

        tracers_for_cl = []
        for tr in self.data['tracers'].keys():
            # Remove tracer number
            tr_nn = self.get_tracer_bare_name(tr)
            if tr_nn in tracers:
                tracers_for_cl.append(tr)

        if wsp:
            tracers_for_cl = self.filter_tracers_wsp(tracers_for_cl)

        return tracers_for_cl

    def get_tracer_bare_name(self, tr):
        if '__' in tr:
            tr = ''.join(tr.split('__')[:-1])
        return tr

    def _get_pair_reqs(self, tn1, tn2):
        tb1 = self.get_tracer_bare_name(tn1)
        tb2 = self.get_tracer_bare_name(tn2)
        pname = f'{tb1}-{tb2}'
        pname_inv = f'{tb1}-{tb2}'
        compute = False
        clcov_from_data = False

        # Find if we want to compute this C_ell
        if pname_inv in self.data['cls']:
            pname = pname_inv
        if pname in self.data['cls']:
            comp = self.data['cls'][pname]['compute']
            if ((comp == 'all') or
                    ((comp == 'auto') and (tn1 == tn2))):
                compute = True

        # Find if this pair's C_ell should be computed
        # from data for the covariance
        clcovlist = self.data['cov'].get('cls_from_data', None)
        if clcovlist:
            if isinstance(clcovlist, str):
                if clcovlist == 'all':
                    # Should all cls be computed from data?
                    clcov_from_data = True
            elif isinstance(clcovlist, list):
                if f'{tn1}-{tn2}' in clcovlist:
                    # Should this particular cl be computed from data?
                    clcov_from_data = True
                if f'{tn2}-{tn1}' in clcovlist:
                    # Should this particular cl be computed from data?
                    clcov_from_data = True
            elif isinstance(clcovlist, dict):
                if pname_inv in clcovlist:
                    pname = pname_inv
                if pname in clcovlist:
                    # Out of this group, should this cl be comp. from data?
                    comp = clcovlist[pname]['compute']
                    if ((comp == 'all') or
                            ((comp == 'auto') and (tn1 == tn2))):
                        clcov_from_data = True
        return {'compute': compute,
                'clcov_from_data': clcov_from_data}

    def get_tracer_matrix(self):
        tr_list = list(self.data['tracers'].keys())
        tr_matrix = {}
        for tn1 in tr_list:
            for tn2 in tr_list:
                tr_matrix[(tn1, tn2)] = self._get_pair_reqs(tn1, tn2)
        return tr_matrix

    def get_tracers_bare_name_pair(self, tr1, tr2, connector='-'):
        tr1_nn = self.get_tracer_bare_name(tr1)
        tr2_nn = self.get_tracer_bare_name(tr2)
        trreq = connector.join([tr1_nn, tr2_nn])
        return trreq

    def get_cl_trs_names(self, wsp=False):
        cl_tracers = []
        tr_names = self.get_tracers_used(wsp)
        for i, tr1 in enumerate(tr_names):
            for tr2 in tr_names[i:]:
                trreq = self.get_tracers_bare_name_pair(tr1, tr2)
                trreq_inv = self.get_tracers_bare_name_pair(tr2, tr1)
                if (trreq not in self.data['cls']) and \
                   (trreq_inv in self.data['cls']):
                    # If the inverse is in the data file, reverse it
                    # internally
                    self.data['cls'][trreq] = self.data['cls'][trreq_inv]
                    del self.data['cls'][trreq_inv]
                elif (trreq not in self.data['cls']) and \
                     (trreq_inv not in self.data['cls']):
                    # If still not present, skip
                    continue
                clreq = self.data['cls'][trreq]['compute']
                if clreq == 'all':
                    pass
                elif (clreq == 'auto') and (tr1 != tr2):
                    continue
                elif clreq == 'None':
                    continue
                cl_tracers.append((tr1, tr2))

        return cl_tracers

    def get_cov_trs_names(self, wsp=False):
        cl_tracers = self.get_cl_trs_names(wsp)
        cov_tracers = []
        for i, trs1 in enumerate(cl_tracers):
            for trs2 in cl_tracers[i:]:
                cov_tracers.append((*trs1, *trs2))

        return cov_tracers

    def get_cov_ng_cl_tracers(self):
        cl_tracers = self.get_cl_trs_names()
        order_ng = self.data['cov']['ng']['order']
        cl_ng = [[] for i in order_ng]
        ix_reverse = []

        for tr1, tr2 in cl_tracers:
            tr1_nn = self.get_tracer_bare_name(tr1)
            tr2_nn = self.get_tracer_bare_name(tr2)
            if (tr1_nn + '-' + tr2_nn) in order_ng:
                ix = order_ng.index(tr1_nn + '-' + tr2_nn)
            elif (tr2_nn + '-' + tr1_nn) in order_ng:
                ix = order_ng.index(tr2_nn + '-' + tr1_nn)
                if ix not in ix_reverse:
                    ix_reverse.append(ix)
            else:
                # raise ValueError(f'Tracers {tr1}-{tr2} not found in NG cov.')
                warn(f'Tracers {tr1}-{tr2} not found in NG cov.')
                continue
            cl_ng[ix].append((tr1, tr2))

        for ix in ix_reverse:
            cl_ng[ix].sort(key=lambda x: x[1])

        return [item for sublist in cl_ng for item in sublist]

    def filter_tracers_wsp(self, tracers):
        tracers_torun = []
        masks = []
        for tr in tracers:
            mtr = self.data['tracers'][tr]['mask_name']
            if mtr not in masks:
                tracers_torun.append(tr)
                masks.append(mtr)

        return tracers_torun

    def check_toeplitz(self, dtype):
        if ('toeplitz' in self.data) and (dtype in self.data['toeplitz']):
            toeplitz = self.data['toeplitz'][dtype]

            l_toeplitz = toeplitz['l_toeplitz']
            l_exact = toeplitz['l_exact']
            dl_band = toeplitz['dl_band']
        else:
            l_toeplitz = l_exact = dl_band = -1

        return l_toeplitz, l_exact, dl_band

    def get_mapper(self, tr):
        config = self.data['tracers'][tr]
        mapper_class = config['mapper_class']
        return mapper_from_name(mapper_class)(config)

    def read_symmetric(self, tr1, tr2):
        trs_names = self.get_cl_trs_names()
        trs_used = self.get_tracers_used()
        # Check if the symmetric Cell is requested or if
        # it's a Cell for the covariance, to compute it keeping the order in
        # the yaml file, to avoid duplications
        if ((tr1, tr2) not in trs_names) and \
           (((tr2, tr1) in trs_names) or
           trs_used.index(tr1) > trs_used.index(tr2)):
            return True
        else:
            return False
