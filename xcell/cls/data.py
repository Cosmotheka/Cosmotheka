#!/usr/bin/python
from glob import glob
from ..mappers import mapper_from_name
from warnings import warn
import yaml
import os
import shutil
import numpy as np


class CustomLoader(yaml.SafeLoader):
    # This was copied from https://stackoverflow.com/questions/528281
    def __init__(self, stream):
        super(CustomLoader, self).__init__(stream)

    def include(self, node):
        filename = self.construct_scalar(node)

        with open(filename, 'r') as f:
            return yaml.load(f, CustomLoader)


CustomLoader.add_constructor('!include', CustomLoader.include)


class Data():
    def __init__(self, data_path='', data={}, override=False,
                 ignore_existing_yml=False):
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
        self._check_yml_in_outdir(override, ignore_existing_yml)
        self.cl_tracers = {'wsp': None, 'no_wsp': None}
        self.cov_tracers = {'wsp': None, 'no_wsp': None}
        self.cls_legend = {'all': 2, 'auto': 1, 'none': 0, 'None': 0}
        self.tr_matrix = None


        # What the code will do if not specified
        self.default_cls_to_compute = 'None'
        self.default_clcov_from_data = 'None'

    def _get_tracers_defined(self):
        trs_section = self._get_section('tracers')
        return trs_section.keys()

    def _get_section(self, section):
        return self.data.get(section, {})

    def _int_to_bool_for_trs(self, tr1, tr2, value):
        auto = tr1 == tr2
        if (value == 0) or ((value == 1) and not auto):
            compute = False
        elif ((value == 1) and auto) or (value > 1):
            compute = True

        return compute

    def _init_tracer_matrix(self):
        trs = self._get_tracers_defined()
        cls_conf = self._get_section('cls')
        cov_conf = self._get_section('cov')

        # Default values should be True or False

        compute_matrix, compute_default = self._get_requested_survey_cls_matrix()
        clcov_from_data_matrix, clcov_from_data_default = \
            self._get_clcov_from_data_matrix()

        tr_matrix = {}
        for i, tr1 in enumerate(trs):
            for j, tr2 in enumerate(trs):
                tr1_nn = self.get_tracer_bare_name(tr1)
                tr2_nn = self.get_tracer_bare_name(tr2)
                survey_comb = (tr1_nn, tr2_nn)

                cmi = compute_matrix.get(survey_comb, compute_default)
                clcov_mi = clcov_from_data_matrix.get(survey_comb,
                                                      clcov_from_data_default)
                # clcov_from_data_matrix can have keys as (survey1, survey2)
                # or (tracer1, tracer2)
                # If clcov_mi has the default value, try using the other
                # keys
                if clcov_mi == clcov_from_data_default:
                    clcov_mi = clcov_from_data_matrix.get((tr1, tr2),
                                                          clcov_from_data_default)


                compute = self._int_to_bool_for_trs(tr1, tr2, cmi)


                clcov_from_data = self._int_to_bool_for_trs(tr1, tr2, clcov_mi)


                tr_matrix[(tr1, tr2)] = {'compute': compute,
                                         'clcov_from_data': clcov_from_data,
                                         'inv': j < i}
        return tr_matrix

    def _get_clcov_from_data_matrix(self, return_default=True):
        conf = self._get_section('cov').get('cls_from_data', {})
        # Backwards compatibility
        if isinstance(conf, str):
            combs = []
            default = self.cls_legend[conf]
        elif isinstance(conf, list):
            combs = conf
            default = self.cls_legend[self.default_clcov_from_data]
        else:
            default = self.cls_legend[conf.get('default',
                                               self.default_clcov_from_data)]
            combs = conf.keys()

        survey_matrix = {}
        for c in combs:
            # Skip entries that are not a combination of pair of surveys
            if '-' not in c:
                continue
            s1, s2 = c.split('-')
            if isinstance(conf, dict):
                val = self.cls_legend[conf[c].get('compute', default)]
            else:
                # If they're a list, note that s1, s2 can be tracer names,
                # instead of survey names
                val = 2
            survey_matrix[(s1, s2)] = val
            survey_matrix[(s2, s1)] = val

        if return_default:
            return survey_matrix, bool(default)

        return survey_matrix

    def _get_requested_survey_cls_matrix(self, return_default=True):
        cls_conf = self._get_section('cls')
        if isinstance(cls_conf, str):
            survey_matrix = self._load_survey_cls_matrix(cls_conf)
            default = bool(self.cls_legend[self.default_cls_to_compute])
        elif 'file' in cls_conf:
            survey_matrix = self._load_survey_cls_matrix(cls_conf['file'])
            default = bool(self.cls_legend[cls_conf.get('default',
                                                        self.default_cls_to_compute)])
        else:
            survey_matrix = self._read_cls_section_matrix()
            default = bool(self.cls_legend[cls_conf.get('default',
                                                        self.default_cls_to_compute)])

        if return_default:
            return survey_matrix, default

        return survey_matrix

    def _load_survey_cls_matrix(self, fname):
            clsf = np.load(fname)
            surveys = clsf['surveys']
            survey_matrix = clsf['cls_matrix']

            matrix = {}
            for i, s1 in enumerate(surveys):
                for j, s2 in enumerate(surveys):
                    matrix[(s1, s2)] = survey_matrix[i, j]

            return matrix

    def _read_cls_section_matrix(self):
        cls_conf = self._get_section('cls')
        combs = cls_conf.keys()
        survey_matrix = {}
        for c in combs:
            s1, s2 = c.split('-')
            val = cls_conf[c]['compute']
            survey_matrix[(s1, s2)] = self.cls_legend[val.lower()]
            survey_matrix[(s2, s1)] = self.cls_legend[val.lower()]

        return survey_matrix

    def get_bias(self, name):
        return self.data['tracers'][name].get('bias', 1.)

    def _check_yml_in_outdir(self, override=False, ignore_existing_yml=False):
        outdir = self.data['output']
        fname = os.path.join(outdir, '*.yml')
        files = glob(fname)
        # Start if-else with sanity checks
        if override and ignore_existing_yml:
            raise ValueError('Only one of override or ignore_existing_yml can '
                             + 'be set.')
        elif len(files) > 1:
            raise ValueError(f'More than 1 YML file in outdir: {outdir}.')
        elif override:
            if len(files):
                warn('Overriding configuration')
            if self.data_path:
                shutil.copy(self.data_path, outdir)
            else:
                self._dump_data()
        elif ignore_existing_yml:
            pass
        elif len(files) == 1:
            warn(f'A YML file was found in outdir: {outdir}. Using it \
                 instead of input config.')
            self.data_path = files[0]
            self.data = self.read_data(files[0])
        elif (len(files) == 0) and self.data_path:
            shutil.copy(self.data_path, outdir)
        else:
            self._dump_data()

    def _dump_data(self):
        outdir = self.data['output']
        fname = os.path.join(outdir, 'data.yml')
        with open(fname, 'w') as f:
            yaml.dump(self.data, f)

    def read_data(self, data_path):
        with open(data_path) as f:
            data = yaml.load(f, CustomLoader)
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

    def will_pair_be_computed(self, tn1, tn2):
        tmat = self.get_tracer_matrix()
        t = tmat[(tn1, tn2)]
        if t['compute']:
            if t['inv']:
                return (tn2, tn1)
            else:
                return (tn1, tn2)
        return None

    def get_tracer_matrix(self):
        if self.tr_matrix is None:
            self.tr_matrix = self._init_tracer_matrix()
        return self.tr_matrix

    def get_tracers_bare_name_pair(self, tr1, tr2, connector='-'):
        tr1_nn = self.get_tracer_bare_name(tr1)
        tr2_nn = self.get_tracer_bare_name(tr2)
        trreq = connector.join([tr1_nn, tr2_nn])
        return trreq

    def _get_cl_trs_names(self, wsp=False):
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

    def get_cl_trs_names(self, wsp=False):
        lab = ['no_wsp', 'wsp'][wsp]
        if self.cl_tracers[lab] is None:
            self.cl_tracers[lab] = self._get_cl_trs_names(wsp)
        return self.cl_tracers[lab]

    def _get_cov_trs_names(self, wsp=False):
        cl_tracers = self.get_cl_trs_names(wsp)
        cov_tracers = []
        for i, trs1 in enumerate(cl_tracers):
            for trs2 in cl_tracers[i:]:
                cov_tracers.append((*trs1, *trs2))

        return cov_tracers

    def get_cov_trs_names(self, wsp=False):
        lab = ['no_wsp', 'wsp'][wsp]
        if self.cov_tracers[lab] is None:
            self.cov_tracers[lab] = self._get_cov_trs_names(wsp)
        return self.cov_tracers[lab]

    def get_cov_extra_cl_tracers(self):
        cl_tracers = self.get_cl_trs_names()
        order_extra = self.data['cov']['extra']['order']
        cl_extra = [[] for i in order_extra]
        ix_reverse = []

        for tr1, tr2 in cl_tracers:
            tr1_nn = self.get_tracer_bare_name(tr1)
            tr2_nn = self.get_tracer_bare_name(tr2)
            if (tr1_nn + '-' + tr2_nn) in order_extra:
                ix = order_extra.index(tr1_nn + '-' + tr2_nn)
            elif (tr2_nn + '-' + tr1_nn) in order_extra:
                ix = order_extra.index(tr2_nn + '-' + tr1_nn)
                if ix not in ix_reverse:
                    ix_reverse.append(ix)
            else:
                warn(f'Tracers {tr1}-{tr2} not found in extra cov.')
                continue
            cl_extra[ix].append((tr1, tr2))

        for ix in ix_reverse:
            cl_extra[ix].sort(key=lambda x: x[1])

        return [item for sublist in cl_extra for item in sublist]

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
        nside = self.data['sphere']['nside']
        coords = self.data['sphere']['coords']
        config['ignore_rerun'] = config.get('ignore_rerun',
                                            self.data.get('ignore_rerun',
                                                          False))
        if 'nside' not in config:
            config['nside'] = nside
        elif config['nside'] != nside:
            raise ValueError(f"Nside mismatch in tracer {tr} and " +
                             f"'sphere': {config['nside']} vs {nside}")
        if 'coords' not in config:
            config['coords'] = coords
        elif config['coords'] != coords:
            raise ValueError(f"Coordinate mismatch in tracer {tr} and " +
                             f"'sphere': {config['coords']} vs {coords}")
        if 'path_rerun' not in config:
            config['path_rerun'] = self.data.get('path_rerun', None)
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
