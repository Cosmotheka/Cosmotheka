#!/usr/bin/python
from glob import glob
from ..mappers import mapper_from_name
from warnings import warn
import yaml
import os
import shutil
import numpy as np


class CustomLoader(yaml.SafeLoader):
    """
    Custom yaml.SafeLoader class that allows to populate the yaml by reading
    other by writing !include path_to_other_yaml
    """
    # This was copied from https://stackoverflow.com/questions/528281
    def __init__(self, stream):
        super(CustomLoader, self).__init__(stream)

    def include(self, node):
        filename = self.construct_scalar(node)

        with open(filename, 'r') as f:
            return yaml.load(f, CustomLoader)


CustomLoader.add_constructor('!include', CustomLoader.include)


class Data():
    """
    Data class. This is in charge of reading the configuration yaml file or
    dictionary.
    """
    def __init__(self, data_path='', data={}, override=False,
                 ignore_existing_yml=False):
        """
        Parameters
        ----------
        data_path: string
            The path to the configuration yaml file
        data: dict
            The loaded configuration. Only one of data_path or data can be
            given
        override: bool
            If True, override existing yaml in output directory.
        ignore_existing_yml: bool
            If True, ignore existing yaml in the output directory and use the
            input configuration. Otherwise, use the existing yaml.

        Raises
        ------
        ValueError
            If both data_path and data are given

        """
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
        # Bellow define dictionaries with keys:
        # 'all': for lists of all considered cases (tracers, pairs of tracers,
        # etc.)
        # 'wsp': for the minimal subset that will produce the relevant
        # workspaces. In the case of self.tracers this is the subset with
        # different masks.
        # tracers are the tracer used
        self.tracers = {'wsp': None, 'all': None}
        # cl_tracers are pair of tracers used to compute the cl
        self.cl_tracers = {'wsp': None, 'all': None}
        # cov_tracers are tuples of 4 tracers
        self.cov_tracers = {'wsp': None, 'all': None}
        self.tr_matrix = None

        # What the code will do if not specified
        self.default_cls_to_compute = 'None'
        self.default_clcov_from_data = 'None'

    def _get_tracers_defined(self):
        """
        Get the tracers listed in the configuration file (that might or might
        not be used).

        Returns
        -------
        tracers: list
            List of the tracers listed in the configuration file
        """
        trs_section = self._get_section('tracers')
        return list(trs_section.keys())

    def _get_section(self, section):
        """
        Get the data of a given section of the configuration file.

        Parameters
        ----------
        section: string
            Name of the section whose content you want to get.

        Returns
        -------
        content: dictionary
            Dictionary with the content of the given section. If it does not
            exist in the configuration, it returns an empty dictionary
        """
        return self.data.get(section, {})

    def _map_compute_to_bool_for_trs(self, tr1, tr2, value):
        """
        Return True or False if the Cell of a pair of tracers is requested or
        not.

        Parameters
        ----------
        tr1: str
            Name of the first tracer
        tr2: str
            Name of the second tracer
        value: str
            If the Cell has to be computed. Possible values are: 'None' or
            'none' if not; 'auto' if only auto-correlations are requested;
            or 'all' if all correlations are requested.

        Returns
        -------
        compute: bool
            True if the Cell of the pair has to be computed or False if not.
        """
        auto = tr1 == tr2
        value = value.lower()
        if (value == 'none') or ((value == 'auto') and not auto):
            compute = False
        elif ((value == 'auto') and auto) or (value == 'all'):
            compute = True
        else:
            raise ValueError("Compute value = {value} not understood. It " +
                             "has to be one of 'all', 'None' or 'auto'")

        return compute

    def _init_tracer_matrix(self):
        """
        Return the matrix with the information of the tracer pairs. If the Cell
        has to be computed, if the Cell from data has to be used for the
        covariance and if the pair of tracers has to be swapped when computing
        the Cell.

        Returns
        -------
        matrix: dict
            Dictionary with keys a tuple of a pair of tracers; i.e.(tr1, tr2)
            and value a dictionary with the following keys:

            - 'compute': if the Cell has to be computed
            - 'clcov_from_data': if the data Cell has to be used for the
              covariance
            - 'inv': if the tracers order has to be swapped when computing the
              Cell.
        """
        trs = self._get_tracers_defined()

        # Default values should be 0, 1, 2
        compute_matrix, compute_default = \
            self._get_requested_survey_cls_matrix()
        clcov_from_data_matrix, clcov_fdata_default = \
            self._get_clcov_from_data_matrix()

        tr_matrix = {}
        for i, tr1 in enumerate(trs):
            for j, tr2 in enumerate(trs):
                tr1_nn = self.get_tracer_bare_name(tr1)
                tr2_nn = self.get_tracer_bare_name(tr2)
                survey_comb = (tr1_nn, tr2_nn)

                cmi = compute_matrix.get(survey_comb, compute_default)
                clcov_mi = clcov_from_data_matrix.get(survey_comb,
                                                      clcov_fdata_default)
                # clcov_from_data_matrix can have keys as (survey1, survey2)
                # or (tracer1, tracer2)
                # If clcov_mi has the default value, try using the other
                # keys
                if clcov_mi == clcov_fdata_default:
                    clcov_mi = clcov_from_data_matrix.get((tr1, tr2),
                                                          clcov_fdata_default)
                compute = \
                    self._map_compute_to_bool_for_trs(tr1, tr2, cmi)

                clcov_from_data = \
                    self._map_compute_to_bool_for_trs(tr1, tr2, clcov_mi)

                tr_matrix[(tr1, tr2)] = {'compute': compute,
                                         'clcov_from_data': clcov_from_data,
                                         'inv': j < i}
        return tr_matrix

    def _get_clcov_from_data_matrix(self, return_default=True):
        """
        Return the matrix of tracer pairs with the information if the Cell for
        the covariance has to be computed from data or not.

        Parameters
        ----------
        return_default: bool
            If True, this method returns the default computation value for the
            Cells

        Returns
        -------
        matrix: dict
            Dictionary with keys a tuple of a pair of tracers; i.e.(tr1, tr2)
            and value the computation value requested (or the default if not
            specified); i.e. one of 'None' or 'none', 'all', 'auto'.

        default: str, (if return_default is True)
            Default value for the cases not specified in the configuration.
        """
        conf = self._get_section('cov').get('cls_from_data', {})
        # Backwards compatibility
        if isinstance(conf, str):
            combs = []
            default = conf
        elif isinstance(conf, list):
            # The default here is the one will be used when constructing the
            # info matrix. This default is not used in the for loop and 'all'
            # where automatically assigned for all non-dictionary instances.
            combs = conf
            default = self.default_clcov_from_data
        else:
            default = conf.get('default', self.default_clcov_from_data)
            combs = conf.keys()

        survey_matrix = {}
        for c in combs:
            # Skip entries that are not a combination of pair of surveys
            if '-' not in c:
                continue
            s1, s2 = c.split('-')
            if isinstance(conf, dict):
                val = conf[c].get('compute', default)
            else:
                # If they're a list, note that s1, s2 can be tracer names,
                # instead of survey names
                val = 'all'
            survey_matrix[(s1, s2)] = val
            survey_matrix[(s2, s1)] = val

        if return_default:
            return survey_matrix, default

        return survey_matrix

    def _get_requested_survey_cls_matrix(self, return_default=True):
        """
        Return the matrix of tracer pairs with the information if the Cell of
        given pair has to be computed or not.

        Parameters
        ----------
        return_default: bool
            If True, this method returns the default computation value for the
            Cells

        Returns
        -------
        matrix: dict
            Dictionary with keys a tuple of a pair of tracers; i.e.(tr1, tr2)
            and value the computation value requested (or the default if not
            specified); i.e. one of 'None' or 'none', 'all', 'auto'.

        default: str, (if return_default is True)
            Default value for the cases not specified in the configuration.
        """
        cls_conf = self._get_section('cls')
        default = self.default_cls_to_compute
        if isinstance(cls_conf, str):
            survey_matrix = self._load_survey_cls_matrix(cls_conf)
        elif 'file' in cls_conf:
            survey_matrix = self._load_survey_cls_matrix(cls_conf['file'])
            default = cls_conf.get('default', default)
        else:
            survey_matrix = self._read_cls_section_matrix()
            default = cls_conf.get('default', default)

        if return_default:
            return survey_matrix, default

        return survey_matrix

    def _load_survey_cls_matrix(self, fname):
        """
        Read a npz file and return the matrix of tracer pairs with the
        information if the Cell of given pair has to be computed or not.

        Parameters
        ----------
        fname: str
            Path to the npz file storing the matrix of tracer pairs with the
            information if the Cell has to be computed or not. The elements of
            the matrix are integers (2 for 'all'; 1 for 'auto', '0' for
            'none').

        Returns
        -------
        matrix: dict
            Dictionary with keys a tuple of a pair of tracers; i.e.(tr1, tr2)
            and value the computation value requested (or the default if not
            specified); i.e. one of 'None' or 'none', 'all', 'auto'.
        """
        clsf = np.load(fname)
        surveys = clsf['surveys']
        survey_matrix = clsf['cls_matrix']

        # The matrix will have values 0, 1, 2 and we will map them to the human
        # friendly 'all', 'auto', 'None'
        cls_legend = {2: 'all', 1: 'auto', 0: 'None'}

        matrix = {}
        for i, s1 in enumerate(surveys):
            for j, s2 in enumerate(surveys):
                matrix[(s1, s2)] = cls_legend[survey_matrix[i, j]]

        return matrix

    def _read_cls_section_matrix(self):
        """
        Read the 'cls' section of the configuration file and return a matrix of
        tracer pairs with the information if the Cell of given pair has to be
        computed or not. It can be a subset of the full matrix.

        Returns
        -------
        matrix: dict
            Dictionary with keys a tuple of a pair of tracers; i.e.(tr1, tr2)
            and value the computation value requested (or the default if not
            specified); i.e. one of 'None' or 'none', 'all', 'auto'.
        """
        cls_conf = self._get_section('cls')
        combs = cls_conf.keys()
        survey_matrix = {}
        for c in combs:
            if c == "default":
                continue
            s1, s2 = c.split('-')
            val = cls_conf[c]['compute']
            survey_matrix[(s1, s2)] = val
            survey_matrix[(s2, s1)] = val

        return survey_matrix

    def get_bias(self, tracer):
        """
        Get the linear galaxy bias assotiated with the input tracer

        Parameters
        ----------
        tracer: string
            Tracer name

        Returns
        -------
        bias: float
            Linear galaxy bias
        """
        return self.data['tracers'][tracer].get('bias', 1.)

    def _check_yml_in_outdir(self, override=False, ignore_existing_yml=False):
        """
        Check if there is a configuration yaml file in the output directory.

        Parameters
        ----------
        override: bool
            If True, override existing yaml in output directory.

        ignore_existing_yml: bool
            If True, ignore existing yaml in the output directory and use the
            input configuration.

        Raises
        ------
        ValueError:
            If both override and ignore_existing_yml are True or if there is
            more than 1 yaml file in the output directory.
        """
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
        """
        Write the loaded configuration to a yaml file called 'data.yml' in the
        output directory
        """
        outdir = self.data['output']
        fname = os.path.join(outdir, 'data.yml')
        with open(fname, 'w') as f:
            yaml.dump(self.data, f)

    def read_data(self, data_path):
        """
        Read the configuration yaml file.

        Parameters
        ----------
        data_path: str
            Path to the configuration yaml file
        """
        with open(data_path) as f:
            data = yaml.load(f, CustomLoader)
        return data

    def get_tracers_used(self, wsp=False):
        """
        Return the tracers used to compute the requested Cell.

        Parameters
        ----------
        wsp: bool
            If True, return only the tracers with different masks

        Return
        ------
        tracers: list
            List of tracers used to compute the requested Cell.

        """
        lab = ['all', 'wsp'][wsp]
        if self.tracers[lab] is None:
            # Get all tracers in the yaml file
            tr_names = self._get_tracers_defined()
            # Get info matrix: this will let us know if a cl is computed or not
            tmat = self.get_tracer_matrix()

            # Tracers now will contain only the tracers for which we actually
            # compute C_ells.
            # Saving only tr1 to keep order
            tracers = []
            for tr1 in tr_names:
                for tr2 in tr_names:
                    req = tmat[(tr1, tr2)]
                    if req['compute'] and (tr1 not in tracers):
                        tracers.append(tr1)
                        break
            if wsp:
                # We only keep those with different masks
                tracers = self._filter_tracers_wsp(tracers)

            self.tracers[lab] = tracers

        return self.tracers[lab]

    def get_tracer_bare_name(self, tracer):
        """
        Return the tracer name without the tomographic index (e.g. for
        DES__0, it returns DES)

        Parameters
        ----------
        tracer: str
            Tracer name

        Return
        ------
        tracer_bare_name: str
            Tracer name without the tomographic index
        """
        if '__' in tracer:
            tracer = ''.join(tracer.split('__')[:-1])
        return tracer

    def will_pair_be_computed(self, tn1, tn2):
        """
        Return the ordered tracer pair for which the Cell will be computed. If
        the Cell will not be computed, return None

        Parameters
        ----------
        tn1: str
            First tracer's name
        tn2: str
            Second tracer's name

        Return
        ------
        tuple or None
            Tupe of the ordered tracer pair for which the Cell will be
            computed. If the Cell will not be computed, return None

        """
        tmat = self.get_tracer_matrix()
        t = tmat[(tn1, tn2)]
        if t['compute']:
            if t['inv']:
                return (tn2, tn1)
            else:
                return (tn1, tn2)
        return None

    def get_tracer_matrix(self):
        """
        Return the matrix with the information of the tracer pairs. If the Cell
        has to be computed, if the Cell from data has to be used for the
        covariance and if the pair of tracers has to be swapped when computing
        the Cell.

        Return
        ------
        matrix: dict
            Dictionary with keys a tuple of a pair of tracers; i.e.(tr1, tr2)
            and value a dictionary with the following keys:

            - 'compute': if the Cell has to be computed
            - 'clcov_from_data': if the data Cell has to be used for the
              covariance
            - 'inv': if the tracers order has to be swapped when computing the
              Cell.
        """
        if self.tr_matrix is None:
            self.tr_matrix = self._init_tracer_matrix()
        return self.tr_matrix

    def get_tracers_bare_name_pair(self, tr1, tr2, connector='-'):
        """
        Return the tracer names without the tomographic index (e.g. for
        DES__0, it returns DES) joined by a connector.

        Parameters
        ----------
        tr1: str
            First tracer's name
        tr2: str
            Second tracer's name
        connector: str
            Connector to join tr1 and tr2's bare names

        Return
        ------
        str
            First (tr1) and second (tr2) names without the tomographic index
            (e.g. for DES__0, it returns DES) joined by a connector.
        """
        tr1_nn = self.get_tracer_bare_name(tr1)
        tr2_nn = self.get_tracer_bare_name(tr2)
        trreq = connector.join([tr1_nn, tr2_nn])
        return trreq

    def get_cl_trs_names(self, wsp=False):
        """
        Return a list of pair of tracers for which to compute the Cell.

        Parameters
        ----------
        wsp: bool
            If True, return the minimal subset of tracers pairs
            to compute all the needed workspaces. Default False.

        Returns
        -------
        cl_tracers: list
            List of pair of tracers for which to compute the Cell.
        """
        lab = ['all', 'wsp'][wsp]
        if self.cl_tracers[lab] is None:
            cl_tracers = []
            tr_names = self.get_tracers_used(wsp)
            tmat = self.get_tracer_matrix()

            for tr1 in tr_names:
                for tr2 in tr_names:
                    req = tmat[(tr1, tr2)]
                    if req['inv'] or (req['compute'] is False):
                        continue
                    cl_tracers.append((tr1, tr2))

            self.cl_tracers[lab] = cl_tracers

        return self.cl_tracers[lab]

    def get_cov_trs_names(self, wsp=False):
        """
        Return a list of tuples with four tracers for which to compute the
        covariance matrix.

        Parameters
        ----------
        wsp: bool
            If True, return the minimal subset of 4 tracers tuples to compute
            all the needed covariance workspaces. Default False.

        Returns
        -------
        cov_tracers: list
            List of tuples of 4 tracers with the tracers to compute a
            covariance block.
        """
        lab = ['all', 'wsp'][wsp]
        if self.cov_tracers[lab] is None:
            cl_tracers = self.get_cl_trs_names(wsp)
            cov_tracers = []
            for i, trs1 in enumerate(cl_tracers):
                for trs2 in cl_tracers[i:]:
                    cov_tracers.append((*trs1, *trs2))

            self.cov_tracers[lab] = cov_tracers
        return self.cov_tracers[lab]

    def get_cov_extra_cl_tracers(self):
        """
        Return a list of pair of tracers in the order that the extra covariance
        has been built.

        Parameters
        ----------
        wsp: bool
            If True, return the minimal subset of tracers pairs
            to compute all the needed workspaces. Default False.

        Returns
        -------
        cl_tracers: list
            List of pair of tracers in the order that the extra covariance
            has been built.
        """
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

    def _filter_tracers_wsp(self, tracers):
        """
        Return the subset of the input tracers with different masks.

        Parameters
        ----------
        tracers: list
            List of tracers (they have to be defined in the configuration
            file).

        Returns
        -------
        tracers_wsp: list
            List of tracers with different masks.
        """
        tracers_torun = []
        masks = []
        for tr in tracers:
            mtr = self.data['tracers'][tr]['mask_name']
            if mtr not in masks:
                tracers_torun.append(tr)
                masks.append(mtr)

        return tracers_torun

    def check_toeplitz(self, dtype):
        """
        Return the Toeplitz approximation parameters (see Fig. 3 of Louis et al
        2020 (arXiv: 2010.14344)

        Parameters
        ----------
        dtype: list
            What you are approximating with Toeplitz ('cls' or 'cov')

        Returns
        -------
        l_toeplitz: int
             ell_toeplitz in Fig. 3 of that paper. Set to -1 if not specified
             in the configuration file.
        l_exact: int
             ell_exact in Fig. 3 of that paper. Set to -1 if not specified in
             the configuration file.
        dl_band: int
             Delta ell band in Fig. 3 of that paper. Set to -1 if not specified
             in the configuration file.
        """
        if ('toeplitz' in self.data) and (dtype in self.data['toeplitz']):
            toeplitz = self.data['toeplitz'][dtype]

            l_toeplitz = toeplitz['l_toeplitz']
            l_exact = toeplitz['l_exact']
            dl_band = toeplitz['dl_band']
        else:
            l_toeplitz = l_exact = dl_band = -1

        return l_toeplitz, l_exact, dl_band

    def get_mapper(self, tr):
        """
        Return the initialized mapper class of the corresponding input tracer.

        Parameters
        ----------
        tr: str
            Tracer name

        Returns
        -------
        mapper:
            Initialized mapper class of the corresponding input tracer.

        """
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
        """
        Return the ordered version of the input tracers. Not that this is
        completely arbitrary and internally fixed to avoid computing the same
        power spectra twice.

        Parameters
        ----------
        tr1: str
            First tracer's name
        tr1: str
            Second tracer's name

        Returns
        -------
        tracers: tuple
            Tuple with the tracers ordered.

        """
        tmat = self.get_tracer_matrix()
        return tmat[(tr1, tr2)]['inv']
