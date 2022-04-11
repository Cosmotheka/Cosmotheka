import glob
import pytest
import os
import yaml
from xcell.cls.data import Data
from ..mappers import mapper_from_name


def read_yaml_file(file):
    with open(file) as f:
        config = yaml.safe_load(f)
    return config


def get_input_file():
    return './xcell/tests/data/desy1_ebossqso_p18cmbk.yml'


def get_config_dict():
    input_file = get_input_file()
    config = read_yaml_file(input_file)
    return config


def get_data():
    input_file = get_input_file()
    return Data(input_file)


def get_data_from_dict():
    input_dict = get_config_dict()
    return Data(data=input_dict)


def remove_yml_file(config):
    outdir = config['output']
    fname = os.path.join(outdir, '*.yml')
    for f in glob.glob(fname):
        os.remove(f)


def remove_outdir(config):
    outdir = config['output']
    if outdir != './xcell/tests/cls/':
        raise ValueError('Outdir looks odd. Not removing it as precaution')
    # Using os.rmdir as it will only remove an empty directory
    elif os.path.isdir(outdir):
        os.rmdir(outdir)


def get_tracer_pair_iterator(data):
    tlist = list(data.data['tracers'].keys())
    for t1 in tlist:
        for t2 in tlist:
            yield t1, t2


def test_input_from_another_file():
    # This tests whether we can load yaml files that
    # include other yaml files within them.
    d = Data('xcell/tests/data/conftest.yml')
    assert d.data['tracers']['DESgc__1']['bias'] == 1.76
    remove_yml_file(d.data)


def test_will_be_computed():
    d = get_data()
    n1 = d.will_pair_be_computed('DESwl__0', 'DESwl__1')
    n2 = d.will_pair_be_computed('DESwl__1', 'DESwl__0')
    n3 = d.will_pair_be_computed('DESwl__0', 'eBOSS__0')
    assert n1 == ('DESwl__0', 'DESwl__1')
    assert n2 == ('DESwl__0', 'DESwl__1')
    assert not n3


def test_get_tracer_matrix():
    # No cls from data
    c = get_config_dict()
    data = Data(data=c)
    m = data.get_tracer_matrix()
    for t1, t2 in get_tracer_pair_iterator(data):
        assert not m[(t1, t2)]['clcov_from_data']
    remove_yml_file(c)

    # All cls from data
    c = get_config_dict()
    c['cov']['cls_from_data'] = 'all'
    data = Data(data=c)
    m = data.get_tracer_matrix()
    for t1, t2 in get_tracer_pair_iterator(data):
        assert m[(t1, t2)]['clcov_from_data']
    remove_yml_file(c)

    # Group cls from data (all)
    c = get_config_dict()
    c['cov']['cls_from_data'] = {'DESgc-DESgc': {'compute': 'all'}}
    data = Data(data=c)
    m = data.get_tracer_matrix()
    for t1, t2 in get_tracer_pair_iterator(data):
        tpair = data.get_tracers_bare_name_pair(t1, t2)
        if tpair == 'DESgc-DESgc':
            assert m[(t1, t2)]['clcov_from_data']
        else:
            assert not m[(t1, t2)]['clcov_from_data']
    remove_yml_file(c)

    # Group cls from data (auto)
    c = get_config_dict()
    c['cov']['cls_from_data'] = {'DESgc-DESgc': {'compute': 'auto'}}
    data = Data(data=c)
    m = data.get_tracer_matrix()
    for t1, t2 in get_tracer_pair_iterator(data):
        tpair = data.get_tracers_bare_name_pair(t1, t2)
        if tpair == 'DESgc-DESgc' and t1 == t2:
            assert m[(t1, t2)]['clcov_from_data']
        else:
            assert not m[(t1, t2)]['clcov_from_data']
    remove_yml_file(c)

    # Some cls from data
    c = get_config_dict()
    c['cov']['cls_from_data'] = ['DESgc__0-DESgc__0', 'DESgc__1-DESwl__1']
    data = Data(data=c)
    m = data.get_tracer_matrix()
    for t1, t2 in get_tracer_pair_iterator(data):
        if t1 == 'DESgc__0' and t2 == 'DESgc__0':
            assert m[(t1, t2)]['clcov_from_data']
        elif t1 == 'DESgc__1' and t2 == 'DESwl__1':
            assert m[(t1, t2)]['clcov_from_data']
        elif t1 == 'DESwl__1' and t2 == 'DESgc__1':
            assert m[(t1, t2)]['clcov_from_data']
        else:
            assert not m[(t1, t2)]['clcov_from_data']
    remove_yml_file(c)


def test_initizalization():
    input_file = get_input_file()
    config = read_yaml_file(input_file)
    # Check that inputting both a file path and config dictionary rieses an
    # error
    with pytest.raises(ValueError):
        Data(input_file, data=config)

    config['bpw_edges'] = [0, 10]
    data2 = Data(data=config, override=True)
    assert os.path.isfile('xcell/tests/cls/data.yml')
    assert data2.data['bpw_edges'] == [0, 10]
    remove_yml_file(config)

    # Check ignore_existing_yml
    data = get_data()
    data2 = Data(data=config, ignore_existing_yml=True)
    assert data2.data['bpw_edges'] != data.data['bpw_edges']
    remove_yml_file(config)

    # Check error is rised when override and ignore_existing_yml are True
    with pytest.raises(ValueError):
        data2 = Data(data=config, ignore_existing_yml=True, override=True)


def test_read_data():
    data = get_data()
    input_file = get_input_file()
    config = get_config_dict()
    assert data.read_data(input_file) == config

    remove_yml_file(config)


def test_read_saved_data():
    data = get_data()
    config = get_config_dict()
    config['bpw_edges'] = [0, 10]
    # Since there's a yml saved in outdir, it will read it instead of config
    data2 = Data(data=config)
    assert data.data == data2.data
    remove_yml_file(data.data)


def test_get_tracers_used():
    data = get_data()
    config = data.data
    tracers_for_cl = data.get_tracers_used()

    # In the xcell/tests/data/desy1_ebossqso_p18cmbk.yml file no eBOSS tracer
    # is used.
    trs = ['DESgc', 'DESwl', 'PLAcv']
    tracers_for_cl_test = []
    for tr in config['tracers']:
        if tr.split('__')[0] in trs:
            tracers_for_cl_test.append(tr)

    assert tracers_for_cl == tracers_for_cl_test
    assert data.get_tracers_used(True) == ['DESgc__0', 'DESwl__0', 'DESwl__1',
                                           'DESwl__2', 'DESwl__3', 'PLAcv']
    remove_yml_file(data.data)


def test_get_tracer_bare_name():
    data = get_data()
    config = data.data

    for tr in config['tracers']:
        bname = data.get_tracer_bare_name(tr)
        assert tr.split('__')[0] == bname
    remove_yml_file(data.data)


@pytest.mark.parametrize('wsp', [True, False])
def test_get_cl_trs_names(wsp):
    data = get_data()
    config = data.data

    cl_trs = data.get_cl_trs_names(wsp)

    cl_trs_test = []
    trs = data.get_tracers_used(wsp)
    for i, tri in enumerate(trs):
        for trj in trs[i:]:
            bn1 = data.get_tracer_bare_name(tri)
            bn2 = data.get_tracer_bare_name(trj)
            key = '-'.join([bn1, bn2])
            if (config['cls'][key]['compute'] == 'auto') and (tri == trj):
                cl_trs_test.append((tri, trj))
            elif config['cls'][key]['compute'] == 'all':
                cl_trs_test.append((tri, trj))

    assert cl_trs == cl_trs_test

    remove_yml_file(config)


@pytest.mark.parametrize('wsp', [True, False])
def test_get_cov_trs_name(wsp):
    data = get_data()

    cl_trs = data.get_cl_trs_names(wsp)

    cov_trs_test = []
    for i, trsi in enumerate(cl_trs):
        for trsj in cl_trs[i:]:
            cov_trs_test.append((*trsi, *trsj))

    assert cov_trs_test == data.get_cov_trs_names(wsp)

    remove_yml_file(data.data)


def test_get_cov_extra_cl_tracers():
    data = get_data()
    config = data.data
    cl_trs = data.get_cl_trs_names()

    d = {}
    for trs in config['cov']['extra']['order']:
        d[trs] = []

    for trs in cl_trs:
        bn1 = data.get_tracer_bare_name(trs[0])
        bn2 = data.get_tracer_bare_name(trs[1])
        key = '-'.join([bn1, bn2])
        if key in d:
            d[key].append(trs)
        else:
            key = '-'.join([bn2, bn1])
            d[key].append(trs)

    # Order the tracers so that gc0 - wl0, gc1 - wl0, etc.
    d['DESwl-DESgc'].sort(key=lambda x: x[1])

    cl_trs_cov = []
    for trs in config['cov']['extra']['order']:
        cl_trs_cov.extend(d[trs])

    assert cl_trs_cov == data.get_cov_extra_cl_tracers()
    remove_yml_file(config)


def test_filter_tracers_wsp():
    data = get_data()
    config = get_config_dict()

    all_tracers = list(config['tracers'].keys())
    trs_wsp = []
    trs_mask = []
    for tr, val in config['tracers'].items():
        mask_name = val['mask_name']
        if mask_name not in trs_mask:
            trs_wsp.append(tr)
            trs_mask.append(mask_name)

    assert trs_wsp == data.filter_tracers_wsp(all_tracers)
    remove_yml_file(config)


def test_check_toeplitz():
    data = get_data()
    config = get_config_dict()

    assert (-1, -1, -1) == data.check_toeplitz('cls')
    assert (2750, 1000, 2000) == data.check_toeplitz('cov')

    remove_yml_file(config)

    del config['toeplitz']
    data = Data(data=config)
    assert (-1, -1, -1) == data.check_toeplitz('cls')
    assert (-1, -1, -1) == data.check_toeplitz('cov')

    remove_yml_file(config)


def test_get_mapper():
    data = get_data()
    config = get_config_dict()

    data.data['tracers']['DESgc__0']['nside'] = 512

    for tr, val in config['tracers'].items():
        class_name = val['mapper_class']
        if tr == 'DESgc__0':
            with pytest.raises(ValueError):
                m = data.get_mapper(tr)
        else:
            m = data.get_mapper(tr)
            assert m.nside == 4096
            assert isinstance(m, mapper_from_name(class_name))

    data.data['tracers']['DESgc__0'].pop('nside')
    data.data['tracers']['DESgc__0']['coords'] = 'G'

    for tr, val in config['tracers'].items():
        class_name = val['mapper_class']
        if tr == 'DESgc__0':
            with pytest.raises(ValueError):
                m = data.get_mapper(tr)
        else:
            m = data.get_mapper(tr)
            assert m.coords == 'C'
            assert isinstance(m, mapper_from_name(class_name))

    remove_yml_file(config)


def test_read_symmetric():
    data = get_data()
    assert data.read_symmetric('DESwl__3', 'DESgc__0')

    remove_yml_file(data.data)


# Remove outdir
remove_outdir(get_config_dict())
