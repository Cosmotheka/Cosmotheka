#!/usr/bin/python
import yaml

def read_data(data):
    with open(data) as f:
        data = yaml.safe_load(f)
    return data

def get_tracers_used(data, wsp=False):
    tracers = []
    for trk, trv in data['cls'].items():
        tr1, tr2 = trk.split('-')
        if trv != 'None':
            tracers.append(tr1)
            tracers.append(tr2)

    tracers_for_cl = []
    for tr in data['tracers'].keys():
        tr_nn = ''.join(s for s in tr if not s.isdigit())
        if tr_nn in tracers:
            tracers_for_cl.append(tr)

    if wsp:
        tracers_for_cl = filter_tracers_wsp(data, tracers_for_cl)

    return tracers_for_cl

def get_cl_tracers(data, wsp=False):
    cl_tracers = []
    tr_names = get_tracers_used(data, wsp)  # [trn for trn in data['tracers']]
    for i, tr1 in enumerate(tr_names):
        for tr2 in tr_names[i:]:
            trreq = ''.join(s for s in (tr1 + '-' + tr2) if not s.isdigit())
            if trreq not in data['cls']:
                continue
            clreq =  data['cls'][trreq]
            if clreq == 'all':
                pass
            elif (clreq == 'auto') and (tr1 != tr2):
                continue
            elif clreq == 'None':
                continue
            cl_tracers.append((tr1, tr2))

    return cl_tracers

def get_cov_tracers(data, wsp=False):
    cl_tracers = get_cl_tracers(data, wsp)
    cov_tracers = []
    for i, trs1 in enumerate(cl_tracers):
        for trs2 in cl_tracers[i:]:
            cov_tracers.append((*trs1, *trs2))

    return cov_tracers

def get_cov_ng_cl_tracers(data):
    cl_tracers = get_cl_tracers(data)
    order_ng = data['cov']['ng']['order']
    cl_ng = [[] for i in order_ng]
    ix_reverse = []

    for tr1, tr2 in cl_tracers:
        tr1_nn = ''.join(s for s in tr1 if not s.isdigit())
        tr2_nn = ''.join(s for s in tr2 if not s.isdigit())
        if (tr1_nn + '-' + tr2_nn) in order_ng:
            ix = order_ng.index(tr1_nn + '-' + tr2_nn)
        elif (tr2_nn + '-' + tr1_nn) in order_ng:
            ix = order_ng.index(tr2_nn + '-' + tr1_nn)
            if ix not in ix_reverse:
                ix_reverse.append(ix)
        else:
            raise ValueError('Tracers {}-{} not found in NG cov.'.format(tr1, tr2))
        cl_ng[ix].append((tr1, tr2))

    for ix in ix_reverse:
        cl_ng[ix].sort(key=lambda x: x[1])

    return [item for sublist in cl_ng for item in sublist]


def filter_tracers_wsp(data, tracers):
    tracers_torun = []
    masks = []
    for tr in tracers:
        mtr = data['tracers'][tr]['mask']
        if  mtr not in masks:
            tracers_torun.append(tr)
            masks.append(mtr)

    return tracers_torun

def get_dof_tracers(data, tracers):
    tr1, tr2 = tracers
    s1 = data['tracers'][tr1]['spin']
    s2 = data['tracers'][tr2]['spin']

    dof = s1 + s2
    if dof == 0:
        dof += 1

    return dof
