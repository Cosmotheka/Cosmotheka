import numpy as np
import os
import sacc
from xcell.cls.data import Data

data = Data('../data/JKs/JK_0/yxgxk.yml')
trlist = data.data['tracers']
cl_tracers = data.get_cl_trs_names()

def read_cl_file(fname):
    d = np.load(fname)
    return d['ell'], d['cl'][0]

def gather_cls(predir):
    s = sacc.Sacc()
    for k, t in trlist.items():
        s.add_tracer('Misc', k, quantity='generic', spin=0)

    for tr1, tr2 in cl_tracers:
        pre1 = tr1.split('__')[0]
        pre2 = tr2.split('__')[0]
        dirname = f'{predir}/{pre1}_{pre2}'
        filename = f'{dirname}/cl_{tr1}_{tr2}.npz'
        l, cl = read_cl_file(filename)
        s.add_ell_cl('cl_00', tr1, tr2, l, cl)
    s.save_fits(f'cls_JK{i}.fits', overwrite=True)

for i in range(530):
    print(f"JK{i}")
    try:
        gather_cls(f'../data/JKs/JK_{i}/')
    except:
        print(f"Error reading JK{i}")
