import numpy as np
import healpy as hp
import os

nside = 1024
mask = hp.ud_grade(hp.read_map("../data/WISExSCOSmask.fits.gz"), nside_out=nside)

nside_jk = 8
npix_jk = hp.nside2npix(nside_jk)
jk_id = hp.ud_grade(range(npix_jk), nside_out=nside)
npix_patch = (nside/nside_jk)**2
fracs = np.array([np.sum(mask[jk_id == jk])/npix_patch for jk in range(npix_jk)])

threshold = 0.5

for i, jk in enumerate(np.arange(npix_jk, dtype=int)[fracs > threshold]):
    print(i, jk)
    dirname = f'../data/JKs/JK_{i}'
    fname_yml = f'{dirname}/yxgxk.yml'
    fname_mask = f'{dirname}/mask_gal.fits.gz'

    os.system(f'mkdir -p {dirname}')
    os.system(f'cp input/yxgxk.yml {fname_yml}')
    os.system(f'sed -i "s#mask: \'../data/WISExSCOSmask.fits.gz\'#mask: \'{fname_mask}\'#g" {fname_yml}')
    os.system(f'sed -i "s#output: \'../data/1024_yxgxk_covmix/\'#output: \'{dirname}\'#g" {fname_yml}')

    msk = mask.copy()
    msk[jk_id == jk] = 0
    hp.write_map(fname_mask, msk, dtype=float, overwrite=True)

    out = ''
    out += '#!/bin/bash\n'
    out += '\n'
    out += f'/usr/bin/python3 run_cls.py {fname_yml} cls --onlogin\n'
    out += f'rm {dirname}/LOWZ_LOWZ/w__mask_LOWZ__mask_LOWZ.fits\n'
    out += f'rm {dirname}/LOWZ_YMILCA/w__mask_LOWZ__mask_YMAP.fits\n'
    out += f'rm {dirname}/LOWZ_KAPPA/w__mask_LOWZ__mask_P18kappa.fits\n'
    out += f'rm {dirname}/YMILCA_YMILCA/w__mask_YMAP__mask_YMAP.fits\n'
    f = open(f'{dirname}/script.sh', 'w')
    f.write(out)
    f.close()
    os.system(f'chmod 755 {dirname}/script.sh')
    os.system(f'addqueue -q cmb -s -c "JK{i}" -n 1x12 -m 2 {dirname}/script.sh')
