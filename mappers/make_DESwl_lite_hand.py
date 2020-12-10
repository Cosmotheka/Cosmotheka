## Executed with:
# addqueue -n 1x10 -m 10 -s -c "mins" -q berg /usr/bin/python3 make_DESwl_lite.py 4
import numpy as np
from astropy.io import fits
from astropy.table import Table


path_data = '/mnt/extraspace/gravityls_3/S8z/data/DES/DESwlMETACAL_catalog_lite.fits'
full_cat = Table.read(path_data, memmap=True)
full_cat.remove_rows(full_cat['zbin_mcal'] != -1)
print(set(full_cat['zbin_mcal']), end=' ', flush=True)
full_cat.write('DESwlMETACAL_catalog_lite.fits')

for i in range(4):
    print(i,  end=' ', flush=True)
    bin_cat = full_cat
    bin_cat.remove_rows(bin_cat['zbin_mcal'] != i) #bins start at -1 for some reason
    bin_cat.write('DESwlMETACAL_catalog_lite_zbin{}.fits'.format(i))
