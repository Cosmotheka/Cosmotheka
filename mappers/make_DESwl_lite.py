## Executed with:
# addqueue -n 1x10 -m 10 -s -c "mins" -q berg /usr/bin/python3 make_DESwl_lite.py 4
import sys
sys.path.append('/mnt/zfsusers/gravityls_3/codes/DEScls/')
import mappers.mapper_DESwl_Carlos as mapper_DESwl_Carlos

nside = 512

path_data = '/mnt/extraspace/damonge/S8z_data/DES_data/shear_catalog/'

config =  {'zbin_cat': path_data + 'y1_source_redshift_binning_v1.fits',
        'data_cat':  path_data + 'mcal-y1a1-combined-riz-unblind-v4-matched.fits',
         'file_nz': path_data + 'y1_redshift_distributions_v1.fits',
         'nside': nside,
         'bin': int(sys.argv[1]),
         'mask_name': 'name'}

mapper_DESwl_Carlos.MapperDESwl(config)
